from copy import deepcopy
from typing import Optional

import torch
import torch.nn as nn
from torch import nn
from torch.nn import Linear
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.norm import GraphNorm
from torch_scatter import scatter

from fairchem.core.common.registry import registry
from fairchem.core.common.utils import conditional_grad
from fairchem.core.models.base import BaseModel
from fairchem.core.models.baseline.embedding import EmbeddingBlock
from fairchem.core.models.baseline.force_decoder import ForceDecoder
from fairchem.core.models.baseline.utils import GaussianSmearing, swish


class InteractionBlock(MessagePassing):
    """Updates atom representations through custom message passing."""

    def __init__(
        self,
        hidden_channels,
        num_filters,
        act,
        mp_type,
        complex_mp,
        graph_norm,
    ):
        super(InteractionBlock, self).__init__()

        self.regress_forces = True

        self.act = act
        self.mp_type = mp_type
        self.hidden_channels = hidden_channels
        self.complex_mp = complex_mp
        self.graph_norm = graph_norm
        if graph_norm:
            self.graph_norm = GraphNorm(
                hidden_channels if "updown" not in self.mp_type else num_filters
            )

        if self.mp_type == "simple":
            self.lin_h = nn.Linear(hidden_channels, hidden_channels)

        elif self.mp_type == "updownscale":
            self.lin_geom = nn.Linear(num_filters, num_filters)
            self.lin_down = nn.Linear(hidden_channels, num_filters)
            self.lin_up = nn.Linear(num_filters, hidden_channels)

        elif self.mp_type == "updownscale_base":
            self.lin_geom = nn.Linear(num_filters + 2 * hidden_channels, num_filters)
            self.lin_down = nn.Linear(hidden_channels, num_filters)
            self.lin_up = nn.Linear(num_filters, hidden_channels)

        elif self.mp_type == "updown_local_env":
            self.lin_down = nn.Linear(hidden_channels, num_filters)
            self.lin_geom = nn.Linear(num_filters, num_filters)
            self.lin_up = nn.Linear(2 * num_filters, hidden_channels)

        else:  # base
            self.lin_geom = nn.Linear(
                num_filters + 2 * hidden_channels, hidden_channels
            )
            self.lin_h = nn.Linear(hidden_channels, hidden_channels)

        if self.complex_mp:
            self.other_mlp = nn.Linear(hidden_channels, hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):
        if self.mp_type != "simple":
            nn.init.xavier_uniform_(self.lin_geom.weight)
            self.lin_geom.bias.data.fill_(0)
        if self.complex_mp:
            nn.init.xavier_uniform_(self.other_mlp.weight)
            self.other_mlp.bias.data.fill_(0)
        if self.mp_type in {"updownscale", "updownscale_base", "updown_local_env"}:
            nn.init.xavier_uniform_(self.lin_up.weight)
            self.lin_up.bias.data.fill_(0)
            nn.init.xavier_uniform_(self.lin_down.weight)
            self.lin_down.bias.data.fill_(0)
        else:
            nn.init.xavier_uniform_(self.lin_h.weight)
            self.lin_h.bias.data.fill_(0)

    @conditional_grad(torch.enable_grad())
    def forward(self, h, edge_index, e):
        """Forward pass of the Interaction block.
        Called in FAENet forward pass to update atom representations.

        Args:
            h (tensor): atom embedddings. (num_atoms, hidden_channels)
            edge_index (tensor): adjacency matrix. (2, num_edges)
            e (tensor): edge embeddings. (num_edges, num_filters)

        Returns:
            (tensor): updated atom embeddings
        """
        # Define edge embedding
        if self.mp_type in {"base", "updownscale_base"}:
            e = torch.cat([e, h[edge_index[0]], h[edge_index[1]]], dim=1)

        if self.mp_type in {
            "updownscale",
            "base",
            "updownscale_base",
        }:
            e = self.act(self.lin_geom(e))

        # --- Message Passing block --

        if self.mp_type == "updownscale" or self.mp_type == "updownscale_base":
            h = self.act(self.lin_down(h))  # downscale node rep.
            h = self.propagate(edge_index, x=h, W=e)  # propagate
            if self.graph_norm:
                h = self.act(self.graph_norm(h))
            h = self.act(self.lin_up(h))  # upscale node rep.

        elif self.mp_type == "updown_local_env":
            h = self.act(self.lin_down(h))
            chi = self.propagate(edge_index, x=h, W=e, local_env=True)
            e = self.lin_geom(e)
            h = self.propagate(edge_index, x=h, W=e)  # propagate
            if self.graph_norm:
                h = self.act(self.graph_norm(h))
            h = torch.cat((h, chi), dim=1)
            h = self.lin_up(h)

        elif self.mp_type in {"base", "simple"}:
            h = self.propagate(edge_index, x=h, W=e)  # propagate
            if self.graph_norm:
                h = self.act(self.graph_norm(h))
            h = self.act(self.lin_h(h))

        else:
            raise ValueError("mp_type provided does not exist")

        if self.complex_mp:
            h = self.act(self.other_mlp(h))

        return h

    def message(self, x_j, W, local_env=None):
        if local_env is not None:
            return W
        else:
            return x_j * W


class OutputBlock(nn.Module):
    """Compute task-specific predictions from final atom representations."""

    def __init__(self, energy_head, hidden_channels, act, out_dim=1):
        super().__init__()
        self.energy_head = energy_head
        self.act = act

        self.lin1 = Linear(hidden_channels, hidden_channels // 2)
        self.lin2 = Linear(hidden_channels // 2, out_dim)

        if self.energy_head == "weighted-av-final-embeds":
            self.w_lin = Linear(hidden_channels, 1)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin1.weight)
        self.lin1.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.lin2.weight)
        self.lin2.bias.data.fill_(0)
        if self.energy_head == "weighted-av-final-embeds":
            nn.init.xavier_uniform_(self.w_lin.weight)
            self.w_lin.bias.data.fill_(0)

    def forward(self, h, edge_index, edge_weight, batch, alpha):
        """Forward pass of the Output block.
        Called in FAENet to make prediction from final atom representations.

        Args:
            h (tensor): atom representations. (num_atoms, hidden_channels)
            edge_index (tensor): adjacency matrix. (2, num_edges)
            edge_weight (tensor): edge weights. (num_edges, )
            batch (tensor): batch indices. (num_atoms, )
            alpha (tensor): atom attention weights for late energy head. (num_atoms, )

        Returns:
            (tensor): graph-level representation (e.g. energy prediction)
        """
        if self.energy_head == "weighted-av-final-embeds":
            alpha = self.w_lin(h)

        # MLP
        h = self.lin1(h)
        h = self.act(h)
        h = self.lin2(h)

        if self.energy_head in {
            "weighted-av-initial-embeds",
            "weighted-av-final-embeds",
        }:
            h = h * alpha

        # Global pooling
        out = scatter(h, batch, dim=0, reduce="add")

        return out


@registry.register_model("baseline")
class Baseline(BaseModel):
    r"""Non-symmetry preserving GNN model for 3D atomic systems,
    called FAENet: Frame Averaging Equivariant Network.

    Args:
        cutoff (float): Cutoff distance for interatomic interactions.
            (default: :obj:`6.0`)
        frame_averaging (str): symmetry preserving method (already) applied
            ("2D", "3D", "DA", "")
        act (str): Activation function
            (default: `swish`)
        hidden_channels (int): Hidden embedding size.
            (default: `128`)
        tag_hidden_channels (int): Hidden tag embedding size.
            (default: :obj:`32`)
        pg_hidden_channels (int): Hidden period and group embedding size.
            (default: :obj:`32`)
        phys_embeds (bool): Do we include fixed physics-aware embeddings.
            (default: :obj: `True`)
        phys_hidden_channels (int): Hidden size of learnable physics-aware embeddings.
            (default: :obj:`0`)
        num_interactions (int): The number of interaction (i.e. message passing) blocks.
            (default: :obj:`4`)
        num_gaussians (int): The number of gaussians :math:`\mu` to encode distance info.
            (default: :obj:`50`)
        num_filters (int): The size of convolutional filters.
            (default: :obj:`128`)
        second_layer_MLP (bool): Use 2-layers MLP at the end of the Embedding block.
            (default: :obj:`False`)
        skip_co (str): Add a skip connection between each interaction block and
            energy-head. (`False`, `"add"`, `"concat"`, `"concat_atom"`)
        mp_type (str): Specificies the Message Passing type of the interaction block.
            (`"base"`, `"updownscale_base"`, `"updownscale"`, `"updown_local_env"`, `"simple"`):
        graph_norm (bool): Whether to apply batch norm after every linear layer.
            (default: :obj:`True`)
        complex_mp (bool); Whether to add a second layer MLP at the end of each Interaction
            (default: :obj:`True`)
        energy_head (str): Method to compute energy prediction
            from atom representations.
            (`None`, `"weighted-av-initial-embeds"`, `"weighted-av-final-embeds"`)
        out_dim (int): size of the output tensor for graph-level predicted properties ("energy")
            Allows to predict multiple properties at the same time.
            (default: :obj:`1`)
        pred_as_dict (bool): Set to False to return a (property) prediction tensor.
            By default, predictions are returned as a dictionary with several keys (e.g. energy, forces)
            (default: :obj:`True`)
        force_decoder_type (str): Specifies the type of force decoder
            (`"simple"`, `"mlp"`, `"res"`, `"res_updown"`)
        force_decoder_model_config (dict): contains information about the
            for decoder architecture (e.g. number of layers, hidden size).
    """

    def __init__(
        self,
        num_atoms: int,  # not used
        bond_feat_dim: int,  # not used
        num_targets: int,  # not used
        otf_graph: bool,  # not used
        cutoff: float = 6.0,
        frame_averaging: str = "",
        max_neighbors: int = 40,
        act: str = "swish",
        hidden_channels: int = 128,
        tag_hidden_channels: int = 32,
        pg_hidden_channels: int = 32,
        phys_embeds: bool = True,
        phys_hidden_channels: int = 0,
        num_interactions: int = 4,
        num_gaussians: int = 50,
        num_filters: int = 128,
        second_layer_MLP: bool = True,
        skip_co: str = "concat",
        mp_type: str = "updownscale_base",
        graph_norm: bool = True,
        complex_mp: bool = False,
        energy_head: Optional[str] = None,
        out_dim: int = 1,
        pred_as_dict: bool = True,
        force_decoder_type: Optional[str] = "mlp",
        force_decoder_model_config: Optional[dict] = {"mlp": {"hidden_channels": 256}},
    ):
        super(Baseline, self).__init__()

        self.regress_forces = True
        self.otf_graph = True
        self.use_pbc = True
        self.enforce_max_neighbors_strictly = True

        self.max_neighbors = max_neighbors
        self.frame_averaging = frame_averaging
        self.act = act
        self.complex_mp = complex_mp
        self.cutoff = cutoff
        self.energy_head = energy_head
        self.force_decoder_type = force_decoder_type
        self.force_decoder_model_config = force_decoder_model_config
        self.graph_norm = graph_norm
        self.hidden_channels = hidden_channels
        self.mp_type = mp_type
        self.num_filters = num_filters
        self.num_gaussians = num_gaussians
        self.num_interactions = num_interactions
        self.pg_hidden_channels = pg_hidden_channels
        self.phys_embeds = phys_embeds
        self.phys_hidden_channels = phys_hidden_channels
        self.second_layer_MLP = second_layer_MLP
        self.skip_co = skip_co
        self.tag_hidden_channels = tag_hidden_channels
        self.pred_as_dict = pred_as_dict

        if self.mp_type == "simple":
            self.num_filters = self.hidden_channels

        self.act = (
            (getattr(nn.functional, self.act) if self.act != "swish" else swish)
            if isinstance(self.act, str)
            else self.act
        )
        assert callable(self.act), (
            "act must be a callable function or a string "
            + "describing that function in torch.nn.functional"
        )

        # Gaussian Basis
        self.distance_expansion = GaussianSmearing(0.0, self.cutoff, self.num_gaussians)

        # Embedding block
        self.embed_block = EmbeddingBlock(
            self.num_gaussians,
            self.num_filters,
            self.hidden_channels,
            self.tag_hidden_channels,
            self.pg_hidden_channels,
            self.phys_hidden_channels,
            self.phys_embeds,
            self.act,
            self.second_layer_MLP,
        )

        # Interaction block
        self.interaction_blocks = nn.ModuleList(
            [
                InteractionBlock(
                    self.hidden_channels,
                    self.num_filters,
                    self.act,
                    self.mp_type,
                    self.complex_mp,
                    self.graph_norm,
                )
                for _ in range(self.num_interactions)
            ]
        )

        # Output block
        self.output_block = OutputBlock(
            self.energy_head, self.hidden_channels, self.act, out_dim
        )

        # Energy head
        if self.energy_head == "weighted-av-initial-embeds":
            self.w_lin = Linear(self.hidden_channels, 1)

        # Force head
        self.decoder = ForceDecoder(
            self.force_decoder_type,
            self.hidden_channels,
            self.force_decoder_model_config,
            self.act,
        )

        # Skip co
        if self.skip_co == "concat":
            self.mlp_skip_co = Linear(out_dim * (self.num_interactions + 1), out_dim)
        elif self.skip_co == "concat_atom":
            self.mlp_skip_co = Linear(
                ((self.num_interactions + 1) * self.hidden_channels),
                self.hidden_channels,
            )

    def forward(self, batch, mode="train", crystal_task=True):
        """Perform a model forward pass when frame averaging is applied.

        Args:
            batch (data.Batch): batch of graphs with attributes:
                - original atom positions (`pos`)
                - batch indices (to which graph in batch each atom belongs to) (`batch`)
                - frame averaged positions, cell and rotation matrices (`fa_pos`, `fa_cell`, `fa_rot`)
            model: model instance
            mode (str, optional): model mode. Defaults to "train".
                ("train", "eval")
            crystal_task (bool, optional): Whether crystals (molecules) are considered.
                If they are, the unit cell (3x3) is affected by frame averaged and expected as attribute.
                (default: :obj:`True`)

        Returns:
            (dict): model predictions tensor for "energy" and "forces".
        """
        if isinstance(batch, list):
            batch = batch[0]
        if not hasattr(batch, "natoms"):
            batch.natoms = torch.unique(batch.batch, return_counts=True)[1]

        # Distinguish Frame Averaging prediction from traditional case.
        if self.frame_averaging and self.frame_averaging != "DA":
            original_pos = batch.pos
            if crystal_task:
                original_cell = batch.cell
            e_all, f_all, gt_all = [], [], []

            # Compute model prediction for each frame
            for i in range(len(batch.fa_pos)):
                batch.pos = batch.fa_pos[i]
                if crystal_task:
                    batch.cell = batch.fa_cell[i]
                # Forward pass
                preds = self._forward(deepcopy(batch), mode=mode)
                e_all.append(preds["energy"])
                fa_rot = None

                # Force predictions are rotated back to be equivariant
                if preds.get("forces") is not None:
                    fa_rot = torch.repeat_interleave(
                        batch.fa_rot[i], batch.natoms, dim=0
                    )
                    # Transform forces to guarantee equivariance of FA method
                    g_forces = (
                        preds["forces"]
                        .view(-1, 1, 3)
                        .bmm(fa_rot.transpose(1, 2).to(preds["forces"].device))
                        .view(-1, 3)
                    )
                    f_all.append(g_forces)

                # Energy conservation loss
                if preds.get("forces_grad_target") is not None:
                    if fa_rot is None:
                        fa_rot = torch.repeat_interleave(
                            batch.fa_rot[i], batch.natoms, dim=0
                        )
                    # Transform gradients to stay consistent with FA
                    g_grad_target = (
                        preds["forces_grad_target"]
                        .view(-1, 1, 3)
                        .bmm(
                            fa_rot.transpose(1, 2).to(
                                preds["forces_grad_target"].device
                            )
                        )
                        .view(-1, 3)
                    )
                    gt_all.append(g_grad_target)

            batch.pos = original_pos
            if crystal_task:
                batch.cell = original_cell

            # Average predictions over frames
            preds["energy"] = sum(e_all) / len(e_all)
            if len(f_all) > 0 and all(y is not None for y in f_all):
                preds["forces"] = sum(f_all) / len(f_all)
            if len(gt_all) > 0 and all(y is not None for y in gt_all):
                preds["forces_grad_target"] = sum(gt_all) / len(gt_all)

        # Traditional case (no frame averaging)
        else:
            preds = self._forward(batch, mode=mode)

        if preds["energy"].shape[-1] == 1:
            preds["energy"] = preds["energy"].view(-1)

        return preds

    def _forward(self, data, mode="train"):
        """Main Forward pass.

        Args:
            data (Data): input data object, with 3D atom positions (pos)

        Returns:
            (dict): predicted energy, forces and final atomic hidden states
        """
        # energy gradient w.r.t. positions will be computed
        if mode == "train":
            data.pos.requires_grad_(True)

        # predict energy and forces
        preds = self.energy_forward(data)
        preds["forces"] = self.forces_forward(preds)

        return preds

    def forces_forward(self, preds):
        """Predicts forces for 3D atomic systems.
        Can be utilised to predict any atom-level property.

        Args:
            preds (dict): dictionnary with final atomic representations
                (hidden_state) and predicted properties (e.g. energy)
                for each graph

        Returns:
            (dict): additional predicted properties, at an atom-level (e.g. forces)
        """
        if self.decoder:
            return self.decoder(preds["hidden_state"])

    def energy_forward(self, data):
        """Predicts any graph-level property (e.g. energy) for 3D atomic systems.

        Args:
            data (data.Batch): Batch of graphs data objects.

        Returns:
            (dict): predicted properties for each graph (key: "energy")
                and final atomic representations (key: "hidden_state")
        """
        # Pre-process data (e.g. pbc, cutoff graph, etc.)
        (
            edge_index,
            _,  # edge_dist
            rel_pos,
            _,  # cell_offsets
            _,  # cell_offset_distances
            _,  # neighbors
        ) = self.generate_graph(
            data,
            enforce_max_neighbors_strictly=self.enforce_max_neighbors_strictly,
        )
        z = data.atomic_numbers.long()
        batch = data.batch
        edge_weight = rel_pos.norm(dim=-1)

        edge_attr = self.distance_expansion(edge_weight)  # RBF of pairwise distances
        assert z.dim() == 1 and z.dtype == torch.long

        # Embedding block
        h, e = self.embed_block(
            z, rel_pos, edge_attr, data.tags if hasattr(data, "tags") else None
        )

        # Compute atom weights for late energy head
        if self.energy_head == "weighted-av-initial-embeds":
            alpha = self.w_lin(h)
        else:
            alpha = None

        # Interaction blocks
        energy_skip_co = []
        for interaction in self.interaction_blocks:
            if self.skip_co == "concat_atom":
                energy_skip_co.append(h)
            elif self.skip_co:
                energy_skip_co.append(
                    self.output_block(h, edge_index, edge_weight, batch, alpha)
                )
            h = h + interaction(h, edge_index, e)

        # Atom skip-co
        if self.skip_co == "concat_atom":
            energy_skip_co.append(h)
            h = self.act(self.mlp_skip_co(torch.cat(energy_skip_co, dim=1)))

        energy = self.output_block(h, edge_index, edge_weight, batch, alpha)

        # Skip-connection
        energy_skip_co.append(energy)
        if self.skip_co == "concat":
            energy = self.mlp_skip_co(torch.cat(energy_skip_co, dim=1))
        elif self.skip_co == "add":
            energy = sum(energy_skip_co)

        preds = {"energy": energy, "hidden_state": h}

        return preds


# @registry.register_model("baseline")
# class Baseline(BaseModel):

#     def __init__(
#         self,
#         num_atoms: int,  # not used
#         bond_feat_dim: int,  # not used
#         num_targets: int,  # not used
#         regress_forces: bool = True,
#     ) -> None:
#         super().__init__()

#         self.regress_forces = regress_forces

#         self.forces_linear = nn.Linear(3, 3)
#         self.energy_coef = nn.Parameter(torch.tensor(1.0))

#     @conditional_grad(torch.enable_grad())
#     def forward(self, data):
#         outputs = {
#             "forces": self.forces_linear(data["pos"].float()),
#             "energy": self.energy_coef * data["energy"].float(),
#         }
#         return outputs
