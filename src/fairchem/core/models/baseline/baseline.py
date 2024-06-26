import torch
import torch.nn as nn

from fairchem.core.common.registry import registry
from fairchem.core.common.utils import conditional_grad
from fairchem.core.models.base import BaseModel


@registry.register_model("baseline")
class Baseline(BaseModel):

    def __init__(
        self,
        num_atoms: int,  # not used
        bond_feat_dim: int,  # not used
        num_targets: int,  # not used
        regress_forces: bool = True,
    ) -> None:
        super().__init__()

        self.regress_forces = regress_forces

        self.forces_linear = nn.Linear(3, 3)
        self.energy_coef = nn.Parameter(torch.tensor(1.0))

    @conditional_grad(torch.enable_grad())
    def forward(self, data):
        outputs = {
            "forces": self.forces_linear(data["pos"].float()),
            "energy": self.energy_coef * data["energy"].float(),
        }
        return outputs
