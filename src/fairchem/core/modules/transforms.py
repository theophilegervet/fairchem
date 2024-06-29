from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from fairchem.core.common.utils import cg_change_mat, irreps_sum
from fairchem.core.preprocessing.transforms.frame_averaging import FrameAveraging

if TYPE_CHECKING:
    from torch_geometric.data import Data


class DataTransforms:
    def __init__(self, config) -> None:
        self.config = config

    def __call__(self, data_object):
        if not self.config:
            return data_object

        for transform_fn in self.config:
            if transform_fn == "normalizer":
                # TODO: Normalization information used in the trainers. Ignore here
                # for now.
                continue
            elif transform_fn == "FrameAveraging":
                transform = eval(transform_fn)(**self.config[transform_fn])
                data_object = transform(data_object)
            else:
                data_object = eval(transform_fn)(data_object, self.config[transform_fn])

        return data_object


def decompose_tensor(data_object, config) -> Data:
    tensor_key = config["tensor"]
    rank = config["rank"]

    if tensor_key not in data_object:
        return data_object

    if rank != 2:
        raise NotImplementedError

    tensor_decomposition = torch.einsum(
        "ab, cb->ca",
        cg_change_mat(rank),
        data_object[tensor_key].reshape(1, irreps_sum(rank)),
    )

    for decomposition_key in config["decomposition"]:
        irrep_dim = config["decomposition"][decomposition_key]["irrep_dim"]
        data_object[decomposition_key] = tensor_decomposition[
            :,
            max(0, irreps_sum(irrep_dim - 1)) : irreps_sum(irrep_dim),
        ]

    return data_object
