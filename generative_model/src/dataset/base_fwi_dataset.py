# Last modified: 2024-04-30
#
# Copyright 2023 Bingxin Ke, ETH Zurich. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# --------------------------------------------------------------------------
# If you find this code useful, we kindly ask you to cite our paper in your work.
# Please find bibtex at: https://github.com/prs-eth/Marigold#-citation
# If you use or adapt this code, please attribute to https://github.com/prs-eth/marigold.
# More information about the method can be found at https://marigoldmonodepth.github.io
# --------------------------------------------------------------------------

import os
from enum import Enum

import numpy as np
import torch
from skimage import transform
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode, Resize


class DatasetMode(Enum):
    EVAL = "evaluate"
    TRAIN = "train"


class EpsFileNameMode(Enum):
    """Prediction file naming modes"""

    id = 1  # id.png
    bscan_id = 2  # rgb_id.png or Bscan_id.npy
    i_d_rgb = 3  # i_d_1_rgb.png
    rgb_i_d = 4


class BaseFWIDataset(Dataset):
    def __init__(
        self,
        mode: DatasetMode,
        filename_ls_path: str,
        dataset_dir: str,
        disp_name: str,
        min_eps: float,
        max_eps: float,
        resize_to_hw=None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.mode = mode
        # dataset info
        self.filename_ls_path = filename_ls_path
        self.dataset_dir = dataset_dir
        assert os.path.exists(
            self.dataset_dir
        ), f"Dataset does not exist at: {self.dataset_dir}"
        self.disp_name = disp_name
        self.min_eps = min_eps
        self.max_eps = max_eps

        # training arguments
        self.resize_to_hw = resize_to_hw

        # Load filenames
        with open(self.filename_ls_path, "r") as f:
            self.filenames = [
                s.split() for s in f.readlines()
            ]  # [['rgb.png', 'depth.tif'], [], ...]
        
        src = np.array([[0, 0], [230, 51], [230, 230], [0, 230]])
        dst = np.array([[0, 0], [230, 0], [230, 230], [0, 230]])
        self.tform = transform.ProjectiveTransform()
        self.tform.estimate(dst, src)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        rasters, other = self._get_data_item(index)
        if DatasetMode.TRAIN == self.mode:
            rasters = self._training_preprocess(rasters)
        # merge
        outputs = rasters
        outputs.update(other)
        return outputs

    def _get_data_item(self, index):
        bscan_rel_path, eps_rel_path = self._get_data_path(index=index)

        rasters = {}

        # RGB data
        rasters.update(self._load_bscan_data(bscan_rel_path=bscan_rel_path))

        # Depth data
        eps_data = self._load_eps_data(eps_rel_path=eps_rel_path)
        rasters.update(eps_data)

        other = {"index": index, "bscan_relative_path": bscan_rel_path}

        return rasters, other

    def _load_bscan_data(self, bscan_rel_path):
        # Read Bscan data
        bscan = self._read_npy(bscan_rel_path)
        # TODO: Warp?
        bscan = transform.warp(bscan, self.tform, output_shape=(230, 230))
        b_min, b_max = bscan.min(), bscan.max()
        bscan_norm = (bscan - b_min) / (b_max - b_min) * 2 - 1 # [-1, 1]

        outputs = {
            # "bscan_raw": torch.from_numpy(bscan).float().unsqueeze(0),
            "bscan_norm": torch.from_numpy(bscan_norm).float().unsqueeze(0),
        }
        return outputs

    def _load_eps_data(self, eps_rel_path):
        # Read eps data
        outputs = {}
        eps_raw = self._read_npy(eps_rel_path)
        eps_norm = (eps_raw - 1.0) / 9.0 * 2 - 1  # [-1, 1]
        eps_norm = torch.from_numpy(eps_norm).float().unsqueeze(0)  # [1, H, W]
        outputs["eps_norm"] = eps_norm.clone()

        return outputs

    def _get_data_path(self, index):
        filename_line = self.filenames[index]

        # Get data path
        bscan_rel_path = filename_line[0]
        eps_rel_path = filename_line[1]
        return bscan_rel_path, eps_rel_path

    def _read_npy(self, img_rel_path) -> np.ndarray:
        npy_to_read = os.path.join(self.dataset_dir, img_rel_path)
        image = np.load(npy_to_read, mmap_mode='r').astype(np.float32)  # [H, W]
        return image

    def _training_preprocess(self, rasters):
        # Resize
        if self.resize_to_hw is not None:
            resize_transform = Resize(
                size=self.resize_to_hw, interpolation=InterpolationMode.NEAREST_EXACT
            )
            rasters = {k: resize_transform(v) for k, v in rasters.items()}
        return rasters


def get_pred_name(bscan_basename, name_mode, suffix=".npy"):
    assert EpsFileNameMode.bscan_id == name_mode
    pred_basename = "pred_" + bscan_basename.split("_")[1]
    pred_basename = os.path.splitext(pred_basename)[0] + suffix

    return pred_basename
