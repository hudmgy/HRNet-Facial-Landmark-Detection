# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Created by Tianheng Cheng(tianhengcheng@gmail.com), Yang Zhao
# ------------------------------------------------------------------------------

import os
import random

import torch
import torch.utils.data as data
import pandas as pd
from PIL import Image
import numpy as np

from ..utils.transforms import fliplr_joints, crop, generate_target, transform_pixel


class Face300WSD(data.Dataset):

    def __init__(self, cfg, is_train=True, transform=None):
        # specify annotation file for dataset
        if is_train:
            self.csv_file = cfg.DATASET.TRAINSET
        else:
            self.csv_file = cfg.DATASET.TESTSET

        self.is_train = is_train
        self.transform = transform
        self.data_root = cfg.DATASET.ROOT
        self.input_size = cfg.MODEL.IMAGE_SIZE
        self.output_size = cfg.MODEL.HEATMAP_SIZE
        self.sigma = cfg.MODEL.SIGMA
        self.scale_factor = cfg.DATASET.SCALE_FACTOR
        self.rot_factor = cfg.DATASET.ROT_FACTOR
        self.label_type = cfg.MODEL.TARGET_TYPE
        self.flip = cfg.DATASET.FLIP

        # load annotations
        self.landmarks_frame = pd.read_csv(self.csv_file)

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):

        fname = self.landmarks_frame.iloc[idx, 0]
        image_path = os.path.join(self.data_root, fname)
        scale = self.landmarks_frame.iloc[idx, 1]

        center_w = self.landmarks_frame.iloc[idx, 2]
        center_h = self.landmarks_frame.iloc[idx, 3]
        center = torch.Tensor([center_w, center_h])

        pts = self.landmarks_frame.iloc[idx, 4:].values
        pts = pts.astype('float').reshape(-1, 2)
        img = np.array(Image.open(image_path).convert('RGB'), dtype=np.float32)
        tpts = torch.Tensor(pts)
        center = torch.Tensor(center)

        meta = {'index': idx, 'center': center, 'scale': scale,
                'pts': torch.Tensor(pts), 'tpts': tpts}

        return img, fname, meta

if __name__ == '__main__':
    pass
