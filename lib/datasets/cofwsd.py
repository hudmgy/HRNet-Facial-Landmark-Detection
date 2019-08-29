# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Created by Tianheng Cheng(tianhengcheng@gmail.com), Yang Zhao
# ------------------------------------------------------------------------------

import math
import random

import torch
import torch.utils.data as data
import numpy as np

from hdf5storage import loadmat
from ..utils.transforms import fliplr_joints, crop, generate_target, transform_pixel
import os.path as osp
import cv2

class COFWSD(data.Dataset):

    def __init__(self, cfg, is_train=True, transform=None):
        # specify annotation file for dataset
        if is_train:
            self.mat_file = cfg.DATASET.TRAINSET
            self.img_path = osp.join(osp.dirname(self.mat_file), 'train')
        else:
            self.mat_file = cfg.DATASET.TESTSET
            self.img_path = osp.join(osp.dirname(self.mat_file), 'test')

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
        self.mat = loadmat(self.mat_file)
        if is_train:
            self.images = self.mat['IsTr']
            self.pts = self.mat['phisTr']
        else:
            self.images = self.mat['IsT']
            self.pts = self.mat['phisT']

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        img = self.images[idx][0]
        image_path = osp.join(self.img_path, '%04d.jpg'%idx)

        if len(img.shape) == 2:
            img = img.reshape(img.shape[0], img.shape[1], 1)
            img = np.repeat(img, 3, axis=2)

        pts = self.pts[idx][0:58].reshape(2, -1).transpose()

        xmin = np.min(pts[:, 0])
        xmax = np.max(pts[:, 0])
        ymin = np.min(pts[:, 1])
        ymax = np.max(pts[:, 1])

        center_w = (math.floor(xmin) + math.ceil(xmax)) / 2.0
        center_h = (math.floor(ymin) + math.ceil(ymax)) / 2.0

        scale = max(math.ceil(xmax) - math.floor(xmin), math.ceil(ymax) - math.floor(ymin)) / 200.0
        center = torch.Tensor([center_w, center_h])

        tpts = torch.Tensor(pts)
        center = torch.Tensor(center)

        meta = {'index': idx, 'center': center, 'scale': scale,
                'pts': torch.Tensor(pts), 'tpts': tpts}

        h = 100 * scale
        img = cv2.rectangle(img, (int(center_w-h),int(center_h-h)), (int(center_w+h), int(center_h+h)), (0,255,255))
        for i in range(tpts.shape[0]):
            img = cv2.circle(img, (int(tpts[i,0]), int(tpts[i,1])), 2, (0,0,255))

        return img, image_path, meta


if __name__ == '__main__':

    pass
