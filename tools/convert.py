# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Created by Tianheng Cheng(tianhengcheng@gmail.com)
# ------------------------------------------------------------------------------

import os
import pprint
import argparse

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import lib.models as models
from lib.config import config, update_config
from lib.utils import utils
from lib.datasets import get_dataset
from lib.core import function
import cv2
import os.path as osp
import pandas as pd
import ipdb


def parse_args():

    parser = argparse.ArgumentParser(description='Train Face Alignment')

    parser.add_argument('--cfg', help='experiment configuration filename',
                        required=True, type=str)
    parser.add_argument('--model-file', help='model parameters', required=True, type=str)

    args = parser.parse_args()
    update_config(config, args)
    return args


def main():

    args = parse_args()

    logger, final_output_dir, tb_log_dir = \
        utils.create_logger(config, args.cfg, 'test')

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.determinstic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    config.defrost()
    config.MODEL.INIT_WEIGHTS = False
    config.freeze()
    model = models.get_face_alignment_net(config)

    dataset_type = get_dataset(config)
    dataset = dataset_type(config, is_train=False)

    fp = open('data/free/xxtest.csv', 'w')
    keys = open('data/wflw/face_landmarks_wflw_test.csv').readline()
    fp.write(keys)
    for i in range(len(dataset)):
        # ipdb.set_trace()
        img, image_path, meta = dataset[i]
        fname = osp.join('data/free/xximages', osp.basename(image_path))
        fp.write('%s,1,128,128'%fname)
        tpts = meta['tpts'] * 4
        for j in range(tpts.shape[0]):
            cv2.circle(img, (tpts[j, 0], tpts[j, 1]), 1, (0, 0, 255))
            fp.write(',%d,%d'%(tpts[j,0], tpts[j,1]))
        fp.write('\n')
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(fname, img)
    fp.close()


if __name__ == '__main__':
    main()
