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


def main_wflw():
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
    dataset = dataset_type(config, is_train=True)

    fp = open('data/wflw/face_landmarks70_wflw_train.csv', 'w')
    for i in range(len(dataset)):
        img, image_path, meta = dataset[i]
        fold, name = image_path.split('/')[-2], image_path.split('/')[-1]
        folder = osp.join('data/wflw/xximages', fold)
        if not osp.exists(folder):
            os.makedirs(folder)
        fname = osp.join(folder, name)
        scale = meta['scale']
        center = meta['center']
        tpts = meta['tpts']

        selpts = []
        for j in range(0,33,2):
            selpts.append(tpts[j])
        # eyebow
        selpts.append(tpts[33])
        selpts.append((tpts[34] + tpts[41]) / 2)
        selpts.append((tpts[35] + tpts[40]) / 2)
        selpts.append((tpts[36] + tpts[39]) / 2)
        selpts.append((tpts[37] + tpts[38]) / 2)
        selpts.append((tpts[42] + tpts[50]) / 2)
        selpts.append((tpts[43] + tpts[49]) / 2)
        selpts.append((tpts[44] + tpts[48]) / 2)
        selpts.append((tpts[45] + tpts[47]) / 2)
        selpts.append(tpts[46])
        # nose
        for j in range(51,60):
            selpts.append(tpts[j])
        # eye
        selpts.append(tpts[60])
        selpts.append((tpts[61] + tpts[62]) / 2)
        selpts.append(tpts[63])
        selpts.append(tpts[64])
        selpts.append(tpts[65])
        selpts.append((tpts[66] + tpts[67]) / 2)
        selpts.append(tpts[68])
        selpts.append(tpts[69])
        selpts.append((tpts[70] + tpts[71]) / 2)
        selpts.append(tpts[72])
        selpts.append((tpts[73] + tpts[74]) / 2)
        selpts.append(tpts[75])
        for j in range(76,98):
            selpts.append(tpts[j])

        fp.write('%s,%.2f,%.1f,%.1f' % (osp.join(fold, name), scale, center[0], center[1]))
        for spt in selpts:
            cv2.circle(img, (spt[0], spt[1]), 1, (0, 0, 255))
            fp.write(',%f,%f' % (spt[0], spt[1]))
        fp.write('\n')
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(fname, img)
    fp.close()


def main_cofw():
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

    ipdb.set_trace()
    dataset_type = get_dataset(config)
    dataset = dataset_type(config, is_train=True)

    fp = open('data/cofw/test.csv', 'w')
    for i in range(len(dataset)):
        # ipdb.set_trace()
        img, image_path, meta = dataset[i]
        fname = osp.join('data/cofw/test', osp.basename(image_path))
        fp.write('%s,1,128,128'%fname)
        tpts = meta['tpts']
        for j in range(tpts.shape[0]):
            fp.write(',%d,%d'%(tpts[j,0], tpts[j,1]))
        fp.write('\n')
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(fname, img)
    fp.close()


def main_300w():
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

    fp = open('data/300w/face_landmarks70_300w_test.csv', 'w')
    for i in range(len(dataset)):
        # ipdb.set_trace()
        img, fname, meta = dataset[i]
        filename = osp.join('data/300w/xximages', fname)
        if not osp.exists(osp.dirname(filename)):
            os.makedirs(osp.dirname(filename))
        scale = meta['scale']
        center = meta['center']
        tpts = meta['tpts']

        selpts = []
        for j in range(0,68):
            selpts.append(tpts[j])
        selpts.append(tpts[36:42].mean(0))
        selpts.append(tpts[42:48].mean(0))

        fp.write('%s,%.2f,%.1f,%.1f' % (fname, scale, center[0], center[1]))
        for spt in selpts:
            img = cv2.circle(img, (spt[0], spt[1]), 1+center[0]//400, (255, 0, 0))
            fp.write(',%f,%f' % (spt[0], spt[1]))
        fp.write('\n')
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(filename, img)
    fp.close()


def main_wflwe70():
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
    dataset = dataset_type(config, is_train=True)

    for i in range(len(dataset)):
        # ipdb.set_trace()
        img, fname, meta = dataset[i]
        filename = osp.join('data/wflwe70/xximages', fname)
        if not osp.exists(osp.dirname(filename)):
            os.makedirs(osp.dirname(filename))
        scale = meta['scale']
        center = meta['center']
        tpts = meta['tpts']

        for spt in tpts:
            img = cv2.circle(img, (4*spt[0], 4*spt[1]), 1+center[0]//400, (255, 0, 0))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(filename, img)


if __name__ == '__main__':
    main_wflwe70()
