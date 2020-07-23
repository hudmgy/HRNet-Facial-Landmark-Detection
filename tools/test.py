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
import ipdb


def parse_args():

    parser = argparse.ArgumentParser(description='Train Face Alignment')

    parser.add_argument('--cfg', help='experiment configuration filename',
                        required=True, type=str)
    parser.add_argument('--model-file', help='model parameters', required=True, type=str)

    parser.add_argument('--onnx-export', type=str, default='',
                    help="convert model to onnx")

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
    if args.onnx_export:
        torch_out = torch.onnx._export(model, torch.rand(1, 3, config.IMAGE_SIZE), 
                osp.join(final_output_dir, args.onnx_export), export_params=True)
        return

    gpus = list(config.GPUS)
    if gpus[0] > -1:
        model = nn.DataParallel(model, device_ids=gpus).cuda()

    # load model
    if gpus[0] > -1:
        state_dict = torch.load(args.model_file)
    else:
        state_dict = torch.load(args.model_file, map_location='cpu')
    if 'state_dict' in state_dict.keys():
        state_dict = state_dict['state_dict']
        model.load_state_dict(state_dict)
    else:
        if gpus[0] > -1:
            model.module.load_state_dict(state_dict)
        else:
            model.load_state_dict(state_dict)

    dataset_type = get_dataset(config)
    dataset = dataset_type(config, is_train=False)

    test_loader = DataLoader(
        dataset=dataset,
        batch_size=config.TEST.BATCH_SIZE_PER_GPU*len(gpus),
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=config.PIN_MEMORY
    )

    ipdb.set_trace()
    nme, predictions = function.inference(config, test_loader, model)

    torch.save(predictions, os.path.join(final_output_dir, 'predictions.pth'))


if __name__ == '__main__':
    main()
