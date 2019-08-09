import os
import sys
import os.path as osp
import torch
import numpy as np
import pprint
import pandas as pd
import cv2
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import ipdb


def draw_kpts(src_dir, target_dir, landmarks_gt, predictions):
    for idx in range(len(landmarks_gt)):
        image_path = osp.join(src_dir,  landmarks_gt.iloc[idx, 0])
        scale = landmarks_gt.iloc[idx, 1]
        center_w = landmarks_gt.iloc[idx, 2]
        center_h = landmarks_gt.iloc[idx, 3]
        center = torch.Tensor([center_w, center_h])
        pts = landmarks_gt.iloc[idx, 4:].values
        pts = pts.astype('float').reshape(-1, 2)

        img = cv2.imread(image_path)
        preds = predictions[idx]
        for i in range(preds.shape[0]):
            cv2.circle(img, (preds[i,0], preds[i,1]), 1, (0,0,255))

        target_file = osp.join(target_dir, osp.basename(image_path))
        if not osp.exists(osp.dirname(target_file)):
            os.makedirs(osp.dirname(target_file))
        cv2.imwrite(target_file, img)


if __name__=='__main__':
    csv_file = 'data/wflw/face_landmarks_wflw_test.csv'
    src_dir = 'data/wflw/images'
    final_output_dir = 'output/WFLW/face_alignment_wflw_hrnet_w18'
    target_dir = osp.join(final_output_dir, 'draw')

    landmarks_gt = pd.read_csv(csv_file)
    predictions = torch.load(osp.join(final_output_dir, 'predictions.pth'))

    draw_kpts(src_dir, target_dir, landmarks_gt, predictions)