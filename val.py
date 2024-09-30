import argparse
import logging
import os
import random

import warnings
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torch.nn.functional as F

from fuseDPT.dpt import FuseDPT
from datasets.matterport3d import Matterport3D
from datasets.stanford2d3d import Stanford2D3D
from util.loss import SiLogLoss
from util.metric import eval_depth
from util.utils import init_log


parser = argparse.ArgumentParser(description='Depth Anything V2 for Metric Depth Estimation')

parser.add_argument('--encoder', default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])
parser.add_argument('--dataset', default='matterport3d', choices=['matterport3d', 'stanford2d3d'])
parser.add_argument('--img-size', default=280, type=int)
parser.add_argument('--min-depth', default=0.001, type=float)
parser.add_argument('--max-depth', default=8, type=float)
parser.add_argument('--pretrained-from', type=str)

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def main():
    torch.cuda.set_device(DEVICE)

    args = parser.parse_args()

    logger = init_log('global', logging.INFO)
    logger.propagate = 0

    size = (args.img_size, args.img_size * 2)
    if args.dataset == 'matterport3d':
        valset = Matterport3D('datasets/splits/M3D_v1_val.txt', 'val', size=size)
    elif args.dataset == 'stanford2d3d':
        valset = Stanford2D3D('datasets/splits/stanford2d3d_val.txt', 'val', size=size)
    else:
        raise NotImplementedError

    valloader = DataLoader(valset, batch_size=1, pin_memory=False, num_workers=12, drop_last=True)

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    model = FuseDPT(**model_configs[args.encoder]).to(DEVICE)

    state_dict = torch.load(args.pretrained_from, map_location=DEVICE)
    if args.pretrained_from:
        model.load_state_dict(state_dict['model'], strict=False)

    model.eval()

    previous_best = {'d1': 0, 'd2': 0, 'd3': 0, 'abs_rel': 100, 'sq_rel': 100, 'rmse': 100, 'rmse_log': 100, 'log10': 100, 'silog': 100}

    results = {'d1': 0.0, 'd2': 0.0, 'd3': 0.0, 'abs_rel': 0.0, 'sq_rel': 0.0, 'rmse': 0.0, 'rmse_log': 0.0, 'log10': 0.0, 'silog': 0.0}
    nsamples = 0.0

    for i, sample in enumerate(valloader):
        img, cb_img, depth, valid_mask = sample['eq_image'].to(DEVICE).float(), sample['cb_image'].to(DEVICE).float(), sample['depth'].to(DEVICE)[0], sample['valid_mask'].to(DEVICE)[0]

        with torch.no_grad():
            pred = model(img, cb_img)
            pred = F.interpolate(pred[:, None], depth.shape[-2:], mode='bilinear', align_corners=True)[0, 0]

        valid_mask = (valid_mask == 1) & (depth >= args.min_depth) & (depth <= args.max_depth)

        if valid_mask.sum() < 10:
            continue

        cur_results = eval_depth(pred[valid_mask], depth[valid_mask])

        for k in results.keys():
            results[k] += cur_results[k]
        nsamples += 1

    logger.info('==========================================================================================')
    logger.info('{:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}'.format(*tuple(results.keys())))
    logger.info('{:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}'.format(*tuple([(v / nsamples) for v in results.values()])))
    logger.info('==========================================================================================')

if __name__ == '__main__':
    main()
