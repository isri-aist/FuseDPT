import argparse
import logging
import os
import pprint
import random

import warnings
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from fuseDPT.dpt import FuseDPT
from datasets.matterport3d import Matterport3D
from datasets.stanford2d3d import Stanford2D3D
from util.dist_helper import setup_distributed
from util.loss import SiLogLoss
from util.metric import eval_depth
from util.utils import init_log

import multiprocessing as mp


parser = argparse.ArgumentParser(description='Depth Anything V2 for Metric Depth Estimation')

parser.add_argument('--encoder', default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])
parser.add_argument('--dataset', default='matterport3d', choices=['matterport3d', 'stanford2d3d'])
parser.add_argument('--img-size', default=560, type=int)
parser.add_argument('--min-depth', default=0.001, type=float)
parser.add_argument('--max-depth', default=8, type=float)
parser.add_argument('--epochs', default=150, type=int)
parser.add_argument('--bs', default=2, type=int)
parser.add_argument('--lr', default=0.000005, type=float)
parser.add_argument('--pretrained-from', type=str)
parser.add_argument('--save-path', type=str, required=True)
parser.add_argument('--local-rank', default=0, type=int)
parser.add_argument('--port', default=None, type=int)

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'



def main():
    torch.cuda.set_device(DEVICE)

    torch.autograd.set_detect_anomaly(True)
    
    args = parser.parse_args()

    logger = init_log('global', logging.INFO)
    logger.propagate = 0

    size = (args.img_size, args.img_size*2)
    if args.dataset == 'matterport3d':
        trainset = Matterport3D('datasets/splits/M3D_v1_train.txt', 'train', size=size)
        valset = Matterport3D('datasets/splits/M3D_v1_val.txt', 'val', size=size)
    elif args.dataset == 'stanford2d3d':
        trainset = Stanford2D3D('datasets/splits/stanford2d3d_train.txt', 'train', size=size)
        valset = Stanford2D3D('datasets/splits/stanford2d3d_val.txt', 'val', size=size)
    else:
        raise NotImplementedError
    
    #np.random.seed(1000)
    #random.seed(1000)
    #torch.manual_seed(1000)
    #torch.cuda.manual_seed(1000)
    #torch.cuda.manual_seed_all(1000)
    #torch.backends.cudnn.enabled = False
    #torch.backends.cudnn.benchmark = False
    #torch.backends.cudnn.deterministic = True

    trainloader = DataLoader(trainset, batch_size=args.bs, pin_memory=False, num_workers=12, drop_last=True, shuffle=False)
    valloader = DataLoader(valset, batch_size=1, pin_memory=False, num_workers=12, drop_last=True)

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    model = FuseDPT(**model_configs[args.encoder]).to(DEVICE)

    # Fix initialization for the output layer so it's closer to the true depth values

    torch.nn.init.uniform_(model.depth_head.eq_scratch.output_conv2[0].weight, 0.01, 0.1)
    torch.nn.init.uniform_(model.depth_head.eq_scratch.output_conv2[0].bias, 0.01, 0.1)
    torch.nn.init.uniform_(model.depth_head.eq_scratch.output_conv2[2].weight, 0.01, 0.1)
    torch.nn.init.uniform_(model.depth_head.eq_scratch.output_conv2[2].bias, 0.01, 0.1)

    state_dict = torch.load(args.pretrained_from, map_location='cuda')

    #print(state_dict.keys())

    if args.pretrained_from:
        model.load_state_dict(state_dict, strict=False)

    criterion = SiLogLoss().to(DEVICE)

    model.erp_pretrained.requires_grad_(False)
    model.cmp_pretrained.requires_grad_(False)

    #torch.nn.init.kaiming_uniform_(model.depth_head.eq_scratch.output_conv2[3].weight)
    #model.depth_head.eq_scratch.output_conv2[2].reset_parameters()

    optimizer = AdamW([{'params': [param for name, param in model.named_parameters() if 'pretrained' in name], 'lr': args.lr},
                       {'params': [param for name, param in model.named_parameters() if 'pretrained' not in name], 'lr': args.lr * 10}],
                      lr=args.lr, betas=(0.9, 0.999), weight_decay=0.01)
    
    #optimizer.load_state_dict(state_dict['optimizer'])
    #optimizer.param_groups[0]["lr"] = args.lr
    #optimizer.param_groups[1]["lr"] = args.lr * 10

    total_iters = args.epochs * len(trainloader)

    previous_best = {'d1': 0, 'd2': 0, 'd3': 0, 'abs_rel': 100, 'sq_rel': 100, 'rmse': 100, 'rmse_log': 100, 'log10': 100, 'silog': 100}

    for epoch in range(args.epochs):
        logger.info('===========> Epoch: {:}/{:}, d1: {:.3f}, d2: {:.3f}, d3: {:.3f}'.format(epoch, args.epochs, previous_best['d1'], previous_best['d2'], previous_best['d3']))
        
        if epoch == 50:
            model.erp_pretrained.requires_grad_(True)
            model.cmp_pretrained.requires_grad_(True)

        model.train()
        total_loss = 0
        
        for i, sample in enumerate(trainloader):
            optimizer.zero_grad()
            
            img, cb_img, depth, valid_mask = sample['aug_eq_image'].to(DEVICE), sample['aug_cb_image'].to(DEVICE), sample['depth'].to(DEVICE), sample['valid_mask'].to(DEVICE)

            if valid_mask.sum() < 10:
                continue
            
            #print(img.max())
            #print(img.min())
            #print(cb_img.max())
            #print(cb_img.min())

            #print(sample['image_path'])

            #logger.info(sample['depth'])

            pred = model(img, cb_img)
            
            #print(pred)

            # Print output conv layers if the vanishing gradient problem happens again
            #if not torch.any(pred!=0):
            #    print(model.depth_head.eq_scratch.output_conv2[0].weight)
            #    print(model.depth_head.eq_scratch.output_conv2[2].weight)
            #print(depth)
            #print(pred)
            #print(img)
            #print(cb_img)
            #print(model.depth_head.state_dict())
            #pred[pred==0] = args.min_depth

            # Remove NaN values from prediction
            
            loss = criterion(depth, pred, (valid_mask == 1) & (depth >= args.min_depth) & (depth <= args.max_depth))
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            iters = epoch * len(trainloader) + i
            
            lr = args.lr * (1 - iters / total_iters) ** 0.9
            
            optimizer.param_groups[0]["lr"] = lr
            optimizer.param_groups[1]["lr"] = lr * 10
            if i % 100 == 0:
                logger.info('Iter: {}/{}, LR: {:.7f}, Loss: {:.3f}'.format(i, len(trainloader), optimizer.param_groups[0]['lr'], loss.item()))
        
        model.eval()
        
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

        for k in results.keys():
            if k in ['d1', 'd2', 'd3']:
                previous_best[k] = max(previous_best[k], (results[k] / nsamples))
            else:
                previous_best[k] = min(previous_best[k], (results[k] / nsamples))

        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'previous_best': previous_best,
            'val_results':results
        }

        torch.save(checkpoint, os.path.join(args.save_path, f'm3d_560_{epoch}.pth'))

if __name__ == '__main__':
    main()