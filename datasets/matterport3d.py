import cv2
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose

from metric_depth.dataset.transform import Resize, NormalizeImage, PrepareForNet, Crop

import torch
from torch.utils import data
from torchvision import transforms

from fuseDPT.util.utils import Equirec2Cube

import random
import numpy as np

class Matterport3D(Dataset):
    def __init__(self, filelist_path, mode, size=(560, 1120), color_aug=True, LR_filp_aug=True, yaw_rot_aug=True):
        
        self.mode = mode
        self.size = size

        self.color_aug=color_aug
        self.LR_filp_aug=LR_filp_aug
        self.yaw_rot_aug=yaw_rot_aug
        
        with open(filelist_path, 'r') as f:
            self.filelist = f.read().splitlines()
        
        self.h, self.w = self.size

        self.e2c = Equirec2Cube(self.h, self.w, self.h // 2)

        #self.max_depth_meters = 10.0

        if self.color_aug:
            try:
                self.brightness = (0.8, 1.2)
                self.contrast = (0.8, 1.2)
                self.saturation = (0.8, 1.2)
                self.hue = (-0.1, 0.1)
                self.color_aug= transforms.ColorJitter(self.brightness, self.contrast, self.saturation, self.hue)
            except TypeError:
                self.brightness = 0.2
                self.contrast = 0.2
                self.saturation = 0.2
                self.hue = 0.1
                self.color_aug = transforms.ColorJitter(self.brightness, self.contrast, self.saturation, self.hue)
        
        self.transform = Compose([
            Resize(
                width=self.w,
                height=self.h,
                resize_target=True if mode == 'train' else False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])

        self.cb_transform = Compose([
            Resize(
                width=self.w // 2,
                height=self.h // 2,
                resize_target=True if mode == 'train' else False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            PrepareForNet(),
        ])

        
    def __getitem__(self, item):
        img_path = self.filelist[item].split(' ')[0]
        depth_path = self.filelist[item].split(' ')[1]
        
        eq_image = cv2.imread(img_path)
        eq_image = cv2.cvtColor(eq_image, cv2.COLOR_BGR2RGB) / 255.0

        #print('After loading max', eq_image.max())
        #print('After loading min', eq_image.min())
        
        depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
        depth = cv2.resize(depth, dsize=(self.w, self.h), interpolation=cv2.INTER_NEAREST)
        #depth = depth.astype(np.float32) / 100.0
        depth[depth > 8] = 8 + 1e-6

        
        #depth_fd = h5py.File(depth_path, "r")
        #distance_meters = np.array(depth_fd['dataset'])
        #depth = hypersim_distance_to_depth(distance_meters)
        
        #depth = np.expand_dims(sample['depth'], 2)
        #print(depth.shape)

        #if self.mode == 'train' and self.yaw_rot_aug:
            # random yaw rotation
        #    roll_idx = random.randint(0, self.w)
        #    eq_image = np.roll(eq_image, roll_idx, 1)
        #    depth = np.roll(depth, roll_idx, 1)

        #print(depth.dims())

        if self.mode == 'train' and self.LR_filp_aug and random.random() > 0.5:
            eq_image = cv2.flip(eq_image, 1)
            depth = cv2.flip(depth, 1)

        if self.mode == 'train' and self.color_aug and random.random() > 0.5:
            aug_eq_image = np.asarray(self.color_aug(transforms.ToPILImage()(eq_image)))
        else:
            aug_eq_image = eq_image.copy()

        sample = self.transform({'image': aug_eq_image, 'depth': depth})
        sample['aug_image'] = self.transform({'image': aug_eq_image})['image']

        #print('After norm max', sample['image'].max())
        #print('After norm min', sample['image'].min())

        eq_image = sample['image']
        aug_eq_image = sample['aug_image']
        depth = sample['depth']

        cb_image = self.e2c.run(eq_image.transpose(1,2,0))
        aug_cb_image = self.e2c.run(sample['image'].transpose(1,2,0))

        cb_image = self.cb_transform({'image': cb_image})['image']
        aug_cb_image = self.cb_transform({'image': aug_cb_image})['image']

        #eq_image = torch.from_numpy(eq_image)
        #cb_image = torch.from_numpy(cb_image)
        #DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        #eq_image = eq_image.to(DEVICE)
        #cb_image = cb_image.to(DEVICE)

        #print(eq_image.shape)
        #print(cb_image.shape)
        

        sample['eq_image'] = torch.from_numpy(eq_image)
        sample['cb_image'] = torch.from_numpy(cb_image)
        sample['aug_eq_image'] = torch.from_numpy(aug_eq_image)
        sample['aug_cb_image'] = torch.from_numpy(aug_cb_image)

        sample['depth'] = torch.from_numpy(depth)
        
        sample['valid_mask'] = (torch.isnan(sample['depth']) == 0) 
        #sample['depth'][sample['valid_mask'] == 0] = 0.0001
        
        sample['image_path'] = self.filelist[item].split(' ')[0]
        
        return sample

    def __len__(self):
        return len(self.filelist)