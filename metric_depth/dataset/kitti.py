import cv2
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose

from dataset.transform import Resize, NormalizeImage, PrepareForNet



from __future__ import print_function
import os
import cv2
import numpy as np
import random

import torch
from torch.utils import data
from torchvision import transforms

from fuseDPT.util.utils import Equirec2Cube


def read_list(list_file):
    rgb_depth_list = []
    with open(list_file) as f:
        lines = f.readlines()
        for line in lines:
            rgb_depth_list.append(line.strip().split(" "))
    return rgb_depth_list


class Matterport3D_old(data.Dataset):
    """The Matterport3D Dataset"""

    def __init__(self, root_dir, list_file, height=512, width=1024, disable_color_augmentation=False,
                 disable_LR_filp_augmentation=False, disable_yaw_rotation_augmentation=False, is_training=False):
        """
        Args:
            root_dir (string): Directory of the Stanford2D3D Dataset.
             (string): Path to the txt file contain the list of image and depth files.
            height, width: input size.
            disable_color_augmentation, disable_LR_filp_augmentation,
            disable_yaw_rotation_augmentation: augmentation options.
            is_training (bool): True if the dataset is the training set.
        """
        self.root_dir = root_dir
        self.rgb_depth_list = read_list(list_file)

        self.w = width
        self.h = height

        self.max_depth_meters = 10.0

        self.color_augmentation = not disable_color_augmentation
        self.LR_filp_augmentation = not disable_LR_filp_augmentation
        self.yaw_rotation_augmentation = not disable_yaw_rotation_augmentation

        self.is_training = is_training

        self.e2c = Equirec2Cube(self.h, self.w, self.h // 2)

        if self.color_augmentation:
            try:
                self.brightness = (0.8, 1.2)
                self.contrast = (0.8, 1.2)
                self.saturation = (0.8, 1.2)
                self.hue = (-0.1, 0.1)
                self.color_aug= transforms.ColorJitter.get_params(
                    self.brightness, self.contrast, self.saturation, self.hue)
            except TypeError:
                self.brightness = 0.2
                self.contrast = 0.2
                self.saturation = 0.2
                self.hue = 0.1
                self.color_aug = transforms.ColorJitter.get_params(
                    self.brightness, self.contrast, self.saturation, self.hue)

        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.rgb_depth_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        inputs = {}

        rgb_name = os.path.join(self.root_dir, self.rgb_depth_list[idx][0])
        rgb = cv2.imread(rgb_name)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, dsize=(self.w, self.h), interpolation=cv2.INTER_CUBIC)

        depth_name = os.path.join(self.root_dir, self.rgb_depth_list[idx][1])
        gt_depth = cv2.imread(depth_name, -1)
        gt_depth = cv2.resize(gt_depth, dsize=(self.w, self.h), interpolation=cv2.INTER_NEAREST)
        gt_depth = gt_depth.astype(np.float)/4000
        gt_depth[gt_depth > self.max_depth_meters+1] = self.max_depth_meters + 1


        if self.is_training and self.yaw_rotation_augmentation:
            # random yaw rotation
            roll_idx = random.randint(0, self.w)
            rgb = np.roll(rgb, roll_idx, 1)
            gt_depth = np.roll(gt_depth, roll_idx, 1)

        if self.is_training and self.LR_filp_augmentation and random.random() > 0.5:
            rgb = cv2.flip(rgb, 1)
            gt_depth = cv2.flip(gt_depth, 1)

        if self.is_training and self.color_augmentation and random.random() > 0.5:
            aug_rgb = np.asarray(self.color_aug(transforms.ToPILImage()(rgb)))
        else:
            aug_rgb = rgb

        #cube_rgb, cube_gt_depth = self.e2c.run(rgb, gt_depth[..., np.newaxis])
        cube_rgb = self.e2c.run(rgb)
        cube_aug_rgb = self.e2c.run(aug_rgb)

        rgb = self.to_tensor(rgb.copy())
        cube_rgb = self.to_tensor(cube_rgb.copy())
        aug_rgb = self.to_tensor(aug_rgb.copy())
        cube_aug_rgb = self.to_tensor(cube_aug_rgb.copy())

        inputs["rgb"] = rgb
        inputs["normalized_rgb"] = self.normalize(aug_rgb)

        inputs["cube_rgb"] = cube_rgb
        inputs["normalized_cube_rgb"] = self.normalize(cube_aug_rgb)


        inputs["gt_depth"] = torch.from_numpy(np.expand_dims(gt_depth, axis=0))
        inputs["val_mask"] = ((inputs["gt_depth"] > 0) & (inputs["gt_depth"] <= self.max_depth_meters)
                                & ~torch.isnan(inputs["gt_depth"]))

        """
        cube_gt_depth = torch.from_numpy(np.expand_dims(cube_gt_depth[..., 0], axis=0))
        inputs["cube_gt_depth"] = cube_gt_depth
        inputs["cube_val_mask"] = ((cube_gt_depth > 0) & (cube_gt_depth <= self.max_depth_meters)
                                   & ~torch.isnan(cube_gt_depth))
        """

        return inputs


class Matterport3D(Dataset):
    def __init__(self, filelist_path, mode, size=(518, 518), color_aug=False, LR_filp_aug=False, yaw_rot_aug=False):
        
        self.mode = mode
        self.size = size

        self.color_aug=color_aug
        self.LR_filp_aug=LR_filp_aug
        self.yaw_rot_aug=yaw_rot_aug
        
        with open(filelist_path, 'r') as f:
            self.filelist = f.read().splitlines()
        
        self.w, self.h = size

        self.e2c = Equirec2Cube(self.h, self.w, self.h // 2)

        self.max_depth_meters = 10.0

        if self.color_augmentation:
            try:
                self.brightness = (0.8, 1.2)
                self.contrast = (0.8, 1.2)
                self.saturation = (0.8, 1.2)
                self.hue = (-0.1, 0.1)
                self.color_aug= transforms.ColorJitter.get_params(
                    self.brightness, self.contrast, self.saturation, self.hue)
            except TypeError:
                self.brightness = 0.2
                self.contrast = 0.2
                self.saturation = 0.2
                self.hue = 0.1
                self.color_aug = transforms.ColorJitter.get_params(
                    self.brightness, self.contrast, self.saturation, self.hue)
        
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
        ] + ([Crop(size[0])] if self.mode == 'train' else []))
        
    def __getitem__(self, item):
        img_path = self.filelist[item].split(' ')[0]
        depth_path = self.filelist[item].split(' ')[1]
        
        eq_image = cv2.imread(img_path)
        eq_image = cv2.cvtColor(eq_image, cv2.COLOR_BGR2RGB) / 255.0
        
        depth = cv2.imread(depth_path, -1)
        depth = cv2.resize(depth, dsize=(self.w, self.h), interpolation=cv2.INTER_NEAREST)
        depth = depth.astype(np.float) / 4000
        depth[depth > self.max_depth_meters+1] = self.max_depth_meters + 1

        
        #depth_fd = h5py.File(depth_path, "r")
        #distance_meters = np.array(depth_fd['dataset'])
        #depth = hypersim_distance_to_depth(distance_meters)
        
        sample = self.transform({'eq_image': eq_image, 'depth':depth})

        eq_image = sample['eq_image']
        depth = sample['depth']

        if self.mode == 'train' and self.yaw_rotation_augmentation:
            # random yaw rotation
            roll_idx = random.randint(0, self.w)
            eq_image = np.roll(eq_image, roll_idx, 1)
            depth = np.roll(depth, roll_idx, 1)

        if self.mode == 'train' and self.LR_filp_augmentation and random.random() > 0.5:
            eq_image = cv2.flip(eq_image, 1)
            depth = cv2.flip(depth, 1)

        if self.mode == 'train' and self.color_augmentation and random.random() > 0.5:
            aug_eq_image = np.asarray(self.color_aug(transforms.ToPILImage()(eq_image)))
        else:
            aug_eq_image = eq_image

        cb_image = self.e2c.run(eq_image)
        aug_cb_image = self.e2c.run(aug_eq_image)

        sample['eq_image'] = torch.from_numpy(eq_image)
        sample['cb_image'] = torch.from_numpy(cb_image)
        sample['aug_eq_image'] = torch.from_numpy(aug_eq_image)
        sample['aug_cb_image'] = torch.from_numpy(aug_cb_image)

        sample['depth'] = torch.from_numpy(depth)
        
        sample['valid_mask'] = (torch.isnan(sample['depth']) == 0)
        sample['depth'][sample['valid_mask'] == 0] = 0
        
        sample['image_path'] = self.filelist[item].split(' ')[0]
        
        return sample

    def __len__(self):
        return len(self.filelist)