import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

def _make_scratch(in_shape, out_shape, groups=1, expand=False):
    scratch = nn.Module()

    out_shape1 = out_shape
    out_shape2 = out_shape
    out_shape3 = out_shape
    if len(in_shape) >= 4:
        out_shape4 = out_shape

    if expand:
        out_shape1 = out_shape
        out_shape2 = out_shape * 2
        out_shape3 = out_shape * 4
        if len(in_shape) >= 4:
            out_shape4 = out_shape * 8

    scratch.layer1_rn = nn.Conv2d(in_shape[0], out_shape1, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
    scratch.layer2_rn = nn.Conv2d(in_shape[1], out_shape2, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
    scratch.layer3_rn = nn.Conv2d(in_shape[2], out_shape3, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
    if len(in_shape) >= 4:
        scratch.layer4_rn = nn.Conv2d(in_shape[3], out_shape4, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)

    return scratch


class ResidualConvUnit(nn.Module):
    """Residual convolution module.
    """

    def __init__(self, features, activation, bn):
        """Init.

        Args:
            features (int): number of features
        """
        super().__init__()

        self.bn = bn

        self.groups=1

        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups)
        
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups)

        if self.bn == True:
            self.bn1 = nn.BatchNorm2d(features)
            self.bn2 = nn.BatchNorm2d(features)

        self.activation = activation

        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: output
        """
        
        out = self.activation(x)
        out = self.conv1(out)
        if self.bn == True:
            out = self.bn1(out)
       
        out = self.activation(out)
        out = self.conv2(out)
        if self.bn == True:
            out = self.bn2(out)

        if self.groups > 1:
            out = self.conv_merge(out)

        return self.skip_add.add(out, x)


class FeatureFusionBlock(nn.Module):
    """Feature fusion block.
    """

    def __init__(
        self, 
        features, 
        activation, 
        deconv=False, 
        bn=False, 
        expand=False, 
        align_corners=True,
        size=None
    ):
        """Init.
        
        Args:
            features (int): number of features
        """
        super(FeatureFusionBlock, self).__init__()

        self.deconv = deconv
        self.align_corners = align_corners

        self.groups=1

        self.expand = expand
        out_features = features
        if self.expand == True:
            out_features = features // 2
        
        self.out_conv = nn.Conv2d(features, out_features, kernel_size=1, stride=1, padding=0, bias=True, groups=1)

        self.CEELayer = CEELayer(features)

        self.resConfUnit1 = ResidualConvUnit(features, activation, bn)
        self.resConfUnit2 = ResidualConvUnit(features, activation, bn)
        
        self.skip_add = nn.quantized.FloatFunctional()

        self.size=size

    def forward(self, *xs, size=None):
        """Forward pass.

        Returns:
            tensor: output
        """
        output = xs[0]

        if len(xs) == 2:
            output = self.CEELayer(output, xs[1])
        if len(xs) == 3:
            res = self.CEELayer(xs[1], xs[2]) 

            res = self.resConfUnit1(res)

            output = self.skip_add.add(output, res)

        output = self.resConfUnit2(output)

        if (size is None) and (self.size is None):
            modifier = {"scale_factor": 2}
        elif size is None:
            modifier = {"size": self.size}
        else:
            modifier = {"size": size}

        output = nn.functional.interpolate(output, **modifier, mode="bilinear", align_corners=self.align_corners)
        
        output = self.out_conv(output)

        return output

# from https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py
class SELayer(nn.Module):
    def __init__(self, channel, reduction=14):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class CEELayer(nn.Module):
    def __init__(self, channels, SE=True):
        super(CEELayer, self).__init__()

        self.res_conv1 = nn.Conv2d(channels * 2, channels, kernel_size=1, padding=0, bias=False)
        self.res_bn1 = nn.BatchNorm2d(channels)

        self.res_conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.res_bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

        self.SE = SE
        if self.SE:
            self.selayer = SELayer(channels * 2)

        self.conv = nn.Conv2d(channels * 2, channels, 1, bias=False)

    def forward(self, equi_feat, c2e_feat):

        x = torch.cat([equi_feat, c2e_feat], 1)
        
        x = self.relu(self.res_bn1(self.res_conv1(x)))
        shortcut = self.res_bn2(self.res_conv2(x))

        x = c2e_feat + shortcut
        x = torch.cat([equi_feat, x], 1)

        if self.SE:
            x = self.selayer(x)
        
        x = self.relu(self.conv(x))
        
        return x
    
# Based on https://github.com/sunset1995/py360convert
class Cube2Equirec(nn.Module):
    def __init__(self, face_w, equ_h, equ_w):
        super(Cube2Equirec, self).__init__()
        '''
        face_w: int, the length of each face of the cubemap
        equ_h: int, height of the equirectangular image
        equ_w: int, width of the equirectangular image
        '''

        self.face_w = face_w
        self.equ_h = equ_h
        self.equ_w = equ_w


        # Get face id to each pixel: 0F 1R 2B 3L 4U 5D
        self._equirect_facetype()
        self._equirect_faceuv()


    def _equirect_facetype(self):
        '''
        0F 1R 2B 3L 4U 5D
        '''
        tp = np.roll(np.arange(4).repeat(self.equ_w // 4)[None, :].repeat(self.equ_h, 0), 3 * self.equ_w // 8, 1)

        # Prepare ceil mask
        mask = np.zeros((self.equ_h, self.equ_w // 4), np.bool_)
        idx = np.linspace(-np.pi, np.pi, self.equ_w // 4) / 4
        idx = self.equ_h // 2 - np.round(np.arctan(np.cos(idx)) * self.equ_h / np.pi).astype(int)
        for i, j in enumerate(idx):
            mask[:j, i] = 1
        mask = np.roll(np.concatenate([mask] * 4, 1), 3 * self.equ_w // 8, 1)

        tp[mask] = 4
        tp[np.flip(mask, 0)] = 5

        self.tp = tp
        self.mask = mask

    def _equirect_faceuv(self):

        lon = ((np.linspace(0, self.equ_w -1, num=self.equ_w, dtype=np.float32 ) +0.5 ) /self.equ_w - 0.5 ) * 2 *np.pi
        lat = -((np.linspace(0, self.equ_h -1, num=self.equ_h, dtype=np.float32 ) +0.5 ) /self.equ_h -0.5) * np.pi

        lon, lat = np.meshgrid(lon, lat)

        coor_u = np.zeros((self.equ_h, self.equ_w), dtype=np.float32)
        coor_v = np.zeros((self.equ_h, self.equ_w), dtype=np.float32)

        for i in range(4):
            mask = (self.tp == i)
            coor_u[mask] = 0.5 * np.tan(lon[mask] - np.pi * i / 2)
            coor_v[mask] = -0.5 * np.tan(lat[mask]) / np.cos(lon[mask] - np.pi * i / 2)

        mask = (self.tp == 4)
        c = 0.5 * np.tan(np.pi / 2 - lat[mask])
        coor_u[mask] = c * np.sin(lon[mask])
        coor_v[mask] = c * np.cos(lon[mask])

        mask = (self.tp == 5)
        c = 0.5 * np.tan(np.pi / 2 - np.abs(lat[mask]))
        coor_u[mask] = c * np.sin(lon[mask])
        coor_v[mask] = -c * np.cos(lon[mask])

        # Final renormalize
        coor_u = (np.clip(coor_u, -0.5, 0.5)) * 2
        coor_v = (np.clip(coor_v, -0.5, 0.5)) * 2

        # Convert to torch tensor
        self.tp = torch.from_numpy(self.tp.astype(np.float32) / 2.5 - 1)
        self.coor_u = torch.from_numpy(coor_u)
        self.coor_v = torch.from_numpy(coor_v)

        sample_grid = torch.stack([self.coor_u, self.coor_v, self.tp], dim=-1).view(1, 1, self.equ_h, self.equ_w, 3)
        self.sample_grid = nn.Parameter(sample_grid, requires_grad=False)

    def forward(self, cube_feat):

        bs, ch, h, w = cube_feat.shape

        #print(h, self.face_w)
        #print(w, self.face_w)

        assert h == self.face_w and w // 6 == self.face_w

        cube_feat = cube_feat.view(bs, ch, 1,  h, w)
        cube_feat = torch.cat(torch.split(cube_feat, self.face_w, dim=-1), dim=2)

        cube_feat = cube_feat.view([bs, ch, 6, self.face_w, self.face_w])
        sample_grid = torch.cat(bs * [self.sample_grid], dim=0)
        equi_feat = F.grid_sample(cube_feat, sample_grid, padding_mode="border", align_corners=True)

        return equi_feat.squeeze(2)