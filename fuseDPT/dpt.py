import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose

from .dinov2 import DINOv2
from .util.blocks import FeatureFusionBlock, _make_scratch
from .util.transform import Resize, NormalizeImage, PrepareForNet

from fuseDPT.util.utils import Equirec2Cube
from fuseDPT.util.blocks import Cube2Equirec


def _make_fusion_block(features, use_bn, size=None):
    return FeatureFusionBlock(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
        size=size,
    )


class ConvBlock(nn.Module):
    def __init__(self, in_feature, out_feature):
        super().__init__()
        
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_feature, out_feature, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_feature),
            nn.ReLU(True)
        )
    
    def forward(self, x):
        return self.conv_block(x)


class DPTHead(nn.Module):
    def __init__(
        self, 
        in_channels, 
        features=256, 
        use_bn=False, 
        out_channels=[256, 512, 1024, 1024], 
        use_clstoken=False,
        size=(560,1120)
    ):
        super(DPTHead, self).__init__()

        self.equi_h = size[0]
        self.equi_w = size[1]
        self.cube_h = size[0] // 2
        
        self.use_clstoken = use_clstoken
        
        self.projects = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channel,
                kernel_size=1,
                stride=1,
                padding=0,
            ) for out_channel in out_channels
        ])
        
        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(
                in_channels=out_channels[0],
                out_channels=out_channels[0],
                kernel_size=4,
                stride=4,
                padding=0),
            nn.ConvTranspose2d(
                in_channels=out_channels[1],
                out_channels=out_channels[1],
                kernel_size=2,
                stride=2,
                padding=0),
            nn.Identity(),
            nn.Conv2d(
                in_channels=out_channels[3],
                out_channels=out_channels[3],
                kernel_size=3,
                stride=2,
                padding=1)
        ])
        
        if use_clstoken:
            self.readout_projects = nn.ModuleList()
            for _ in range(len(self.projects)):
                self.readout_projects.append(
                    nn.Sequential(
                        nn.Linear(2 * in_channels, in_channels),
                        nn.GELU()))
        
        self.eq_scratch = _make_scratch(
            out_channels,
            features,
            groups=1,
            expand=False,
        )

        self.cm_scratch = _make_scratch(
            out_channels,
            features,
            groups=1,
            expand=False,
        )
        
        self.eq_scratch.stem_transpose = None
        self.cm_scratch.stem_transpose = None
        
        self.eq_scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.eq_scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.eq_scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.eq_scratch.refinenet4 = _make_fusion_block(features, use_bn)

        self.cm_scratch.e2c_4 = Cube2Equirec(self.cube_h // 28, self.equi_h // 28, self.equi_w // 28)
        self.cm_scratch.e2c_3 = Cube2Equirec(self.cube_h // 14, self.equi_h // 14, self.equi_w // 14)
        self.cm_scratch.e2c_2 = Cube2Equirec(self.cube_h // 7, self.equi_h // 7, self.equi_w // 7)
        self.cm_scratch.e2c_1 = Cube2Equirec(int(self.cube_h / 3.5), int(self.equi_h / 3.5), int(self.equi_w / 3.5))

        head_features_1 = features
        head_features_2 = 32
        
        self.eq_scratch.output_conv1 = nn.Conv2d(head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1)
        self.eq_scratch.output_conv2 = nn.Sequential(
            nn.Conv2d(head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1),
            #nn.LayerNorm((head_features_2, size[0] // 2, size[1] // 2)),
            nn.ReLU(True),
            nn.Conv2d(head_features_2, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True),
            nn.Identity(),
        )
    
    def forward(self, erp_features, cmp_features, patch_h, patch_w, cb_patch_h, cb_patch_w):
        eq_out = []

        for i, x in enumerate(erp_features):
            if self.use_clstoken:
                x, cls_token = x[0], x[1]
                readout = cls_token.unsqueeze(1).expand_as(x)
                x = self.readout_projects[i](torch.cat((x, readout), -1))
            else:
                x = x[0]
            
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))
            
            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            
            eq_out.append(x)

        cm_out = []

        for i, x in enumerate(cmp_features):
            if self.use_clstoken:
                x, cls_token = x[0], x[1]
                readout = cls_token.unsqueeze(1).expand_as(x)
                x = self.readout_projects[i](torch.cat((x, readout), -1))
            else:
                x = x[0]
            
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], cb_patch_h, cb_patch_w))
            
            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            
            cm_out.append(x)
        
        eq_layer_1, eq_layer_2, eq_layer_3, eq_layer_4 = eq_out
        
        eq_layer_1_rn = self.eq_scratch.layer1_rn(eq_layer_1)
        eq_layer_2_rn = self.eq_scratch.layer2_rn(eq_layer_2)
        eq_layer_3_rn = self.eq_scratch.layer3_rn(eq_layer_3)
        eq_layer_4_rn = self.eq_scratch.layer4_rn(eq_layer_4)

        cm_layer_1, cm_layer_2, cm_layer_3, cm_layer_4 = cm_out
        
        cm_layer_1_rn = self.cm_scratch.layer1_rn(cm_layer_1)
        cm_layer_2_rn = self.cm_scratch.layer2_rn(cm_layer_2)
        cm_layer_3_rn = self.cm_scratch.layer3_rn(cm_layer_3)
        cm_layer_4_rn = self.cm_scratch.layer4_rn(cm_layer_4)

        e2c_layer_4 = self.cm_scratch.e2c_4(cm_layer_4_rn)
        path_4 = self.eq_scratch.refinenet4(eq_layer_4_rn, e2c_layer_4, size=eq_layer_3_rn.shape[2:])       
        e2c_layer_3 = self.cm_scratch.e2c_3(cm_layer_3_rn) 
        path_3 = self.eq_scratch.refinenet3(path_4, eq_layer_3_rn, e2c_layer_3, size=eq_layer_2_rn.shape[2:])
        e2c_layer_2 = self.cm_scratch.e2c_2(cm_layer_2_rn) 
        path_2 = self.eq_scratch.refinenet2(path_3, eq_layer_2_rn, e2c_layer_2, size=eq_layer_1_rn.shape[2:])
        e2c_layer_1 = self.cm_scratch.e2c_1(cm_layer_1_rn) 
        path_1 = self.eq_scratch.refinenet1(path_2, eq_layer_1_rn, e2c_layer_1)

        #print("End of path:", path_1)
        
        out = self.eq_scratch.output_conv1(path_1)
        #print('Before interp', out.max())
        #print(out.min())
        
        #print("First", out)
        out = F.interpolate(out, (int(patch_h * 14), int(patch_w * 14)), mode="bilinear", align_corners=True)
        #print('After interp', out.max())
        #print(out.min())
        out = self.eq_scratch.output_conv2(out)
        #print('After conv', out.max())
        #print(out.min())
        #print("second", out)
        #print(out)
        
        return out


class FuseDPT(nn.Module):
    def __init__(
        self, 
        encoder='vitl', 
        features=256, 
        out_channels=[256, 512, 1024, 1024], 
        use_bn=False, 
        use_clstoken=False
    ):
        super(FuseDPT, self).__init__()
        
        self.intermediate_layer_idx = {
            'vits': [2, 5, 8, 11],
            'vitb': [2, 5, 8, 11], 
            'vitl': [4, 11, 17, 23], 
            'vitg': [9, 19, 29, 39]
        }
        
        self.encoder = encoder
        self.erp_pretrained = DINOv2(model_name=encoder)
        self.cmp_pretrained = DINOv2(model_name=encoder)

        self.h, self.w = 0, 0

        self.e2c = None
        
        self.depth_head = DPTHead(self.erp_pretrained.embed_dim, features, use_bn, out_channels=out_channels, use_clstoken=use_clstoken)
    
    def forward(self, x, y):
        patch_h, patch_w = x.shape[-2] // 14, x.shape[-1] // 14
        cb_patch_h, cb_patch_w = y.shape[-2] // 14, y.shape[-1] // 14

        #print(patch_h, patch_w)
        #print(cb_patch_h, cb_patch_w)
        
        erp_pretrained = self.erp_pretrained.get_intermediate_layers(x, self.intermediate_layer_idx[self.encoder], return_class_token=True)
        cmp_pretrained = self.cmp_pretrained.get_intermediate_layers(y, self.intermediate_layer_idx[self.encoder], return_class_token=True)

        depth = self.depth_head(erp_pretrained, cmp_pretrained, patch_h, patch_w, cb_patch_h, cb_patch_w)

        #print(depth.max())
        #print(depth.min())

        depth = F.relu(depth)

        #print(depth.max())
        #print(depth.min())
        
        return depth.squeeze(1)

    @torch.no_grad()
    def infer_image(self, raw_image, input_size=560):
        image, cb_image, (h, w) = self.image2tensor(raw_image, input_size)
        
        depth = self.forward(image, cb_image)
        
        depth = F.interpolate(depth[:, None], (h, w), mode="bilinear", align_corners=True)[0, 0]
        
        return depth.cpu().numpy()
        
    
    def image2tensor(self, raw_image, input_size=560):        
        transform = Compose([
            Resize(
                width=input_size,
                height=input_size,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])

        cb_transform = Compose([
            Resize(
                width=input_size // 2,
                height=input_size // 2,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])

        self.h, self.w = raw_image.shape[:2]

        if not self.e2c:
            self.e2c = Equirec2Cube(self.h, self.w, self.h // 2)
        
        image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0
        
        image = transform({'image': image})['image']
        cb_image = self.e2c.run(image.transpose(1,2,0))

        cb_image = cb_transform({'image': cb_image})['image']

        image = torch.from_numpy(image).unsqueeze(0)
        cb_image = torch.from_numpy(cb_image).unsqueeze(0)
        
        DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        image = image.to(DEVICE)
        cb_image = cb_image.to(DEVICE)
        
        return image, cb_image, (self.h, self.w)
