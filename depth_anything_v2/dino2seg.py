import math
import os

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

from dinov2 import DINOv2
from util.blocks import FeatureFusionBlock, _make_scratch
from base import ConvBlock
from torchvision.transforms import Compose
from util.transform import Resize, NormalizeImage, PrepareForNet

class DPTSegmentationHead(nn.Module):
    def __init__(
            self,
            in_channels=768,  # transformer embedding dim
            features=256,  # decoder channels
            out_channels=[256, 512, 1024, 1024],  # for the 4 scale projections
            num_classes=40,  # NYUDv2
            use_bn=False,
            use_clstoken=False
    ):
        super().__init__()
        self.use_clstoken = use_clstoken

        self.projects = nn.ModuleList([
            nn.Conv2d(in_channels, out_ch, kernel_size=1)
            for out_ch in out_channels
        ])
        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(out_channels[0], out_channels[0], kernel_size=4, stride=4),
            nn.ConvTranspose2d(out_channels[1], out_channels[1], kernel_size=2, stride=2),
            nn.Identity(),
            nn.Conv2d(out_channels[3], out_channels[3], kernel_size=3, stride=2, padding=1)
        ])

        if use_clstoken:
            self.readout_projects = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(2 * in_channels, in_channels),
                    nn.GELU()
                ) for _ in range(4)
            ])

        self.scratch = _make_scratch(out_channels, features, groups=1, expand=False)
        self.scratch.refinenet1 = FeatureFusionBlock(features, nn.ReLU(), bn=use_bn)
        self.scratch.refinenet2 = FeatureFusionBlock(features, nn.ReLU(), bn=use_bn)
        self.scratch.refinenet3 = FeatureFusionBlock(features, nn.ReLU(), bn=use_bn)
        self.scratch.refinenet4 = FeatureFusionBlock(features, nn.ReLU(), bn=use_bn)

        head_features_1 = features
        head_features_2 = 32

        # reduce channels
        self.scratch.output_conv1 = nn.Conv2d(
            head_features_1, head_features_1 // 2,
            kernel_size=3, stride=1, padding=1
        )

        # final segmentation prediction
        self.scratch.output_conv2 = nn.Sequential(
            nn.Conv2d(head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(head_features_2, num_classes, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, out_features, ph, pw):
        out = []
        for i, x in enumerate(out_features):
            if self.use_clstoken:
                x, cls_token = x[0], x[1]
                readout = cls_token.unsqueeze(1).expand_as(x)
                x = self.readout_projects[i](torch.cat((x, readout), -1))
            else:
                x = x[0]  # (B, N, C)

            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], ph, pw))
            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            out.append(x)

        layer_1, layer_2, layer_3, layer_4 = out

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        out = self.scratch.output_conv1(path_1)
        out = F.interpolate(out, (int(ph * 14), int(pw * 14)), mode="bilinear", align_corners=True)
        out = self.scratch.output_conv2(out)

        return out  # raw logits (B, num_classes, H, W)


class Dino2Seg(nn.Module):
    def __init__(
            self,
            encoder='vitb',
            num_classes=40,
            image_height=476,
            image_width=630,
            features=768,
            out_channels=[256, 512, 1024, 1024],
            use_bn=False,
            use_clstoken=False,
            model_weights_dir="",
            device="cuda",
    ):
        super(Dino2Seg, self).__init__()

        self.intermediate_layer_idx = {
            'vits': [2, 5, 8, 11],
            'vitb': [2, 5, 8, 11],
            'vitl': [4, 11, 17, 23],
            'vitg': [9, 19, 29, 39]
        }
        self.image_height = image_height
        self.image_width = image_width

        self.encoder = encoder
        self.pretrained = DINOv2(model_name=encoder)
        self.device = device

        # ──────────────── WEIGHT LOADING LOGIC ──────────────── #
        vitb_weight_file = None
        seg_weight_file = None

        if model_weights_dir and os.path.isdir(model_weights_dir):
            files = os.listdir(model_weights_dir)
            # Find backbone
            for f in files:
                if "depth_anything" in f and (f.endswith(".pth") or f.endswith(".pt")):
                    vitb_weight_file = os.path.join(model_weights_dir, f)
                if "seg" in f and (f.endswith(".pth") or f.endswith(".pt")):
                    seg_weight_file = os.path.join(model_weights_dir, f)

        # Load backbone weights if available
        if vitb_weight_file:
            print(f"Loading ViT-b backbone weights from: {vitb_weight_file}")
            state_dict = torch.load(vitb_weight_file, map_location='cpu')
            adjusted_state_dict = {
                k.replace('pretrained.', ''): v
                for k, v in state_dict.items() if k.startswith('pretrained.')
            }
            missing_keys, unexpected_keys = self.pretrained.load_state_dict(adjusted_state_dict, strict=False)
            if missing_keys:
                print("[Backbone] Missing keys:", missing_keys)
            if unexpected_keys:
                print("[Backbone] Unexpected keys:", unexpected_keys)
            self.pretrained.eval()
            for param in self.pretrained.parameters():
                param.requires_grad = False
        else:
            print("No ViT-b backbone weights found.")

        # Setup segmentation head
        self.seg_head = DPTSegmentationHead(
            in_channels=features,
            num_classes=num_classes,
            out_channels=out_channels,
            use_bn=use_bn,
            use_clstoken=use_clstoken,
        )

        # Load segmentation head weights if available
        if seg_weight_file:
            print(f"Loading segmentation head weights from: {seg_weight_file}")
            seg_state_dict = torch.load(seg_weight_file, map_location='cpu')
            missing_keys, unexpected_keys = self.seg_head.load_state_dict(seg_state_dict, strict=False)
            # print warnings if keys don't match exactly
            if missing_keys:
                print("[Seg Head] Missing keys:", missing_keys)
            if unexpected_keys:
                print("[Seg Head] Unexpected keys:", unexpected_keys)
        else:
            print("No segmentation head weights found.")

        self.pretrained.to(self.device)
        self.seg_head.to(self.device)


    def forward(self, x):
        patch_h, patch_w = x.shape[-2] // 14, x.shape[-1] // 14

        features = self.pretrained.get_intermediate_layers(x, self.intermediate_layer_idx[self.encoder],
                                                           return_class_token=True)

        seg_logits = self.seg_head(
            out_features=features,
            ph=patch_h,
            pw=patch_w
        )

        return seg_logits.squeeze(1)

    @torch.no_grad()
    def infer_image(self, image):
        seg_logits = self.forward(image)
        seg_probs = F.softmax(seg_logits, dim=1)
        segmentation_pred = torch.argmax(seg_probs, dim=1)

        return seg_probs, segmentation_pred.cpu().numpy()

    # def image2tensor(self, raw_image, input_size=518):
    #     transform = Compose([
    #         Resize(
    #             width=input_size,
    #             height=input_size,
    #             resize_target=False,
    #             keep_aspect_ratio=True,
    #             ensure_multiple_of=14,
    #             resize_method='lower_bound',
    #             image_interpolation_method=cv2.INTER_CUBIC,
    #         ),
    #         NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #         PrepareForNet(),
    #     ])
    #
    #     h, w = raw_image.shape[:2]
    #
    #     image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0
    #
    #     image = transform({'image': image})['image']
    #     image = torch.from_numpy(image).unsqueeze(0)
    #
    #     DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    #     image = image.to(DEVICE)
    #
    #     return image, (h, w)