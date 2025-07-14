import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2

from dino2seg import DPTSegmentationHead
from dpt import DPTHead
from torchvision.transforms import Compose

from dinov2 import DINOv2
from util.transform import Resize, NormalizeImage, PrepareForNet
from util.blocks import FeatureFusionBlock, _make_scratch

class SegmentationDeformableDepth(nn.Module):
    def __init__(
            self,
            encoder='vitb',
            num_classes=6,
            image_height=476,
            image_width=630,
            features=768,
            out_channels=[256, 512, 1024, 1024],
            use_bn=False,
            use_clstoken=False,
            model_weights_dir="",
            device="cuda",
    ):
        super(SegmentationDeformableDepth, self).__init__()

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
        depth_weight_file = None

        if model_weights_dir and os.path.isdir(model_weights_dir):
            files = os.listdir(model_weights_dir)
            # Find backbone
            for f in files:
                if "vitb" in f and (f.endswith(".pth") or f.endswith(".pt")):
                    vitb_weight_file = os.path.join(model_weights_dir, f)
                if "seg" in f and (f.endswith(".pth") or f.endswith(".pt")):
                    seg_weight_file = os.path.join(model_weights_dir, f)
                if "depth" in f and (f.endswith(".pth") or f.endswith(".pt")):
                    depth_weight_file = os.path.join(model_weights_dir, f)

        # Load backbone weights if available
        if vitb_weight_file:
            print(f"Loading ViT-b backbone weights from: {depth_weight_file}")
            state_dict = torch.load(depth_weight_file, map_location='cpu')
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

        # Setup depth head
        self.depth_head = DPTHead(
            in_channels=768,  # ViT-B output dimension
            features=128,  # internal refinenet channels
            use_bn=False,
            out_channels=[96, 192, 384, 768],  # projections expected by checkpoint
            use_clstoken=False,
        )

        # Load depth head weights if available
        if depth_weight_file:
            print(f"Loading depth head weights from: {depth_weight_file}")
            depth_state_dict = torch.load(depth_weight_file, map_location='cpu')

            adjusted_state_dict = {
                k.replace('depth_head.', ''): v
                for k, v in depth_state_dict.items() if k.startswith('depth_head.')
            }
            missing_keys, unexpected_keys = self.depth_head.load_state_dict(adjusted_state_dict, strict=False)

            # print warnings if keys don't match exactly
            if missing_keys:
                print("[Depth Head] Missing keys:", missing_keys)
            if unexpected_keys:
                print("[Depth Head] Unexpected keys:", unexpected_keys)
        else:
            print("No depth head weights found.")

        self.pretrained.to(self.device)
        self.seg_head.to(self.device)
        self.depth_head.to(self.device)


    def forward(self, x):
        patch_h, patch_w = x.shape[-2] // 14, x.shape[-1] // 14

        features = self.pretrained.get_intermediate_layers(x, self.intermediate_layer_idx[self.encoder],
                                                           return_class_token=True)

        # depth = self.depth_head(features, patch_h, patch_w)
        # depth = F.relu(depth)

        seg_logits = self.seg_head(
            out_features=features,
            ph=patch_h,
            pw=patch_w
        )

        return None, seg_logits.squeeze(1)

    @torch.no_grad()
    def infer_image(self, image):
        depth, seg_logits = self.forward(image)

        seg_probs = F.softmax(seg_logits, dim=1)
        segmentation_pred = torch.argmax(seg_probs, dim=1)

        return None, segmentation_pred.cpu().numpy()

    def image2tensor(self, raw_image, input_size=518):
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

        h, w = raw_image.shape[:2]

        image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0

        image = transform({'image': image})['image']
        image = torch.from_numpy(image).unsqueeze(0)

        DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        image = image.to(DEVICE)

        return image, (h, w)