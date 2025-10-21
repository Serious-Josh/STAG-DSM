import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from typing import Optional

# /////////////////////////////////
# ARCHITECTURE 1: ROI + Temporal EfficientNet
# /////////////////////////////////


# EfficientNet backbone with dual-branch head (eyes + mouth) and temporal mean pooling
# Inputs: tensor of [B, C, H, W]/[B, T, C, H, W]
# Outputs: tensor of [B, D], where D is EfficientNet classifier
class EfficientNetFeatureExtractor(nn.Module):

    def __init__(self):
        super().__init__()
        m = torchvision.models.efficientnet_b0(torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        in_feats = m.classifier[1].in_features
        m.classifier = nn.Sequential(nn.Identity())
        self.backbone = m
        self.feature_dim = in_feats

        # ROI branch: project last conv map to 512 channels, then average-pool ROIs
        self.roi_proj = nn.Conv2d(in_feats, 512, kernel_size=1, bias=False)
        self.roi_proj_bn = nn.BatchNorm2d(512)
        # MLP for ROI vectors (shared)
        self.roi_mlp = nn.Sequential(
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.2),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
        )
        # Fusion back to backbone dim so classifier heads remain unchanged
        self.fusion = nn.Linear(in_feats + 256, in_feats)

        # Canonical ROI boxes in 112x112 aligned space (x, y, w, h)
        # These approximate typical eye/mouth positions for a 5pt-aligned face.
        leye = (int(112 * 0.35 - 32), int(112 * 0.46 - 32), 64, 64)
        reye = (int(112 * 0.65 - 32), int(112 * 0.46 - 32), 64, 64)
        mouth = (int(112 * 0.50 - 32), int(112 * 0.82 - 32), 64, 64)
        # Clamp to valid bounds just in case
        def _clamp_box(box):
            x, y, w, h = box
            x = max(0, min(112 - w, x))
            y = max(0, min(112 - h, y))
            return (x, y, w, h)

        leye = _clamp_box(leye)
        reye = _clamp_box(reye)
        mouth = _clamp_box(mouth)
        self.register_buffer('_roi_leye', torch.tensor(leye, dtype=torch.float32))
        self.register_buffer('_roi_reye', torch.tensor(reye, dtype=torch.float32))
        self.register_buffer('_roi_mouth', torch.tensor(mouth, dtype=torch.float32))

    def _pool_roi(self, fmap: torch.Tensor, box_112: torch.Tensor) -> torch.Tensor:
        # fmap: [N, C, Hf, Wf] from last conv; box_112: [4] (x,y,w,h) in 112x112 space
        _, _, Hf, Wf = fmap.shape
        scale_x = Wf / 112.0
        scale_y = Hf / 112.0
        x, y, w, h = box_112
        x1 = torch.clamp(torch.round(x * scale_x).long(), 0, Wf - 1)
        y1 = torch.clamp(torch.round(y * scale_y).long(), 0, Hf - 1)
        x2 = torch.clamp(torch.round((x + w) * scale_x).long(), 0, Wf)
        y2 = torch.clamp(torch.round((y + h) * scale_y).long(), 0, Hf)
        x2 = torch.maximum(x2, x1 + 1)
        y2 = torch.maximum(y2, y1 + 1)
        roi = fmap[:, :, y1:y2, x1:x2]
        return roi.mean(dim=(2, 3))  # [N, C]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Accept [B, C, H, W] or [B, T, C, H, W]
        has_time = (x.dim() == 5)
        if has_time:
            B, T, C, H, W = x.shape
            x = x.reshape(B * T, C, H, W)
        else:
            B, T = x.shape[0], 1

        # Backbone features
        feats_map = self.backbone.features(x)
        g = self.backbone.avgpool(feats_map)
        g = torch.flatten(g, 1)  # [N, feature_dim]

        # ROI features
        z = self.roi_proj_bn(self.roi_proj(feats_map))
        z = F.relu(z, inplace=True)
        leye = self._pool_roi(z, self._roi_leye)
        reye = self._pool_roi(z, self._roi_reye)
        eyes_512 = 0.5 * (leye + reye)
        mouth_512 = self._pool_roi(z, self._roi_mouth)
        eyes_128 = self.roi_mlp(eyes_512)
        mouth_128 = self.roi_mlp(mouth_512)

        fused = torch.cat([g, eyes_128, mouth_128], dim=1)
        fused = self.fusion(fused)

        # Temporal mean pooling over last up to 8 frames
        if T > 1:
            fused = fused.view(B, T, -1)
            if fused.size(1) > 8:
                fused = fused[:, -8:, :]
            fused = fused.mean(dim=1)
        return fused


# (Softmax head removed; we use ArcFace exclusively)


class ArcFaceHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int, m: float = 0.5, s: float = 64.0, eps: float = 1e-7):
        super().__init__()
        self.W = nn.Parameter(torch.randn(num_classes, in_dim))
        nn.init.xavier_uniform_(self.W)
        self.m = float(m)
        self.s = float(s)
        self.eps = float(eps)

    def forward(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        x_norm = F.normalize(x, p=2, dim=1)
        W_norm = F.normalize(self.W, p=2, dim=1)
        cos = torch.matmul(x_norm, W_norm.t())
        if labels is None:
            return cos * self.s
        labels = labels.long()
        idx = torch.arange(x.size(0), device=x.device)
        cos_y = cos[idx, labels].clamp(-1.0 + self.eps, 1.0 - self.eps)
        theta = torch.acos(cos_y)
        cos_m = torch.cos(theta + self.m)
        cos_out = cos.clone()
        cos_out[idx, labels] = cos_m
        return cos_out * self.s


# Not actually used, this is mostly for demonstration. Actual running code instantiates backbone/head seperately.
# Instantiation of the backbone/head.
class DSMModel(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__()
        # EfficientNetFeatureExtractor internally loads ImageNet weights for B0.
        self.backbone = EfficientNetFeatureExtractor()
        in_dim = self.backbone.feature_dim
        self.head_type = 'arcface'
        self.head = ArcFaceHead(in_dim, num_classes)

    @property
    def feature_dim(self) -> int:
        return self.backbone.feature_dim

    def forward(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        feats = self.backbone(x)
        return self.head(feats, labels)



# //////////////////////////////////
# ARCHITECTURE 2: Multi-Scale Fusion EfficientNet
# //////////////////////////////////

# Fuses EfficientNet early/mid/late feature maps
class MultiScaleFusionEfficientNet(nn.Module):

    def __init__(self, num_classes: int, proj_channels: int = 256, fuse_out_channels: int = 256, dropout: float = 0.2, verbose: bool = False):
        super().__init__()
        # EfficientNet-B0 backbone, same weights/preprocessing as existing
        m = torchvision.models.efficientnet_b0(torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        m.classifier = nn.Sequential(nn.Identity())
        self.backbone = m
        self.verbose = bool(verbose)

        # Known channel sizes for B0 taps (features indices: 2, 4, 8)
        c_low, c_mid, c_high = 24, 80, 1280
        self.proj_low = nn.Sequential(nn.Conv2d(c_low, proj_channels, kernel_size=1, bias=False), nn.BatchNorm2d(proj_channels), nn.ReLU(inplace=True))
        self.proj_mid = nn.Sequential(nn.Conv2d(c_mid, proj_channels, kernel_size=1, bias=False), nn.BatchNorm2d(proj_channels), nn.ReLU(inplace=True))
        self.proj_high = nn.Sequential(nn.Conv2d(c_high, proj_channels, kernel_size=1, bias=False), nn.BatchNorm2d(proj_channels), nn.ReLU(inplace=True))

        self.fuse = nn.Sequential(
            nn.Conv2d(proj_channels * 3, fuse_out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(fuse_out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=float(dropout)) if dropout and dropout > 0 else nn.Identity(),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.feature_dim = fuse_out_channels
        self.head_type = 'arcface'
        self.head = ArcFaceHead(self.feature_dim, num_classes)

    def _tap_features_b0(self, x: torch.Tensor):
        # Collect outputs at indices 2 (low), 4 (mid), 8 (high)
        feats = self.backbone.features
        out = x
        f_low = f_mid = f_high = None
        for i, layer in enumerate(feats):
            out = layer(out)
            if i == 2:
                f_low = out
            elif i == 4:
                f_mid = out
            elif i == 8:
                f_high = out
        return f_low, f_mid, f_high

    def forward(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Accept [B, C, H, W]; if [B, T, C, H, W] take the last frame (no temporal pooling here)
        if x.dim() == 5:
            x = x[:, -1]  # last frame
        f_low, f_mid, f_high = self._tap_features_b0(x)
        # Project to common channels
        p_low = self.proj_low(f_low)
        p_mid = self.proj_mid(f_mid)
        p_high = self.proj_high(f_high)
        # Align spatial size to low
        H, W = p_low.shape[2], p_low.shape[3]
        p_mid = F.interpolate(p_mid, size=(H, W), mode='bilinear', align_corners=False)
        p_high = F.interpolate(p_high, size=(H, W), mode='bilinear', align_corners=False)
        fused = torch.cat([p_low, p_mid, p_high], dim=1)
        fused = self.fuse(fused)
        if self.verbose:
            print(f"[MultiScaleFusion] fused feature map: {fused.shape}")
        vec = torch.flatten(self.pool(fused), 1)
        return self.head(vec, labels)

# Model builder
def build_model(*args, **kwargs):
    arch = kwargs.pop('arch', 'roi_temporal')
    # Backwards compatibility: first positional may be num_classes
    num_classes = kwargs.pop('num_classes', None)
    if num_classes is None and args:
        num_classes = args[0]
        args = args[1:]
    if arch == 'roi_temporal':
        return DSMModel(num_classes=num_classes)
    elif arch == 'multiscale':
        return MultiScaleFusionEfficientNet(num_classes=num_classes, **kwargs)
    else:
        raise ValueError("arch must be 'roi_temporal' or 'multiscale'")


def save_checkpoint(model: nn.Module, path: str, arch: str, extra: Optional[dict] = None) -> None:
    state = {
        'arch': arch,
        'head_type': getattr(model, 'head_type', None),
        'feature_dim': getattr(model, 'feature_dim', None),
    }
    # Expect wrapper with backbone and head
    if hasattr(model, 'backbone') and hasattr(model, 'head'):
        state['extractor'] = model.backbone.state_dict()
        state['head'] = model.head.state_dict()
    state.update(extra or {})
    torch.save(state, path)
