import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from typing import Optional


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


class SoftmaxHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int, p_drop: float = 0.2):
        super().__init__()
        self.bn = nn.BatchNorm1d(in_dim)
        self.drop = nn.Dropout(p_drop)
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn(x)
        x = self.drop(x)
        return self.fc(x)


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
    def __init__(self, num_classes: int, head: str = 'arcface', pretrained: bool = True):
        super().__init__()
        # EfficientNetFeatureExtractor internally loads ImageNet weights for B0.
        self.backbone = EfficientNetFeatureExtractor()
        in_dim = self.backbone.feature_dim
        head = (head or 'arcface').lower()
        if head == 'softmax':
            self.head_type = 'softmax'
            self.head = SoftmaxHead(in_dim, num_classes)
        elif head == 'arcface':
            self.head_type = 'arcface'
            self.head = ArcFaceHead(in_dim, num_classes)
        else:
            raise ValueError("head must be 'softmax' or 'arcface'")

    @property
    def feature_dim(self) -> int:
        return self.backbone.feature_dim

    def forward(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        feats = self.backbone(x)
        if self.head_type == 'softmax':
            return self.head(feats)
        else:
            return self.head(feats, labels)


def build_model(num_classes: int, head: str = 'arcface', pretrained: bool = True) -> DSMModel:
    return DSMModel(num_classes=num_classes, head=head, pretrained=pretrained)
