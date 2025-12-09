# Local server script to run on dedicated hardware
# Runs inference based on trained checkpoint


import argparse
import io
import logging
import os
import time
from collections import deque
from typing import Callable, Deque, Dict, List, Optional, Sequence, Tuple, cast

from fastapi import FastAPI, File, HTTPException, UploadFile
import numpy as np
from PIL import Image, ImageOps
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import uvicorn

try:
    import cv2  # type: ignore
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore


DEFAULT_DMD_CLASS_NAMES: List[str] = [
    "eyes open",
    "eyes closed",
    "eyes undefined",
    "blinking",
    "yawning with hand",
    "yawning no hand",
]

try:
    PIL_RESAMPLE_BILINEAR = Image.Resampling.BILINEAR  # type: ignore[attr-defined]
except AttributeError:  # pragma: no cover
    PIL_RESAMPLE_BILINEAR = Image.BILINEAR  # type: ignore[attr-defined]


def select_device(preferred: str = "auto") -> torch.device:
    pref = (preferred or "auto").lower()

    def _mps_available() -> bool:
        try:
            mps = getattr(torch.backends, "mps", None)
            return bool(mps) and bool(getattr(mps, "is_available", lambda: False)())
        except Exception:
            return False

    if pref == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if pref == "mps" and _mps_available():
        return torch.device("mps")
    if pref == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if _mps_available():
            return torch.device("mps")
    return torch.device("cpu")


def _reference_five_point_112(margin: float = 0.1) -> np.ndarray:
    ref = np.array(
        [
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041],
        ],
        dtype=np.float32,
    )
    out = 112.0
    center = np.array([out / 2.0, out / 2.0], dtype=np.float32)
    scale = max(0.0, 1.0 - float(margin))
    ref_scaled = center + (ref - center) * scale
    return ref_scaled.astype(np.float32)


def roi_boxes_from_landmarks(
    pts_t: Optional[np.ndarray], out_size: int = 112, eye_size: int = 64, mouth_size: int = 64
) -> Tuple[Optional[Tuple[int, int, int, int]], Optional[Tuple[int, int, int, int]], Optional[Tuple[int, int, int, int]]]:
    def clamp_box(cx: float, cy: float, bw: int, bh: int, W: int, H: int):
        x = int(round(cx - bw / 2))
        y = int(round(cy - bh / 2))
        x = max(0, min(W - bw, x))
        y = max(0, min(H - bh, y))
        return (x, y, bw, bh)

    if pts_t is None or not isinstance(pts_t, np.ndarray) or pts_t.shape != (5, 2):
        return (None, None, None)
    W = H = out_size
    left_eye = pts_t[0]
    right_eye = pts_t[1]
    mouth_c = (pts_t[3] + pts_t[4]) / 2.0
    le = clamp_box(float(left_eye[0]), float(left_eye[1]), eye_size, eye_size, W, H)
    re = clamp_box(float(right_eye[0]), float(right_eye[1]), eye_size, eye_size, W, H)
    mo = clamp_box(float(mouth_c[0]), float(mouth_c[1]), mouth_size, mouth_size, W, H)
    return (le, re, mo)


def _center_fit(img: Image.Image, out_size: int) -> Image.Image:
    try:
        return ImageOps.fit(img, (out_size, out_size), method=PIL_RESAMPLE_BILINEAR, centering=(0.5, 0.5))
    except Exception:
        return img.resize((out_size, out_size))


def align_face_5pt(img: Image.Image, pts5: np.ndarray, out_size: int = 112, margin: float = 0.1) -> Tuple[Image.Image, Optional[np.ndarray]]:
    if cv2 is None:
        return (_center_fit(img, out_size), None)
    pts_src = np.asarray(pts5, dtype=np.float32)
    if pts_src.shape != (5, 2):
        return (_center_fit(img, out_size), None)

    ref = _reference_five_point_112(margin=margin)
    M, _ = cv2.estimateAffinePartial2D(pts_src, ref, method=cv2.LMEDS)
    M = np.asarray(M, dtype=np.float32) if M is not None else None
    if M is None:
        return (_center_fit(img, out_size), None)

    img_np = np.array(img.convert("RGB"))
    aligned_np = cv2.warpAffine(
        img_np,
        M,
        (out_size, out_size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    pts_aug = np.hstack([pts_src, np.ones((5, 1), dtype=np.float32)])
    pts_t = (M @ pts_aug.T).T[:, :2]
    return (Image.fromarray(aligned_np), pts_t)


def warp_to_canonical_roi(img: Image.Image, pts_t: Optional[np.ndarray], out_size: int = 112) -> Image.Image:
    if cv2 is None or pts_t is None:
        return img
    if not isinstance(pts_t, np.ndarray) or pts_t.shape != (5, 2):
        return img
    try:
        src_pts = np.vstack(
            [
                np.asarray(pts_t[0], dtype=np.float32),
                np.asarray(pts_t[1], dtype=np.float32),
                np.asarray((pts_t[3] + pts_t[4]) * 0.5, dtype=np.float32),
            ]
        )
        ref_pts = _reference_five_point_112(margin=0.1)
        left_eye = ref_pts[0]
        right_eye = ref_pts[1]
        mouth = np.mean(ref_pts[3:5], axis=0)
        scale = float(out_size) / 112.0
        dst_pts = np.array(
            [
                left_eye * scale,
                right_eye * scale,
                mouth * scale,
            ],
            dtype=np.float32,
        )
        M = cv2.getAffineTransform(src_pts, dst_pts)
        img_rgb = np.array(img.convert("RGB"))
        warped = cv2.warpAffine(
            img_rgb,
            M,
            (out_size, out_size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        return Image.fromarray(warped)
    except Exception:
        return img


class YuNetDetector:
    def __init__(
        self,
        model_path: str = "face_detection_yunet.onnx",
        input_size: Tuple[int, int] = (320, 320),
        conf_threshold: float = 0.6,
        nms_threshold: float = 0.3,
        top_k: int = 5000,
    ):
        if cv2 is None:
            raise RuntimeError("OpenCV is required for YuNet detector.")
        face_detector_factory = getattr(cv2, "FaceDetectorYN_create", None)
        if face_detector_factory is None:
            raise RuntimeError("This OpenCV build lacks FaceDetectorYN.")
        mp = model_path
        if not os.path.isabs(mp):
            cwd_try = os.path.abspath(mp)
            file_try = os.path.join(os.path.dirname(os.path.abspath(__file__)), mp)
            if os.path.isfile(cwd_try):
                mp = cwd_try
            elif os.path.isfile(file_try):
                mp = file_try
        if not os.path.isfile(mp):
            raise FileNotFoundError(f"YuNet ONNX not found: {mp}")
        self.input_size = (int(input_size[0]), int(input_size[1]))
        self.detector = face_detector_factory(
            mp, "", self.input_size, float(conf_threshold), float(nms_threshold), int(top_k)
        )

    def set_input_size(self, w: int, h: int) -> None:
        if (w, h) != tuple(self.input_size):
            self.detector.setInputSize((int(w), int(h)))
            self.input_size = (int(w), int(h))

    def detect(self, img_bgr: np.ndarray):
        if img_bgr is None or img_bgr.size == 0:
            return []
        h, w = img_bgr.shape[:2]
        if h <= 1 or w <= 1:
            return []
        self.set_input_size(w, h)
        try:
            res = self.detector.detect(img_bgr)
        except Exception:
            return []
        faces = None
        if res is None:
            faces = None
        elif isinstance(res, tuple):
            if len(res) == 0:
                faces = None
            elif len(res) == 1:
                faces = res[0]
            else:
                for elem in res:
                    if hasattr(elem, "shape"):
                        faces = elem
                        break
                if faces is None:
                    faces = res[0]
        else:
            faces = res
        if faces is None or not hasattr(faces, "shape") or faces.size == 0:
            return []
        out = []
        for f in np.asarray(faces):
            vals = f.flatten().tolist()
            if len(vals) < 15:
                continue
            kps = [
                (vals[4], vals[5]),
                (vals[6], vals[7]),
                (vals[8], vals[9]),
                (vals[10], vals[11]),
                (vals[12], vals[13]),
            ]
            score = float(vals[14])
            out.append({"kps": np.array(kps, dtype=np.float32), "score": score})
        out.sort(key=lambda d: d.get("score", 0.0), reverse=True)
        return out


class FaceAligner:
    def __init__(self, target_size: int = 112, enable_alignment: bool = True, yunet_path: str = "face_detection_yunet.onnx"):
        self.target_size = int(target_size)
        self.enable_alignment = bool(enable_alignment)
        self.yunet_path = yunet_path
        self.detector: Optional[YuNetDetector] = None
        if self.enable_alignment and cv2 is not None:
            try:
                self.detector = YuNetDetector(model_path=yunet_path, input_size=(320, 320), conf_threshold=0.25, nms_threshold=0.3, top_k=5000)
                logging.info("[FaceAligner] Using YuNet for alignment.")
            except Exception as exc:
                logging.warning("[FaceAligner] %s. Falling back to center crop.", exc)
                self.detector = None

    def __call__(self, frame_bgr: np.ndarray) -> Tuple[Image.Image, Optional[np.ndarray]]:
        if frame_bgr is None or frame_bgr.size == 0:
            raise ValueError("Empty frame for alignment")
        img_rgb = frame_bgr[:, :, ::-1]
        pil = Image.fromarray(img_rgb)
        if self.detector is None:
            return (_center_fit(pil, self.target_size), None)
        try:
            dets = self.detector.detect(frame_bgr)
            if dets:
                pts = dets[0]["kps"]
                aligned, pts_t = align_face_5pt(pil, pts, out_size=self.target_size, margin=0.1)
                return (aligned, pts_t)
        except Exception as exc:
            logging.warning("[FaceAligner] detection failed: %s", exc)
        return (_center_fit(pil, self.target_size), None)


class EfficientNetFeatureExtractor(nn.Module):

    def __init__(self, variant: str = "b0", pretrained: bool = False):
        super().__init__()
        variant = (variant or "b0").lower()
        weights = torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        m = torchvision.models.efficientnet_b0(weights=weights)

        in_feats = cast(int, getattr(m.classifier[1], "in_features"))
        m.classifier = nn.Sequential(nn.Identity())
        self.backbone = m
        self.feature_dim = in_feats
        self.roi_proj = nn.Conv2d(in_feats, 512, kernel_size=1, bias=False)
        self.roi_proj_bn = nn.BatchNorm2d(512)
        self.roi_mlp = nn.Sequential(
            nn.LayerNorm(512),
            nn.Dropout(p=0.2),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
        )
        self.fusion = nn.Linear(in_feats + 256, in_feats)

        ref_pts = _reference_five_point_112(margin=0.1)
        leye, reye, mouth = roi_boxes_from_landmarks(ref_pts, out_size=112, eye_size=64, mouth_size=64)
        if leye is None or reye is None or mouth is None:
            def _center_box(cx, cy, bw, bh, W=112, H=112):
                x = int(round(cx - bw / 2))
                y = int(round(cy - bh / 2))
                x = max(0, min(W - bw, x))
                y = max(0, min(H - bh, y))
                return (x, y, bw, bh)

            leye = _center_box(112 * 0.35, 112 * 0.46, 64, 64)
            reye = _center_box(112 * 0.65, 112 * 0.46, 64, 64)
            mouth = _center_box(112 * 0.50, 112 * 0.82, 64, 64)
        self._roi_leye: torch.Tensor
        self._roi_reye: torch.Tensor
        self._roi_mouth: torch.Tensor
        self.register_buffer("_roi_leye", torch.tensor(leye, dtype=torch.float32))
        self.register_buffer("_roi_reye", torch.tensor(reye, dtype=torch.float32))
        self.register_buffer("_roi_mouth", torch.tensor(mouth, dtype=torch.float32))

    def _pool_roi(self, fmap: torch.Tensor, box_112: torch.Tensor) -> torch.Tensor:
        N, C, Hf, Wf = fmap.shape
        scale_x = Wf / 112.0
        scale_y = Hf / 112.0
        x, y, w, h = box_112
        x1 = torch.clamp(torch.round(x * scale_x).long(), 0, Wf - 1)
        y1 = torch.clamp(torch.round(y * scale_y).long(), 0, Hf - 1)
        x2 = torch.clamp(torch.round((x + w) * scale_x).long(), 0, Wf)
        y2 = torch.clamp(torch.round((y + h) * scale_y).long(), 0, Hf)
        x2 = torch.maximum(x2, x1 + 1)
        y2 = torch.maximum(y2, y1 + 1)
        x1_i, x2_i = int(x1.item()), int(x2.item())
        y1_i, y2_i = int(y1.item()), int(y2.item())
        roi = fmap[:, :, y1_i:y2_i, x1_i:x2_i]
        return roi.mean(dim=(2, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        has_time = x.dim() == 5
        if has_time:
            B, T, C, H, W = x.shape
            x = x.reshape(B * T, C, H, W)
        else:
            B, T = x.shape[0], 1
        feats_map = self.backbone.features(x)
        g = self.backbone.avgpool(feats_map)
        g = torch.flatten(g, 1)
        z = self.roi_proj_bn(self.roi_proj(feats_map))
        z = F.relu(z, inplace=True)
        leye = self._pool_roi(z, self._roi_leye)
        reye = self._pool_roi(z, self._roi_reye)
        mouth = self._pool_roi(z, self._roi_mouth)
        eyes_512 = 0.5 * (leye + reye)
        eyes_128 = self.roi_mlp(eyes_512)
        mouth_128 = self.roi_mlp(mouth)
        fused = torch.cat([g, eyes_128, mouth_128], dim=1)
        fused = self.fusion(fused)
        if T > 1:
            fused = fused.view(B, T, -1)
            if fused.size(1) > 8:
                fused = fused[:, -8:, :]
            fused = fused.mean(dim=1)
        return fused


class ArcFaceHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int, m: float = 0.5, s: float = 64.0, eps: float = 1e-7):
        super().__init__()
        self.W = nn.Parameter(torch.randn(num_classes, in_dim))
        nn.init.xavier_uniform_(self.W)
        self.m = float(m)
        self.s = float(s)
        self.eps = float(eps)
        self.m_vec: Optional[torch.Tensor] = None

    def set_margin_vector(self, m_vec: torch.Tensor) -> None:
        try:
            self.m_vec = m_vec.detach().to(dtype=torch.float32)
        except Exception:
            self.m_vec = None

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
        if isinstance(self.m_vec, torch.Tensor) and self.m_vec.numel() == cos.shape[1]:
            margin = self.m_vec.to(device=x.device, dtype=theta.dtype)[labels]
        else:
            margin = torch.full_like(theta, fill_value=float(self.m))
        cos_m = torch.cos(theta + margin)
        cos_out = cos.clone()
        cos_out[idx, labels] = cos_m
        return cos_out * self.s


def resolve_class_names(ckpt: dict, num_classes: int) -> List[str]:
    base = list(DEFAULT_DMD_CLASS_NAMES)
    class_map = ckpt.get("class_map")
    if isinstance(class_map, dict):
        mapped_names: List[Optional[str]] = [None] * num_classes
        filled = False
        for name, idx in class_map.items():
            try:
                idx_int = int(idx)
            except Exception:
                continue
            if 0 <= idx_int < num_classes:
                mapped_names[idx_int] = str(name)
                filled = True
        if filled:
            resolved: List[str] = []
            for idx, name in enumerate(mapped_names):
                if name is None:
                    if idx < len(base):
                        name = base[idx]
                    else:
                        name = f"class_{idx}"
                resolved.append(name)
            return resolved
    fallback_names: List[str] = []
    for idx in range(num_classes):
        if idx < len(base):
            fallback_names.append(base[idx])
        else:
            fallback_names.append(f"class_{idx}")
    return fallback_names


def build_transform(cfg: dict) -> Tuple[Callable[[Image.Image], torch.Tensor], int]:
    img_size = int(cfg.get("img_size", 112))
    mean = cfg.get("mean", [0.485, 0.456, 0.406])
    std = cfg.get("std", [0.229, 0.224, 0.225])
    pipeline = transforms.Compose(
        [
            transforms.Resize((img_size, img_size), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    return cast(Callable[[Image.Image], torch.Tensor], pipeline), img_size


def load_model(model_path: str, device: torch.device):
    ckpt = torch.load(model_path, map_location="cpu")
    num_classes = int(ckpt.get("num_classes", len(DEFAULT_DMD_CLASS_NAMES)))
    backbone = EfficientNetFeatureExtractor(variant=ckpt.get("eff_variant", "b0"), pretrained=False)
    backbone.load_state_dict(ckpt["backbone_state"], strict=True)
    head = ArcFaceHead(backbone.feature_dim, num_classes)
    head.load_state_dict(ckpt["head_state"], strict=True)
    backbone.to(device).eval()
    head.to(device).eval()
    preprocess_cfg = ckpt.get("preprocess", {})
    transform, img_size = build_transform(preprocess_cfg)
    class_names = resolve_class_names(ckpt, num_classes)
    return backbone, head, transform, class_names, img_size


def initialize_class_indices(class_names: Sequence[str]) -> None:
    global IDX_CLOSED, IDX_BLINK, IDX_YAWN_HAND, IDX_YAWN_NO
    try:
        IDX_CLOSED = class_names.index("eyes closed")
    except ValueError:
        IDX_CLOSED = None
    try:
        IDX_BLINK = class_names.index("blinking")
    except ValueError:
        IDX_BLINK = None
    try:
        IDX_YAWN_HAND = class_names.index("yawning with hand")
    except ValueError:
        IDX_YAWN_HAND = None
    try:
        IDX_YAWN_NO = class_names.index("yawning no hand")
    except ValueError:
        IDX_YAWN_NO = None


def apply_runner_up_bias(
    probs: torch.Tensor, class_names: Sequence[str], enabled: bool, boost_factor: float = 8.0
) -> torch.Tensor:
    if not enabled or probs.numel() == 0:
        return probs
    adjusted = probs.clone()
    try:
        idx_undef = class_names.index("eyes undefined")
    except ValueError:
        return probs
    sorted_idx = torch.argsort(probs, descending=True)
    runner_idx: Optional[int] = None
    for idx in sorted_idx.tolist():
        if idx == idx_undef:
            continue
        runner_idx = idx
        break
    if runner_idx is None:
        return probs
    adjusted[idx_undef] = 0.0
    adjusted[runner_idx] = adjusted[runner_idx] * float(boost_factor)
    total = float(adjusted.sum())
    if total <= 0:
        return probs
    adjusted = adjusted / total
    return adjusted


def treat_blink_as_closed(probs: torch.Tensor, class_names: Sequence[str], enabled: bool, threshold: float = 0.10) -> torch.Tensor:
    if not enabled or probs.numel() == 0:
        return probs
    try:
        idx_blink = class_names.index("blinking")
        idx_closed = class_names.index("eyes closed")
    except ValueError:
        return probs
    blink_val = probs[idx_blink].item()
    if blink_val >= threshold or torch.argmax(probs).item() == idx_blink:
        adjusted = probs.clone()
        adjusted[idx_closed] = max(adjusted[idx_closed], blink_val)
        adjusted[idx_blink] = 0.0
        total = adjusted.sum()
        if total > 0:
            adjusted = adjusted / total
        return adjusted
    return probs


def evaluate_fatigue(
    probs: torch.Tensor,
    treat_blink: bool,
) -> Tuple[bool, str]:
    global EYES_CLOSED_HISTORY, YAWN_SINCE
    if probs.numel() == 0:
        return (False, "")
    now = time.time()
    fatigue = False
    fatigue_reason = ""
    blink_overrides_closed = False
    top_idx = int(torch.argmax(probs).item())
    if treat_blink and IDX_BLINK is not None and probs[IDX_BLINK] >= 0.10:
        if IDX_CLOSED is not None:
            top_idx = IDX_CLOSED
        blink_overrides_closed = True
    if treat_blink and IDX_BLINK is not None and top_idx == IDX_BLINK:
        if IDX_CLOSED is not None:
            top_idx = IDX_CLOSED
        blink_overrides_closed = True
    is_closed = IDX_CLOSED is not None and top_idx == IDX_CLOSED
    EYES_CLOSED_HISTORY.append(bool(is_closed))
    closed_count = sum(1 for val in EYES_CLOSED_HISTORY if val)
    closed_majority = len(EYES_CLOSED_HISTORY) == EYES_CLOSED_HISTORY.maxlen and closed_count > len(EYES_CLOSED_HISTORY) / 2
    if closed_majority:
        fatigue = True
        fatigue_reason = "blinking" if blink_overrides_closed else "eyes closed"

    yawn_active = False
    if IDX_YAWN_HAND is not None and top_idx == IDX_YAWN_HAND:
        yawn_active = True
    if IDX_YAWN_NO is not None and top_idx == IDX_YAWN_NO:
        yawn_active = True
    if yawn_active:
        if YAWN_SINCE is None:
            YAWN_SINCE = now
        if now - (YAWN_SINCE or now) >= 0.5:
            fatigue = True
            fatigue_reason = "yawning"
    else:
        YAWN_SINCE = None
    return (fatigue, fatigue_reason)


app = FastAPI()

BACKBONE: Optional[nn.Module] = None
HEAD: Optional[nn.Module] = None
TRANSFORM_FN: Optional[Callable[[Image.Image], torch.Tensor]] = None
CLASS_NAMES: Optional[List[str]] = None
DEVICE: Optional[torch.device] = None
FACE_ALIGNER: Optional[FaceAligner] = None
INPUT_SIZE: int = 112
WARP_CANONICAL: bool = True
PREVIEW_ENABLED = False
WINDOW_NAME = "DSM Server Preview"
IDX_CLOSED: Optional[int] = None
IDX_BLINK: Optional[int] = None
IDX_YAWN_HAND: Optional[int] = None
IDX_YAWN_NO: Optional[int] = None
EYES_CLOSED_HISTORY: Deque[bool] = deque(maxlen=5)
YAWN_SINCE: Optional[float] = None


def load_image(image_bytes: bytes) -> Image.Image:
    if not image_bytes:
        raise ValueError("Empty image payload")
    try:
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise ValueError("Unable to decode image") from exc


def preprocess_image(image: Image.Image) -> torch.Tensor:
    if TRANSFORM_FN is None:
        raise RuntimeError("Transform pipeline not initialized")
    tensor = cast(torch.Tensor, TRANSFORM_FN(image)).unsqueeze(0)
    return tensor


def initialize_preview_window() -> None:
    global PREVIEW_ENABLED
    if not PREVIEW_ENABLED or cv2 is None:
        return
    try:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    except Exception as exc:
        logging.warning("Unable to open preview window: %s", exc)
        PREVIEW_ENABLED = False


def show_preview(
    image: Image.Image, fatigue: Optional[bool] = None, fatigue_reason: str = ""
) -> None:
    if not PREVIEW_ENABLED or cv2 is None:
        return
    frame = np.array(image)
    try:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    except Exception:
        pass
    if fatigue is not None:
        text = "fatigue" if fatigue else "not fatigue"
        if fatigue_reason:
            text = f"{text} ({fatigue_reason})"
        color = (0, 0, 255) if fatigue else (0, 255, 0)
        cv2.putText(
            frame,
            text,
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
            lineType=cv2.LINE_AA,
        )
    cv2.imshow(WINDOW_NAME, frame)
    cv2.waitKey(1)


def run_inference(
    tensor: torch.Tensor,
    promote_runner_up: bool,
    blink_as_closed: bool,
) -> Tuple[Dict[str, object], np.ndarray, bool, str]:
    if BACKBONE is None or HEAD is None or CLASS_NAMES is None or DEVICE is None:
        raise RuntimeError("Model not initialized")
    with torch.no_grad():
        feats = BACKBONE(tensor.to(DEVICE))
        logits = HEAD(feats)
        probabilities = torch.softmax(logits, dim=1)[0]
        probabilities = treat_blink_as_closed(probabilities, CLASS_NAMES, blink_as_closed)
        probabilities = apply_runner_up_bias(probabilities, CLASS_NAMES, promote_runner_up)
        fatigue, fatigue_reason = evaluate_fatigue(probabilities, blink_as_closed)
        probs_np = probabilities.cpu().numpy()

    results = []
    for idx, prob in enumerate(probs_np):
        class_name = CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else f"class_{idx}"
        results.append((class_name, float(prob)))
    results.sort(key=lambda item: item[1], reverse=True)

    top_class, top_prob = results[0]
    prob_dict = {name: prob for name, prob in results}
    logging.info("Inference result: top=%s (%.2f%%)", top_class, top_prob * 100.0)
    response = {
        "top_class": top_class,
        "top_prob": top_prob,
        "probs": prob_dict,
        "fatigue": {"active": fatigue, "reason": fatigue_reason},
    }
    return response, probs_np, fatigue, fatigue_reason


@app.post("/infer")
async def infer(image: UploadFile = File(None)):
    if image is None:
        raise HTTPException(status_code=400, detail="Image file is required")
    try:
        data = await image.read()
        pil_image = load_image(data)
        processed_image = pil_image
        if FACE_ALIGNER is not None:
            if cv2 is None:
                logging.warning("Alignment requested but OpenCV unavailable.")
            else:
                frame_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                aligned_image, pts_t = FACE_ALIGNER(frame_bgr)
                if WARP_CANONICAL:
                    aligned_image = warp_to_canonical_roi(aligned_image, pts_t, out_size=INPUT_SIZE)
                processed_image = aligned_image
        tensor = preprocess_image(processed_image)
        response, raw_probs, fatigue, fatigue_reason = run_inference(
            tensor, app.state.promote_runner_up, app.state.blink_as_closed
        )
        show_preview(processed_image, fatigue, fatigue_reason)
        return response
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        logging.exception("Inference not ready: %s", exc)
        raise HTTPException(status_code=500, detail="Inference pipeline unavailable") from exc
    except Exception:
        logging.exception("Unhandled error during inference")
        raise HTTPException(status_code=500, detail="Inference failed") from None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DSM FastAPI inference server")
    parser.add_argument(
        "--model-path",
        type=str,
        default=os.path.join("outputs", "last_finetune_arcface.pt"),
        help="Path to DSM .pt checkpoint",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind")
    parser.add_argument("--device", type=str, default="auto", help="Device for inference (auto|cpu|cuda|mps)")
    parser.add_argument(
        "--promote-runner-up",
        action="store_true",
        help="Match run_dsm behavior by boosting runner-up when 'eyes undefined' leads",
    )
    parser.add_argument(
        "--treat-blink-as-closed",
        action="store_true",
        help="When blinking dominates, treat it as eyes closed (mirrors run_dsm behavior)",
    )
    parser.add_argument("--yunet-path", type=str, default="face_detection_yunet.onnx", help="Path to YuNet ONNX model")
    parser.add_argument("--no-align", action="store_true", help="Disable YuNet alignment and use center crops")
    parser.add_argument("--disable-canonical-warp", action="store_true", help="Skip canonical affine warp")
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Open an OpenCV window that shows incoming frames on the server",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    args = parse_args()

    global BACKBONE, HEAD, TRANSFORM_FN, CLASS_NAMES, DEVICE, FACE_ALIGNER, INPUT_SIZE, WARP_CANONICAL, PREVIEW_ENABLED
    DEVICE = select_device(args.device)
    BACKBONE, HEAD, TRANSFORM_FN, CLASS_NAMES, INPUT_SIZE = load_model(args.model_path, DEVICE)

    if CLASS_NAMES is None or len(CLASS_NAMES) == 0:
        raise RuntimeError("Class names unavailable from checkpoint")
    initialize_class_indices(CLASS_NAMES)
    app.state.promote_runner_up = bool(args.promote_runner_up)
    app.state.blink_as_closed = bool(args.treat_blink_as_closed)
    WARP_CANONICAL = not args.disable_canonical_warp
    if not args.no_align:
        try:
            FACE_ALIGNER = FaceAligner(
                target_size=INPUT_SIZE,
                enable_alignment=True,
                yunet_path=args.yunet_path,
            )
        except Exception as exc:
            logging.warning("Alignment setup failed: %s", exc)
            FACE_ALIGNER = None
    else:
        FACE_ALIGNER = None
    if args.preview and cv2 is None:
        logging.warning("Preview requested but OpenCV is not available.")
    PREVIEW_ENABLED = bool(args.preview and cv2 is not None)
    if PREVIEW_ENABLED:
        initialize_preview_window()

    logging.info("Model loaded on %s", DEVICE)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
