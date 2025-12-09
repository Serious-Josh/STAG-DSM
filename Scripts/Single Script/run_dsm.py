# Single file script. Run on hardware with trained checkpoint and video source


import argparse
import os
import sys
import time
from typing import Any, Callable, List, Optional, Sequence, Tuple, cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image, ImageOps
from torchvision import transforms

try:
    import cv2  # type: ignore
except ImportError:
    cv2 = None  # type: ignore

try:
    PIL_RESAMPLE_BILINEAR = Image.Resampling.BILINEAR  # type: ignore[attr-defined]
except AttributeError:  # Pillow < 10
    PIL_RESAMPLE_BILINEAR = Image.BILINEAR  # type: ignore[attr-defined]


DEFAULT_DMD_CLASS_NAMES: List[str] = [
    "eyes open",
    "eyes closed",
    "eyes undefined",
    "blinking",
    "yawning with hand",
    "yawning no hand",
]

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


_CANONICAL_REF_POINTS = _reference_five_point_112(margin=0.1)
CANONICAL_LEFT_EYE = _CANONICAL_REF_POINTS[0].copy()
CANONICAL_RIGHT_EYE = _CANONICAL_REF_POINTS[1].copy()
CANONICAL_MOUTH = np.mean(_CANONICAL_REF_POINTS[3:5], axis=0).astype(np.float32)


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


def roi_boxes_from_landmarks(pts_t: Optional[np.ndarray], out_size: int = 112, eye_size: int = 64, mouth_size: int = 64):
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
        scale = float(out_size) / 112.0
        dst_pts = np.array(
            [
                CANONICAL_LEFT_EYE * scale,
                CANONICAL_RIGHT_EYE * scale,
                CANONICAL_MOUTH * scale,
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
            raise RuntimeError("OpenCV (cv2) is required for YuNetDetector.")
        face_detector_factory = getattr(cv2, "FaceDetectorYN_create", None)
        if face_detector_factory is None:
            raise RuntimeError("This OpenCV build is missing FaceDetectorYN. Install opencv-python>=4.6.")
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
        self.model_path = mp
        self.input_size = (int(input_size[0]), int(input_size[1]))
        self.conf_threshold = float(conf_threshold)
        self.nms_threshold = float(nms_threshold)
        self.top_k = int(top_k)
        self.detector = face_detector_factory(
            self.model_path, "", self.input_size, self.conf_threshold, self.nms_threshold, self.top_k
        )

    def set_input_size(self, w: int, h: int) -> None:
        if (w, h) != tuple(self.input_size):
            self.detector.setInputSize((int(w), int(h)))
            self.input_size = (int(w), int(h))

    def detect(self, img_bgr: np.ndarray):
        if img_bgr is None:
            return []
        h, w = img_bgr.shape[:2]
        if h <= 1 or w <= 1:
            return []

        # Make sure YuNet knows the current frame size
        self.set_input_size(w, h)

        try:
            res = self.detector.detect(img_bgr)
        except Exception:
            return []

        # Normalize to a faces ndarray (or None)
        faces = None

        if res is None:
            faces = None
        elif isinstance(res, tuple):
            # Empty tuple → no detections
            if len(res) == 0:
                faces = None
            # Single element → could be faces array or a count
            elif len(res) == 1:
                faces = res[0]
            else:
                # Prefer the first element that looks like an array
                for elem in res:
                    if hasattr(elem, "shape"):
                        faces = elem
                        break
                # Fallback if nothing looked like an array
                if faces is None:
                    faces = res[0]
        else:
            faces = res

        # No usable faces array
        if faces is None or not hasattr(faces, "shape") or faces.size == 0:
            return []

        results = []
        for f in np.asarray(faces):
            vals = f.flatten().tolist()
            if len(vals) < 15:
                continue
            x, y, bw, bh = (vals[0], vals[1], vals[2], vals[3])
            kps = [
                (vals[4], vals[5]),
                (vals[6], vals[7]),
                (vals[8], vals[9]),
                (vals[10], vals[11]),
                (vals[12], vals[13]),
            ]
            score = float(vals[14])
            results.append(
                {
                    "bbox": (float(x), float(y), float(bw), float(bh)),
                    "kps": kps,
                    "score": score,
                }
            )

        results.sort(key=lambda d: d.get("score", 0.0), reverse=True)
        return results





class FaceAligner:
    def __init__(self, target_size: int = 112, enable_alignment: bool = True, yunet_path: str = "face_detection_yunet.onnx"):
        self.target_size = int(target_size)
        self.enable_alignment = bool(enable_alignment)
        self.yunet_path = yunet_path
        self.detector: Optional[YuNetDetector] = None
        if self.enable_alignment:
            try:
                self.detector = YuNetDetector(model_path=yunet_path, input_size=(320, 320), conf_threshold=0.25, nms_threshold=0.3, top_k=5000)
                print("[FaceAligner] Using YuNet detector for alignment.")
            except Exception as exc:
                print(f"[FaceAligner] Warning: {exc}. Falling back to center crops.", file=sys.stderr)
                self.detector = None

    def __call__(self, frame_bgr: np.ndarray) -> Tuple[Image.Image, Optional[np.ndarray]]:
        if cv2 is not None:
            img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = frame_bgr[:, :, ::-1]
        pil = Image.fromarray(img_rgb)
        if self.detector is None:
            return (_center_fit(pil, self.target_size), None)
        try:
            dets = self.detector.detect(frame_bgr)
            if dets:
                pts = np.array(dets[0]["kps"], dtype=np.float32)
                aligned, pts_t = align_face_5pt(pil, pts, out_size=self.target_size, margin=0.1)
                return (aligned, pts_t)
            else:
                print("[FaceAligner] Face detector found no faces; falling back.", file=sys.stderr)
        except Exception as exc:
            print(f"[FaceAligner] Detection failed: {exc}", file=sys.stderr)
        return (_center_fit(pil, self.target_size), None)


TransformFn = Callable[[Image.Image], torch.Tensor]


class EfficientNetFeatureExtractor(nn.Module):
    def __init__(self, variant: str = "b0", pretrained: bool = False):
        super().__init__()
        variant = (variant or "b0").lower()
        if variant == "b0":
            weights = torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
            m = torchvision.models.efficientnet_b0(weights=weights)
        elif variant == "b1":
            weights = torchvision.models.EfficientNet_B1_Weights.IMAGENET1K_V1 if pretrained else None
            m = torchvision.models.efficientnet_b1(weights=weights)
        else:
            raise ValueError("Supported EfficientNet variants: b0, b1")
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
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"ArcFaceHead expected torch.Tensor input, got {type(x)}")
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


def build_transform(cfg: dict) -> Tuple[TransformFn, int]:
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
    return cast(TransformFn, pipeline), img_size


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


def apply_runner_up_bias(probs: torch.Tensor, class_names: Sequence[str], enabled: bool, boost_factor: float = 8.0) -> torch.Tensor:
    if not enabled or probs.numel() == 0:
        return probs
    adjusted = probs.clone()
    idx_undef = class_names.index("eyes undefined") if "eyes undefined" in class_names else None
    if idx_undef is None:
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


def format_prediction(probs: torch.Tensor, class_names: Sequence[str], topk: int, exclude: Optional[Sequence[str]] = None) -> str:
    probs = probs.clone()
    exclude_set = set(exclude or [])
    sorted_idx = torch.argsort(probs, descending=True)
    entries: List[str] = []
    for idx in sorted_idx.tolist():
        if len(entries) >= topk:
            break
        name = class_names[idx] if idx < len(class_names) else f"class_{idx}"
        if name in exclude_set:
            continue
        entries.append(f"{name}:{probs[idx].item() * 100:.1f}%")
    return " | ".join(entries)


def probability_rows(probs: torch.Tensor, class_names: Sequence[str]) -> List[str]:
    rows: List[str] = []
    for idx, name in enumerate(class_names):
        val = probs[idx].item() if idx < probs.numel() else 0.0
        rows.append(f"{name}:{val * 100:.1f}%")
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the DSM EfficientNet+ArcFace checkpoint on webcam frames.")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index passed to OpenCV VideoCapture (default: 0)")
    parser.add_argument("--camera-path", type=str, default=None, help="Direct camera device path (e.g., @device:pnp:...)")
    parser.add_argument(
        "--model-path",
        type=str,
        default=os.path.join("outputs", "last_finetune_arcface.pt"),
        help="Path to the .pt checkpoint to load.",
    )
    parser.add_argument("--device", type=str, default="auto", help="Device: auto|cpu|cuda|mps (default: auto)")
    parser.add_argument("--yunet-path", type=str, default="face_detection_yunet.onnx", help="Path to YuNet ONNX model.")
    parser.add_argument("--no-align", action="store_true", help="Disable YuNet alignment and use center crops only.")
    parser.add_argument("--disable-canonical-warp", action="store_true", help="Skip the canonical ROI warp.")
    parser.add_argument("--topk", type=int, default=3, help="How many class probabilities to display per frame.")
    parser.add_argument("--max-fps", type=float, default=0.0, help="Optional throttle on inference FPS (0 disables).")
    parser.add_argument("--debug-align", action="store_true", help="Show the aligned 112x112 crop for debugging.")
    parser.add_argument(
        "--promote-runner-up",
        action="store_true",
        help="Ignore 'eyes undefined' and strongly weight the next-best class.",
    )
    parser.add_argument(
        "--treat-blink-as-closed",
        action="store_true",
        help="Treat blinking as eyes closed; blink prob >10%% triggers eyes-closed state.",
    )
    return parser.parse_args()


def _backend_candidates() -> List[Tuple[Optional[int], str]]:
    if cv2 is None:
        return []
    if os.name == "nt":
        return [
            (None, "DEFAULT"),
            (cv2.CAP_DSHOW, "CAP_DSHOW"),
            (cv2.CAP_MSMF, "CAP_MSMF"),
            (cv2.CAP_ANY, "CAP_ANY"),
        ]


def open_camera(index: int, device_path: Optional[str] = None) -> Optional[Any]:
    """Try multiple OpenCV backends to open the requested camera index or explicit path."""
    if cv2 is None:
        return None
    candidates = _backend_candidates()
    targets = []
    if device_path:
        targets.append((device_path, "path"))
    else:
        targets.append((index, "index"))
    for target_value, target_desc in targets:
        for backend_flag, backend_name in candidates:
            try:
                if backend_flag is None:
                    cap = cv2.VideoCapture(target_value)
                else:
                    cap = cv2.VideoCapture(target_value, backend_flag)
            except Exception:
                continue
            if cap is not None and cap.isOpened():
                print(f"[run_dsm] Opened camera {target_desc} via backend {backend_name or 'DEFAULT'}.")
                return cap
            if cap is not None:
                cap.release()
    return None


def main() -> None:
    args = parse_args()
    if cv2 is None:
        print("OpenCV (cv2) is required for webcam streaming. Install it via 'pip install opencv-python'.", file=sys.stderr)
        sys.exit(1)
    device = select_device(args.device)
    if not os.path.isfile(args.model_path):
        print(f"Model path not found: {args.model_path}", file=sys.stderr)
        sys.exit(1)
    print(f"[run_dsm] Loading checkpoint from {args.model_path} on {device} ...")
    backbone, head, transform, class_names, input_size = load_model(args.model_path, device)
    aligner = FaceAligner(target_size=int(input_size), enable_alignment=not args.no_align, yunet_path=args.yunet_path)
    idx_closed = class_names.index("eyes closed") if "eyes closed" in class_names else None
    idx_closed = class_names.index("eyes closed") if "eyes closed" in class_names else None
    idx_blink = class_names.index("blinking") if "blinking" in class_names else None
    idx_yawn_hand = class_names.index("yawning with hand") if "yawning with hand" in class_names else None
    idx_yawn_no = class_names.index("yawning no hand") if "yawning no hand" in class_names else None
    eyes_closed_since: Optional[float] = None
    yawn_since: Optional[float] = None

    cap = open_camera(args.camera, device_path=args.camera_path)
    if cap is None or (not cap.isOpened()):
        print(f"Unable to open webcam index {args.camera}", file=sys.stderr)
        sys.exit(1)
    window_name = "DSM Webcam"
    show_preview = True
    try:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    except Exception as exc:
        show_preview = False
        print(f"[run_dsm] Warning: cannot open preview window ({exc}). Running without GUI.", file=sys.stderr)
    debug_window: Optional[str] = None
    if args.debug_align and show_preview:
        debug_window = "Aligned Crop"
        try:
            cv2.namedWindow(debug_window, cv2.WINDOW_NORMAL)
        except Exception as exc:
            print(f"[run_dsm] Warning: cannot open debug window ({exc}).", file=sys.stderr)
            debug_window = None
    print("Press Ctrl+C or hit 'q'/Esc in the preview window to stop. Output updates inline.")
    frame_interval = 1.0 / args.max_fps if args.max_fps and args.max_fps > 0 else 0.0
    last_infer_time = time.time()
    last_status_len = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue
            now = time.time()
            if frame_interval and (now - last_infer_time) < frame_interval:
                continue
            with torch.no_grad():
                face_img, pts_t = aligner(frame)
                if not args.disable_canonical_warp:
                    face_img = warp_to_canonical_roi(face_img, pts_t, out_size=input_size)
                if args.debug_align and debug_window:
                    cv2_mod = cast(Any, cv2)
                    face_bgr = cv2_mod.cvtColor(np.array(face_img), cv2.COLOR_RGB2BGR)
                    if pts_t is not None:
                        pts_int = np.round(pts_t).astype(int)
                        for (x, y) in pts_int:
                            cv2_mod.circle(face_bgr, (int(x), int(y)), 2, (0, 255, 255), -1)
                    h, w = face_bgr.shape[:2]
                    scale = w / float(input_size or 112)
                    def _draw_box(box, color):
                        if box is None:
                            return
                        x, y, bw, bh = box
                        x1 = int(round(x * scale))
                        y1 = int(round(y * scale))
                        x2 = int(round((x + bw) * scale))
                        y2 = int(round((y + bh) * scale))
                        cv2_mod.rectangle(face_bgr, (x1, y1), (x2, y2), color, 1)
                    _draw_box((CANONICAL_LEFT_EYE[0] - 32, CANONICAL_LEFT_EYE[1] - 32, 64, 64), (0, 0, 255))
                    _draw_box((CANONICAL_RIGHT_EYE[0] - 32, CANONICAL_RIGHT_EYE[1] - 32, 64, 64), (0, 255, 0))
                    _draw_box((CANONICAL_MOUTH[0] - 32, CANONICAL_MOUTH[1] - 32, 64, 64), (255, 0, 0))
                    cv2_mod.imshow(debug_window, face_bgr)
                tensor = transform(face_img).unsqueeze(0).to(device)
                feats = backbone(tensor)
                logits = head(feats)
                probs = torch.softmax(logits, dim=1)[0].cpu()
                probs = apply_runner_up_bias(probs, class_names, args.promote_runner_up)
            fps = 1.0 / max(1e-6, now - last_infer_time)
            last_infer_time = now
            exclude = ["eyes undefined"] if args.promote_runner_up else None
            pred_line = format_prediction(probs, class_names, args.topk, exclude=exclude)
            prob_rows = probability_rows(probs, class_names)
            top_idx = int(torch.argmax(probs).item()) if probs.numel() else -1
            fatigue = False
            fatigue_reason = ""
            blink_overrides_closed = False
            if args.treat_blink_as_closed and idx_blink is not None and probs[idx_blink] >= 0.10:
                top_idx = idx_closed if idx_closed is not None else top_idx
                blink_overrides_closed = True
            if args.treat_blink_as_closed and idx_blink is not None and top_idx == idx_blink:
                top_idx = idx_closed if idx_closed is not None else top_idx
                blink_overrides_closed = True
            if idx_closed is not None and top_idx == idx_closed:
                if eyes_closed_since is None:
                    eyes_closed_since = now
                trigger = 1.0
                if blink_overrides_closed:
                    trigger = 0.0
                if now - eyes_closed_since >= trigger:
                    fatigue = True
                    fatigue_reason = "eyes closed" if not blink_overrides_closed else "blinking"
            else:
                eyes_closed_since = None
            yawn_active = idx_yawn_hand is not None and top_idx == idx_yawn_hand
            yawn_active = yawn_active or (idx_yawn_no is not None and top_idx == idx_yawn_no)
            if yawn_active:
                if yawn_since is None:
                    yawn_since = now
                if now - yawn_since >= 0.5:
                    fatigue = True
                    fatigue_reason = "yawning"
            else:
                yawn_since = None
            fatigue_text = "fatigue" if fatigue else "not fatigue"
            status = f"{pred_line} | {fatigue_text}"
            pad = max(0, last_status_len - len(status))
            sys.stdout.write("\r" + status + (" " * pad))
            sys.stdout.flush()
            last_status_len = len(status)
            if show_preview:
                display_frame = frame.copy()
                base_y = 30
                for idx, row in enumerate(prob_rows):
                    color = (0, 255, 0) if idx == top_idx else (0, 180, 0)
                    cv2.putText(
                        display_frame,
                        row,
                        (10, base_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        color,
                        2,
                        lineType=cv2.LINE_AA,
                    )
                    base_y += 25
                base_y += 30
                fatigue_color = (0, 0, 255) if fatigue else (0, 255, 0)
                cv2.putText(
                    display_frame,
                    fatigue_text if not fatigue_reason else f"{fatigue_text} ({fatigue_reason})",
                    (10, base_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    fatigue_color,
                    2,
                    lineType=cv2.LINE_AA,
                )
                cv2.imshow(window_name, cast(Any, display_frame))
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    print("\n[run_dsm] Preview window closed by user.")
                    break
    except KeyboardInterrupt:
        print("\n[run_dsm] Stopped by user.")
    finally:
        cap.release()
        if show_preview:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
