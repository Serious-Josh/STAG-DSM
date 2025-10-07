import os
import sys
import math
import time
import json
import re
import random
import csv
import io
import argparse
from dataclasses import dataclass
from typing import Optional, List, Tuple

import numpy as np
import multiprocessing as mp
from collections import Counter, deque
import subprocess
from tqdm.auto import tqdm as _tqdm
from facenet_pytorch import MTCNN as _MTCNN
import cv2
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import WeightedRandomSampler, ConcatDataset, Subset, random_split

import torchvision
from torchvision import transforms
from torchvision.transforms import functional as TF
from PIL import Image, ImageOps, ImageFilter, ImageDraw
from typing import Any as _Any, cast

# Pillow resampling compatibility (Pylance-friendly)
try:
    from PIL.Image import Resampling as _PILResampling  # type: ignore[attr-defined]
    PIL_RESAMPLE_BILINEAR: _Any = _PILResampling.BILINEAR
except Exception:
    # Fallback for older Pillow; value 2 corresponds to bilinear
    PIL_RESAMPLE_BILINEAR: _Any = getattr(Image, 'BILINEAR', 2)  # type: ignore[attr-defined]

# Helper RGB colors with loose typing for Pylance
RGBColor = _Any
COL_20: RGBColor = (20, 20, 20)
COL_60: RGBColor = (60, 60, 60)
COL_100: RGBColor = (100, 100, 100)

# -----------------------------
# Utilities
# -----------------------------


def seed_everything(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


_DML_WARNED = False


def torch_device(prefer: Optional[str] = "auto") -> torch.device:
    """Pick the best available device.

    prefer: one of {"auto","cpu","cuda","mps","dml"}
      - auto: prefer CUDA, then MPS (Apple), else CPU
      - dml: fallback alias; uses CUDA/MPS/CPU unless torch-directml is installed
    """
    try:
        pref = (prefer or "auto").lower()
    except Exception:
        pref = "auto"

    # Helper to check MPS
    def _mps_available() -> bool:
        try:
            mps = getattr(torch.backends, "mps", None)
            return bool(mps) and bool(getattr(mps, "is_available", lambda: False)())
        except Exception:
            return False

    # DirectML replacement/fallback: not available without torch-directml
    def _warn_dml():
        global _DML_WARNED
        if not _DML_WARNED:
            try:
                print("DirectML backend not available; falling back to CUDA/MPS/CPU. "
                      "To enable DirectML, install a torch-directml build matching your torch version, "
                      "or choose --device cuda/cpu.")
            except Exception:
                pass
            _DML_WARNED = True

    if pref == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if pref == "mps" and _mps_available():
        return torch.device("mps")
    if pref == "dml":
        _warn_dml()
        if torch.cuda.is_available():
            return torch.device("cuda")
        if _mps_available():
            return torch.device("mps")
        return torch.device("cpu")

    if pref == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if _mps_available():
            return torch.device("mps")
    return torch.device("cpu")


def get_num_workers() -> int:
    try:
        return max(2, mp.cpu_count() // 2)
    except Exception:
        return 2


def progress(iterable, total: Optional[int] = None, unit: str = "it", desc: Optional[str] = None, leave: bool = False):
    """Return a tqdm-wrapped iterable if tqdm is available; otherwise the original iterable."""
    if _tqdm is not None:
        try:
            return _tqdm(iterable, total=total, unit=unit, desc=desc, leave=leave)
        except Exception:
            return iterable
    return iterable


def load_class_map(path: Optional[str]) -> Optional[dict]:
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# -----------------------------
# Dataset definitions
# -----------------------------


class CsvImageDataset(Dataset):
    """
    Generic CSV-backed image dataset.

    CSV format (no header expected by default):
      image_path,label

    - image_path: path to image, absolute or relative to root_dir
    - label: integer class id (0..num_classes-1) or string mapped via class_map

    This is flexible enough to cover AffectNet, RAF-DB, and DMD if you
    pre-export annotations into a uniform CSV.
    """

    def __init__(
        self,
        csv_file: str,
        root_dir: Optional[str] = None,
        transform=None,
        class_map: Optional[dict] = None,
        has_header: bool = False,
        sep: str = ",",
    ):
        super().__init__()
        self.root_dir = root_dir
        self.transform = transform
        self.samples: List[Tuple[str, int]] = []
        self.class_map = class_map
        with open(csv_file, "r", encoding="utf-8") as f:
            lines = f.read().strip().splitlines()
        if has_header and lines:
            lines = lines[1:]
        for line in lines:
            if not line:
                continue
            parts = line.split(sep)
            if len(parts) < 2:
                continue
            path_raw, label_raw = parts[0].strip(), parts[1].strip()
            if self.root_dir and not os.path.isabs(path_raw):
                path = os.path.join(self.root_dir, path_raw)
            else:
                path = path_raw
            if self.class_map and not label_raw.isdigit():
                if label_raw not in self.class_map:
                    raise ValueError(f"Label '{label_raw}' not in class_map")
                label = int(self.class_map[label_raw])
            else:
                label = int(label_raw)
            self.samples.append((path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        with Image.open(path) as im:
            im = im.convert("RGB")
            if self.transform is not None:
                im = self.transform(im)
        # torchvision transforms may return PIL or Tensor; ensure tensor
        if isinstance(im, Image.Image):
            im = TF.to_tensor(im)
        return im, label


# -----------------------------
# Cleaning and preprocessing for generic FER
# -----------------------------


def _try_import_mtcnn():
    return _MTCNN


def is_image_corrupted(path: str) -> bool:
    try:
        with Image.open(path) as im:
            im.verify()
        # Reopen to ensure load succeeds after verify
        with Image.open(path) as im:
            im.convert("RGB").load()
        return False
    except Exception:
        return True


def is_occluded_or_too_dark_bright(img: Image.Image, dark_thresh: float = 0.15, bright_thresh: float = 0.15) -> bool:
    im = np.array(img.convert("L"))  # grayscale
    total = im.size
    too_dark = (im < 10).sum() / total
    too_bright = (im > 245).sum() / total
    return (too_dark > dark_thresh) or (too_bright > bright_thresh)


def is_blurry(img: Image.Image, edge_thresh: float = 4.0) -> bool:
    # Estimate blur via Sobel gradient magnitude variance using torch
    t = TF.to_tensor(img.convert("L")).unsqueeze(0)  # [1,1,H,W]
    sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32, device=t.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32, device=t.device).view(1, 1, 3, 3)
    with torch.no_grad():
        gx = F.conv2d(t, sobel_x, padding=1)
        gy = F.conv2d(t, sobel_y, padding=1)
        mag = (gx.pow(2) + gy.pow(2)).sqrt()
        score = mag.var().item() * 1000.0
    return score < edge_thresh


def blur_metric(img: Image.Image) -> float:
    """Return the same Sobel variance score used in is_blurry, without thresholding."""
    t = TF.to_tensor(img.convert("L")).unsqueeze(0)
    sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32, device=t.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32, device=t.device).view(1, 1, 3, 3)
    with torch.no_grad():
        gx = F.conv2d(t, sobel_x, padding=1)
        gy = F.conv2d(t, sobel_y, padding=1)
        mag = (gx.pow(2) + gy.pow(2)).sqrt()
        score = mag.var().item() * 1000.0
    return float(score)


def laplacian_variance(img: Image.Image) -> float:
    """Variance of Laplacian (OpenCV), typical blur metric. Returns float."""
    if cv2 is None:
        # Fallback: approximate using Sobel variance if cv2 not available
        return blur_metric(img)
    gray = np.array(img.convert('L'), dtype=np.uint8)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(lap.var())


def compute_phash(img: Image.Image) -> int:
    """Compute 64-bit perceptual hash (pHash) using DCT low frequencies."""
    if cv2 is not None:
        try:
            gray = np.array(img.convert('L').resize((32, 32), PIL_RESAMPLE_BILINEAR), dtype=np.float32)
            dct = cv2.dct(gray)
        except Exception:
            dct = None
    else:
        dct = None
    if dct is None:
        # Numpy fallback DCT via FFT (approximate)
        a = np.array(img.convert('L').resize((32, 32), PIL_RESAMPLE_BILINEAR), dtype=np.float32)
        # simple 2D DCT approximation using cosine transform basis
        dct = np.fft.fft2(a)
        dct = np.real(dct)
    low = np.asarray(dct[:8, :8], dtype=np.float32)
    med = float(np.median(low[1:])) if low.size > 1 else float(np.median(low))
    bits = (low > med).astype(np.uint8).flatten()
    h = 0
    for b in bits:
        h = (h << 1) | int(b)
    return int(h)


def hamming_distance(a: int, b: int) -> int:
    return int((a ^ b).bit_count())


def skin_coverage_fraction(img: Image.Image) -> float:
    """Estimate skin pixel coverage fraction in an RGB face crop using simple color thresholds.
    Combines YCrCb and HSV heuristics; returns fraction in [0,1].
    """
    if cv2 is None:
        # Fallback: use luminance-based proxy
        g = np.array(img.convert('L'))
        # assume mid-tone pixels as potential skin
        mask = (g > 60) & (g < 200)
        return float(mask.mean()) if mask.size else 0.0
    bgr = cv2.cvtColor(np.array(img.convert('RGB')), cv2.COLOR_RGB2BGR)
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    cr = np.asarray(ycrcb[:, :, 1], dtype=np.uint8)
    cb = np.asarray(ycrcb[:, :, 2], dtype=np.uint8)
    mask1 = (cr > 135) & (cr < 180) & (cb > 85) & (cb < 135)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h = np.asarray(hsv[:, :, 0], dtype=np.uint8)
    s = np.asarray(hsv[:, :, 1], dtype=np.uint8)
    v = np.asarray(hsv[:, :, 2], dtype=np.uint8)
    mask2 = (h < 25) | (h > 230)
    mask2 &= (s > 30) & (v > 50)
    mask = (mask1 | mask2).astype(np.uint8)
    return float(mask.mean()) if mask.size else 0.0


def _reference_five_point_112(margin: float = 0.10) -> np.ndarray:
    """ArcFace-like reference landmarks for 112x112; apply margin by scaling around center.
    Returns array shape (5,2) float32 in pixel coords.
    """
    ref = np.array([
        [38.2946, 51.6963],  # left eye
        [73.5318, 51.5014],  # right eye
        [56.0252, 71.7366],  # nose
        [41.5493, 92.3655],  # left mouth
        [70.7299, 92.2041],  # right mouth
    ], dtype=np.float32)
    out = 112.0
    center = np.array([out / 2.0, out / 2.0], dtype=np.float32)
    scale = max(0.0, 1.0 - float(margin))
    ref_scaled = center + (ref - center) * scale
    return ref_scaled.astype(np.float32)


def align_face_5pt(img: Image.Image, pts5: np.ndarray, out_size: int = 112, margin: float = 0.10):
    """Similarity-align face using 5-point landmarks to 112x112 output.
    Returns (aligned_PIL, transformed_landmarks ndarray [5,2]). Falls back to center-fit on failure.
    """
    if cv2 is None:
        # Without OpenCV, return center-fit
        try:
            pil = ImageOps.fit(img, (out_size, out_size), method=PIL_RESAMPLE_BILINEAR, centering=(0.5, 0.5))
        except Exception:
            pil = img.resize((out_size, out_size))
        return pil, None

    pts_src = np.asarray(pts5, dtype=np.float32)
    if pts_src.shape != (5, 2):
        try:
            pil = ImageOps.fit(img, (out_size, out_size), method=PIL_RESAMPLE_BILINEAR, centering=(0.5, 0.5))
        except Exception:
            pil = img.resize((out_size, out_size))
        return pil, None

    ref = _reference_five_point_112(margin=margin)
    M, inliers = cv2.estimateAffinePartial2D(pts_src, ref, method=cv2.LMEDS)
    M = np.asarray(M, dtype=np.float32) if M is not None else None
    if M is None:
        try:
            pil = ImageOps.fit(img, (out_size, out_size), method=PIL_RESAMPLE_BILINEAR, centering=(0.5, 0.5))
        except Exception:
            pil = img.resize((out_size, out_size))
        return pil, None

    img_np = np.array(img.convert('RGB'))
    aligned_np = cv2.warpAffine(img_np, M, (out_size, out_size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    # Transform landmarks
    pts_aug = np.hstack([pts_src, np.ones((5, 1), dtype=np.float32)])
    pts_t = (M @ pts_aug.T).T[:, :2]
    return Image.fromarray(aligned_np), pts_t


def roi_boxes_from_landmarks(pts_t: Optional[np.ndarray], out_size: int = 112, eye_size: int = 64, mouth_size: int = 64):
    """Compute ROI boxes (x,y,w,h) for left_eye, right_eye, mouth in aligned image coords.
    Returns tuple of three boxes, each as (x, y, w, h). Boxes are clamped to image bounds.
    """
    def clamp_box(cx, cy, bw, bh, W, H):
        x = int(round(cx - bw / 2))
        y = int(round(cy - bh / 2))
        x = max(0, min(W - bw, x))
        y = max(0, min(H - bh, y))
        return (x, y, bw, bh)

    if pts_t is None or not isinstance(pts_t, np.ndarray) or pts_t.shape != (5, 2):
        return None, None, None
    W = H = out_size
    left_eye = pts_t[0]
    right_eye = pts_t[1]
    mouth_c = (pts_t[3] + pts_t[4]) / 2.0
    le = clamp_box(left_eye[0], left_eye[1], eye_size, eye_size, W, H)
    re = clamp_box(right_eye[0], right_eye[1], eye_size, eye_size, W, H)
    mo = clamp_box(mouth_c[0], mouth_c[1], mouth_size, mouth_size, W, H)
    return le, re, mo


class GenericFERDataset(CsvImageDataset):
    """Csv dataset with cleaning + optional face alignment/cropping.

    - Filters: corrupted files, invalid labels, heuristic occlusion, heavy blur
    - Alignment: if facenet_pytorch.MTCNN available, crops to detected face box
    - Preprocess: resize and normalize later via transform
    """

    def __init__(
        self,
        csv_file: str,
        root_dir: Optional[str] = None,
        transform=None,
        class_map: Optional[dict] = None,
        has_header: bool = False,
        sep: str = ",",
        num_classes: Optional[int] = None,
        enable_alignment: bool = True,
        crop_tight: bool = False,
        margin_range: Tuple[float, float] = (0.08, 0.12),
        target_size: Optional[int] = None,
    ):
        super().__init__(csv_file, root_dir, transform, class_map, has_header, sep)

        # Optional MTCNN alignment
        self.mtcnn = None
        if enable_alignment:
            MTCNN = _try_import_mtcnn()
            if MTCNN is not None:
                # facenet_pytorch supports CUDA or CPU; prefer CUDA when available
                mtcnn_device = 'cuda' if torch.cuda.is_available() else 'cpu'
                self.mtcnn = MTCNN(keep_all=False, device=mtcnn_device)

        # Tight-crop config (used for pretraining)
        self.crop_tight = bool(crop_tight)
        self.margin_range = margin_range
        self.target_size = target_size

        # Build cleaned index
        cleaned: List[Tuple[str, int]] = []
        dropped_corrupt = dropped_occluded = dropped_blurry = 0
        for path, label in self.samples:
            # Label validation if num_classes given
            if num_classes is not None and (label < 0 or label >= num_classes):
                continue
            if is_image_corrupted(path):
                dropped_corrupt += 1
                continue
            try:
                with Image.open(path) as im:
                    im = im.convert("RGB")
                    if is_occluded_or_too_dark_bright(im):
                        dropped_occluded += 1
                        continue
                    if is_blurry(im):
                        dropped_blurry += 1
                        continue
            except Exception:
                dropped_corrupt += 1
                continue
            cleaned.append((path, label))

        self.samples = cleaned
        if dropped_corrupt or dropped_occluded or dropped_blurry:
            print(
                f"[GenericFERDataset] Cleaned: {len(self.samples)} kept | "
                f"corrupt {dropped_corrupt}, occluded {dropped_occluded}, blurry {dropped_blurry}")

    def _align_crop(self, img: Image.Image) -> Image.Image:
        # If tight cropping requested, try to detect bbox and crop with margin
        if self.crop_tight and self.mtcnn is not None:
            try:
                res = self.mtcnn.detect(img, landmarks=False)
                boxes, probs = (None, None)
                if isinstance(res, tuple):
                    if len(res) >= 1:
                        boxes = res[0]
                    if len(res) >= 2:
                        probs = res[1]
                if boxes is not None and len(boxes) > 0:
                    # pick the highest-confidence box
                    idx = 0
                    if probs is not None:
                        idx = int(np.argmax(probs))
                    x1, y1, x2, y2 = boxes[idx]
                    w = max(1.0, float(x2 - x1))
                    h = max(1.0, float(y2 - y1))
                    m = float(random.uniform(self.margin_range[0], self.margin_range[1]))
                    dw = m * w
                    dh = m * h
                    nx1 = max(0.0, x1 - dw)
                    ny1 = max(0.0, y1 - dh)
                    nx2 = min(float(img.width), x2 + dw)
                    ny2 = min(float(img.height), y2 + dh)
                    cropped = img.crop((nx1, ny1, nx2, ny2))
                    # Fit to square target if requested
                    if isinstance(self.target_size, int) and self.target_size > 0:
                        try:
                            cropped = ImageOps.fit(cropped, (self.target_size, self.target_size), method=PIL_RESAMPLE_BILINEAR, centering=(0.5, 0.5))
                        except Exception:
                            cropped = cropped.resize((self.target_size, self.target_size))
                    return cropped
            except Exception:
                pass

        # Fallback: default alignment via MTCNN's forward (if available)
        if self.mtcnn is not None:
            try:
                face = self.mtcnn(img, save_path=None)
                if face is None:
                    return img
                if isinstance(face, list):
                    face = face[0]
                face = face.clamp(0, 1)
                pil = TF.to_pil_image(face)
                if isinstance(self.target_size, int) and self.target_size > 0:
                    pil = ImageOps.fit(pil, (self.target_size, self.target_size), method=PIL_RESAMPLE_BILINEAR, centering=(0.5, 0.5))
                return pil
            except Exception:
                return img
        return img

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        with Image.open(path) as im:
            im = im.convert("RGB")
            im = self._align_crop(im)
            if self.transform is not None:
                im = self.transform(im)
        if isinstance(im, Image.Image):
            im = TF.to_tensor(im)
        return im, label


class RandomTightCrop:
    def __init__(self, max_pixels: int = 8, output_size: Optional[int] = None):
        self.max_pixels = int(max(0, max_pixels))
        self.output_size = output_size

    def __call__(self, img: Image.Image) -> Image.Image:
        if self.max_pixels <= 0:
            return img
        w, h = img.size
        dx = random.randint(0, min(self.max_pixels, max(0, w // 10)))
        dy = random.randint(0, min(self.max_pixels, max(0, h // 10)))
        left = dx
        top = dy
        right = w - dx
        bottom = h - dy
        if right <= left or bottom <= top:
            return img
        img = img.crop((left, top, right, bottom))
        if isinstance(self.output_size, int) and self.output_size > 0:
            try:
                img = ImageOps.fit(img, (self.output_size, self.output_size), method=PIL_RESAMPLE_BILINEAR, centering=(0.5, 0.5))
            except Exception:
                img = img.resize((self.output_size, self.output_size))
        return img


class MotionBlur:
    def __init__(self, p: float = 0.2, k_range: Tuple[int, int] = (3, 7)):
        self.p = p
        self.k_range = k_range

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() >= self.p:
            return img
        kmin, kmax = self.k_range
        k = random.randint(max(3, kmin), max(3, kmax))
        if k % 2 == 0:
            k += 1
        # random horizontal or vertical motion
        orientation = random.choice(["h", "v"])
        size = (k, k)
        kernel = [0.0] * (k * k)
        if orientation == "h":
            row = k // 2
            for c in range(k):
                kernel[row * k + c] = 1.0 / k
        else:
            col = k // 2
            for r in range(k):
                kernel[r * k + col] = 1.0 / k
        try:
            return img.filter(ImageFilter.Kernel(size, kernel, scale=None))
        except Exception:
            return img


class RandomJPEG:
    def __init__(self, p: float = 0.2, quality: Tuple[int, int] = (40, 80)):
        self.p = p
        self.quality = quality

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() >= self.p:
            return img
        q = int(random.randint(self.quality[0], self.quality[1]))
        buf = io.BytesIO()
        try:
            img.save(buf, format="JPEG", quality=q)
            buf.seek(0)
            return Image.open(buf).convert("RGB")
        except Exception:
            return img


class RandomGamma:
    def __init__(self, p: float = 0.5, gamma_range: Tuple[float, float] = (0.9, 1.1)):
        self.p = p
        self.gamma_range = gamma_range

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() >= self.p:
            return img
        g = random.uniform(self.gamma_range[0], self.gamma_range[1])
        try:
            t = TF.to_tensor(img)
            t = TF.adjust_gamma(t, gamma=g)
            return TF.to_pil_image(t)
        except Exception:
            return img


class RandomOcclusionBand:
    def __init__(self, p: float = 0.1):
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() >= self.p:
            return img
        w, h = img.size
        draw = ImageDraw.Draw(img)
        kind = random.choice(["sunglasses", "mask"])
        if kind == "sunglasses":
            y_center = int(0.33 * h)
            band_h = int(random.uniform(0.15, 0.25) * h)
            y1 = max(0, y_center - band_h // 2)
            y2 = min(h, y1 + band_h)
            draw.rectangle([0, y1, w, y2], fill=(0, 0, 0))
        else:  # mask
            y1 = int(0.55 * h)
            y2 = min(h, int(y1 + random.uniform(0.25, 0.4) * h))
            draw.rectangle([0, y1, w, y2], fill=(0, 0, 0))
        return img


class AddGaussianNoise:
    def __init__(self, p: float = 0.1, sigma_range: Tuple[float, float] = (3.0, 8.0)):
        self.p = p
        self.sigma_range = sigma_range

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        if random.random() >= self.p:
            return t
        sigma = random.uniform(self.sigma_range[0], self.sigma_range[1]) / 255.0
        noise = torch.randn_like(t) * sigma
        t = t + noise
        return t.clamp(0.0, 1.0)


def make_transforms(img_size: int = 112):
    # Train (pretraining) augmentations with modest magnitudes
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size), antialias=True),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1), fill=0),
        RandomTightCrop(max_pixels=8, output_size=img_size),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        RandomGamma(p=0.5, gamma_range=(0.9, 1.1)),
        transforms.RandomGrayscale(p=0.1),
        MotionBlur(p=0.2, k_range=(3, 7)),
        RandomJPEG(p=0.2, quality=(40, 80)),
        RandomOcclusionBand(p=0.1),
        transforms.ToTensor(),
        AddGaussianNoise(p=0.1, sigma_range=(3.0, 8.0)),
        transforms.RandomErasing(p=0.25, scale=(0.08, 0.12), value=0),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_tf, val_tf


# -----------------------------
# CSV generation from folder-per-class
# -----------------------------


def _gen_csv_from_folders(root_dir: str, out_csv: str, exts: set, class_map_in: Optional[dict], save_map_path: Optional[str]) -> Optional[str]:
    # Discover class folders (direct children) and build mapping
    root_dir = os.path.abspath(root_dir)
    subdirs = [d for d in sorted(os.listdir(root_dir)) if os.path.isdir(os.path.join(root_dir, d))]
    if not subdirs:
        raise ValueError(f"No class subfolders found under: {root_dir}")

    if class_map_in is None:
        class_map = {name: idx for idx, name in enumerate(subdirs)}
        wrote_map_path = None
        if save_map_path:
            with open(save_map_path, 'w', encoding='utf-8') as f:
                json.dump(class_map, f, ensure_ascii=False, indent=2)
            wrote_map_path = save_map_path
    else:
        class_map = class_map_in
        wrote_map_path = None

    count = 0
    with open(out_csv, 'w', encoding='utf-8') as fcsv:
        for cname in subdirs:
            cdir = os.path.join(root_dir, cname)
            if cname not in class_map:
                # skip folders not in provided mapping
                continue
            label = class_map[cname]
            for r, _, files in os.walk(cdir):
                for fn in files:
                    ext = os.path.splitext(fn)[1].lower()
                    if ext in exts:
                        rel_path = os.path.relpath(os.path.join(r, fn), root_dir)
                        fcsv.write(f"{rel_path},{label}\n")
                        count += 1

    print(f"[CSV GEN] {count} samples written from {len(subdirs)} classes")
    return wrote_map_path


def _gen_csv_from_rafdb(root_dir: str, out_csv: str, split: str = "train", image_subdir: Optional[str] = None):
    """Generate CSV for RAF-DB given its standard structure.

    Expected layout (typical):
      ROOT_DIR/
        EmoLabel/list_train.txt or list_test.txt
        Image/aligned/ or Image/original/

    Each line in list_*.txt commonly contains: <image_name> <label>
      - labels are typically 1..7; we convert to 0..6
    """
    root_dir = os.path.abspath(root_dir)
    anno_dir = os.path.join(root_dir, "EmoLabel")
    if split == "train":
        list_file = os.path.join(anno_dir, "list_train.txt")
    else:
        list_file = os.path.join(anno_dir, "list_test.txt")
    if not os.path.isfile(list_file):
        raise FileNotFoundError(f"RAF-DB annotation file not found: {list_file}")

    # Decide image directory
    if image_subdir is None:
        candidates = ["Image/aligned", "Image/original", "image/aligned", "image/original", "Aligned", "Original"]
        img_root = None
        for c in candidates:
            p = os.path.join(root_dir, c)
            if os.path.isdir(p):
                img_root = p
                break
        if img_root is None:
            raise FileNotFoundError("Could not locate RAF-DB images. Provide --raf_image_subdir.")
    else:
        img_root = os.path.join(root_dir, image_subdir)
        if not os.path.isdir(img_root):
            raise FileNotFoundError(f"RAF-DB images directory not found: {img_root}")

    # Parse lines
    rows = []
    with open(list_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            img_name, label_str = parts[0], parts[1]
            try:
                label = int(label_str)
            except Exception:
                continue
            # Convert 1-based to 0-based if needed
            if label >= 1:
                label -= 1
            img_path = os.path.join(img_root, img_name)
            # Allow images in nested paths; if not exists, try adding extension fallbacks is out-of-scope
            rel_path = os.path.relpath(img_path, img_root)
            rows.append((rel_path.replace('\\', '/'), label))

    if not rows:
        raise ValueError("No RAF-DB samples parsed. Check paths and annotation format.")

    with open(out_csv, 'w', encoding='utf-8') as fcsv:
        for p, l in rows:
            fcsv.write(f"{p},{l}\n")

    print(f"[RAF-DB CSV GEN] {len(rows)} samples from split='{split}' using base '{os.path.relpath(img_root, root_dir)}'")


def build_weighted_sampler(labels: List[int]) -> WeightedRandomSampler:
    counts = Counter(labels)
    num_samples = len(labels)
    # Inverse frequency weights
    weights = [num_samples / (counts[l] * len(counts)) for l in labels]
    return WeightedRandomSampler(weights, num_samples=num_samples, replacement=True)


# -----------------------------
# Feature extractor (backbone)
# -----------------------------


class EfficientNetFeatureExtractor(nn.Module):
    """EfficientNet backbone that outputs feature vectors for images.

    - Uses torchvision EfficientNet variants and removes the classifier head
      so forward() returns pooled features suitable for downstream tasks.
    """

    def __init__(self, variant: str = "b0", pretrained: bool = True):
        super().__init__()
        variant = (variant or "b0").lower()
        if variant == "b0":
            m = torchvision.models.efficientnet_b0(weights=(
                torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
            ))
        elif variant == "b1":
            m = torchvision.models.efficientnet_b1(weights=(
                torchvision.models.EfficientNet_B1_Weights.IMAGENET1K_V1 if pretrained else None
            ))
        else:
            raise ValueError("Supported variants: b0, b1")
        # Remove classifier entirely to avoid dropout; return pooled features
        in_feats = m.classifier[1].in_features
        m.classifier = nn.Sequential(nn.Identity())
        self.backbone = m
        self.feature_dim = in_feats

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


# -----------------------------
# Data loading config
# -----------------------------


@dataclass
class TrainConfig:
    # Data
    affectnet_csv: Optional[str] = None
    affectnet_root: Optional[str] = None
    rafdb_csv: Optional[str] = None
    rafdb_root: Optional[str] = None
    dmd_csv: Optional[str] = None
    dmd_root: Optional[str] = None
    class_map_json: Optional[str] = None

    # Data/Preprocessing
    num_classes: int = 7
    eff_variant: str = "b0"  # unused now; kept for CLI compatibility
    img_size: int = 112

    # Loader/Batching
    batch_size: int = 64
    lr: float = 3e-4  # unused
    weight_decay: float = 1e-4  # unused
    epochs_pretrain: int = 5  # unused
    epochs_main: int = 10  # unused

    # Misc
    seed: int = 42
    out_dir: str = "./outputs"
    amp: bool = True  # unused
    balance_sampling: bool = True


def build_loaders(cfg: TrainConfig, device: Optional[torch.device] = None):
    train_tf, val_tf = make_transforms(cfg.img_size)

    class_map = load_class_map(cfg.class_map_json)

    # Pretraining: combine AffectNet + RAF-DB
    pretrain_datasets: List[Dataset] = []
    if cfg.affectnet_csv:
        pretrain_datasets.append(GenericFERDataset(
            cfg.affectnet_csv, cfg.affectnet_root, transform=train_tf, class_map=class_map,
            num_classes=cfg.num_classes, enable_alignment=True,
            crop_tight=True, margin_range=(0.08, 0.12), target_size=cfg.img_size
        ))
    if cfg.rafdb_csv:
        pretrain_datasets.append(GenericFERDataset(
            cfg.rafdb_csv, cfg.rafdb_root, transform=train_tf, class_map=class_map,
            num_classes=cfg.num_classes, enable_alignment=True,
            crop_tight=True, margin_range=(0.08, 0.12), target_size=cfg.img_size
        ))

    if len(pretrain_datasets) == 0:
        raise ValueError("No pretraining datasets provided. Provide AffectNet and/or RAF-DB CSVs.")

    pretrain_train = ConcatDataset(pretrain_datasets)

    # For validation during pretrain, just reuse a small split from the concat
    # In practice, provide explicit val CSVs. Here we do a simple split.
    total = len(pretrain_train)
    val_size = max(1000, int(0.02 * total))
    train_size = total - val_size
    pretrain_train_ds, pretrain_val_ds = random_split(
        pretrain_train, [train_size, val_size], generator=torch.Generator().manual_seed(123)
    )

    # Replace transforms for val subset if present
    # random_split wraps underlying dataset; ensure val uses val_tf
    def wrap_with_val_tf(ds):
        base = ds.dataset if hasattr(ds, 'dataset') else ds
        # ConcatDataset or CsvImageDataset
        if isinstance(base, ConcatDataset):
            for d in base.datasets:
                if hasattr(d, 'transform'):
                    cast(_Any, d).transform = val_tf
        elif hasattr(base, 'transform'):
            cast(_Any, base).transform = val_tf

    wrap_with_val_tf(pretrain_val_ds)

    # DMD dataset for main training
    dmd_train_loader = None
    dmd_val_loader = None
    dmd_train_ds = None
    dmd_val_ds = None
    if cfg.dmd_csv:
        dmd_train_full = CsvImageDataset(cfg.dmd_csv, cfg.dmd_root, transform=train_tf, class_map=class_map)
        # Simple split for val if user didn't provide one; here just split 90/10
        total_dmd = len(dmd_train_full)
        val_dmd = max(200, int(0.1 * total_dmd))
        train_dmd = total_dmd - val_dmd
        dmd_train_ds, dmd_val_ds = random_split(
            dmd_train_full, [train_dmd, val_dmd], generator=torch.Generator().manual_seed(456)
        )
        wrap_with_val_tf(dmd_val_ds)

    def labels_from_dataset(ds: _Any) -> List[int]:
        lbls = []
        if isinstance(ds, Subset):
            for i in ds.indices:
                # underlying ConcatDataset or CsvImageDataset
                base = ds.dataset
                if isinstance(base, ConcatDataset):
                    # Need to locate which subdataset an index falls into
                    idx = i
                    for d in base.datasets:
                        n = len(cast(_Any, d))
                        if idx < n:
                            _, lab = d[idx]
                            lbls.append(lab)
                            break
                        idx -= n
                else:
                    _, lab = base[i]
                    lbls.append(lab)
        else:
            try:
                n = len(ds)  # type: ignore[arg-type]
            except Exception:
                return lbls
            for i in range(n):
                try:
                    _, lab = ds[i]
                    lbls.append(lab)
                except Exception:
                    break
        return lbls

    # Samplers
    pretrain_sampler = None
    dmd_sampler = None
    if cfg.balance_sampling:
        try:
            pretrain_labels = labels_from_dataset(pretrain_train_ds)
            pretrain_sampler = build_weighted_sampler(pretrain_labels)
        except Exception:
            pretrain_sampler = None
        try:
            dmd_labels = labels_from_dataset(dmd_train_ds)
            dmd_sampler = build_weighted_sampler(dmd_labels)
        except Exception:
            dmd_sampler = None

    # Loaders
    num_workers = get_num_workers()
    # Pin memory only helps with CUDA host->device transfers
    try:
        dev = device if device is not None else torch_device()
        pin_mem = isinstance(dev, torch.device) and dev.type == 'cuda'
    except Exception:
        pin_mem = False
    pretrain_train_loader = DataLoader(
        cast(Dataset, pretrain_train_ds), batch_size=cfg.batch_size, shuffle=(pretrain_sampler is None),
        sampler=pretrain_sampler, num_workers=num_workers, pin_memory=pin_mem
    )
    pretrain_val_loader = DataLoader(
        cast(Dataset, pretrain_val_ds), batch_size=cfg.batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_mem
    )
    if cfg.dmd_csv:
        dmd_train_loader = DataLoader(
            cast(Dataset, dmd_train_ds), batch_size=cfg.batch_size, shuffle=(dmd_sampler is None),
            sampler=dmd_sampler, num_workers=num_workers, pin_memory=pin_mem
        )
        dmd_val_loader = DataLoader(
            cast(Dataset, dmd_val_ds), batch_size=cfg.batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_mem
        )

    return pretrain_train_loader, pretrain_val_loader, dmd_train_loader, dmd_val_loader


def build_generic_eval_loaders(cfg: TrainConfig, args) -> List[Tuple[str, DataLoader]]:
    evals: List[Tuple[str, DataLoader]] = []
    _, val_tf = make_transforms(cfg.img_size)
    class_map = load_class_map(cfg.class_map_json)
    num_workers = get_num_workers()

    if getattr(args, 'affectnet_eval_csv', None):
        ds = GenericFERDataset(args.affectnet_eval_csv, getattr(args, 'affectnet_eval_root', cfg.affectnet_root), transform=val_tf,
                               class_map=class_map, num_classes=cfg.num_classes, enable_alignment=True)
        evals.append(("affectnet_eval", DataLoader(ds, batch_size=cfg.batch_size, shuffle=False,
                                                   num_workers=num_workers, pin_memory=True)))

    if getattr(args, 'rafdb_eval_csv', None):
        # For RAF-DB, eval may come from a different root than training one
        ds = GenericFERDataset(args.rafdb_eval_csv, getattr(args, 'rafdb_eval_root', cfg.rafdb_root), transform=val_tf,
                               class_map=class_map, num_classes=cfg.num_classes, enable_alignment=True)
        evals.append(("rafdb_eval", DataLoader(ds, batch_size=cfg.batch_size, shuffle=False,
                                               num_workers=num_workers, pin_memory=True)))

    return evals


def evaluate(*args, **kwargs):
    raise RuntimeError("Training/evaluation code removed. This script now focuses on data loading and preprocessing.")


def train_one_phase(*args, **kwargs):
    raise RuntimeError("Training/evaluation code removed. This script now focuses on data loading and preprocessing.")


def main():

    parser = argparse.ArgumentParser(description="FER data pipeline: CSV generation, loading, preprocessing, and feature extractor utilities")
    # Device control
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps", "dml"],
                        help=("Device: auto, cpu, cuda, mps (Apple), or dml. "
                              "Note: dml falls back to CUDA/MPS/CPU unless torch-directml is installed."))
    # CSV generation utility (folder-per-class)
    parser.add_argument("--gen_csv_from_folders", nargs=2, metavar=("ROOT_DIR", "OUT_CSV"),
                        help="Scan ROOT_DIR expecting folder-per-class and write OUT_CSV with image_path,label, then exit")
    parser.add_argument("--extensions", type=str, default=".jpg,.jpeg,.png,.bmp",
                        help="Comma-separated extensions to include for CSV generation")
    parser.add_argument("--save_class_map", type=str, default=None,
                        help="Optional path to save inferred class_map JSON when generating CSV")
    parser.add_argument("--use_class_map", type=str, default=None,
                        help="Optional existing class_map JSON to map folder names to ids during CSV generation")
    # Aligned crop export
    parser.add_argument("--export_aligned_crops", type=str, default=None,
                        help="Export aligned/tightly-cropped 112x112 face images and a CSV under this output root")
    parser.add_argument("--export_csv_name", type=str, default="aligned_metadata.csv",
                        help="Filename for the metadata CSV (relative to --export_aligned_crops)")
    # Feature extractor utilities
    parser.add_argument("--export_feature_extractor", type=str, default=None,
                        help="Export a backbone feature extractor checkpoint to the given path (.pt)")
    parser.add_argument("--extract_features_csv", type=str, default=None,
                        help="CSV to extract features from (image_path,label)")
    parser.add_argument("--extract_features_root", type=str, default=None,
                        help="Root dir for images referenced by --extract_features_csv")
    parser.add_argument("--extract_features_out", type=str, default=None,
                        help="Path to save extracted features as .npz (features, labels, paths)")
    parser.add_argument("--load_backbone_ckpt", type=str, default=None,
                        help="Optional checkpoint with a saved backbone state_dict to load before export/extraction")
    # DMD video export (face-only)
    parser.add_argument("--dmd_export_faces", type=str, default=None,
                        help="Export face frames from DMD videos into images and CSV under this output root")
    parser.add_argument("--dmd_json", type=str, default=None,
                        help="Path to DMD annotations JSON (optional; improves labels/splits if available)")
    parser.add_argument("--dmd_modality", type=str, default="rgb", choices=["rgb", "ir", "depth", "auto"],
                        help="Which DMD modality to extract from; if auto, prefers rgb then ir then depth")
    parser.add_argument("--dmd_fps", type=float, default=2.0, help="Target frames-per-second to sample from videos")
    parser.add_argument("--dmd_max_frames", type=int, default=300, help="Max frames to extract per video")
    parser.add_argument("--dmd_color", type=str, default="auto", choices=["auto", "rgb", "gray"],
                        help="Color handling for DMD frames: auto=rgb for rgb modality, gray for ir/depth")
    parser.add_argument("--dmd_camera_regex", type=str, default=None,
                        help="Optional regex with a capturing group to extract camera ID from filename (e.g., '(cam\\d+)'; use raw string r'(cam\\d+)')")
    parser.add_argument("--dmd_exclude_boundary_s", type=float, default=0.5,
                        help="Exclude first/last N seconds of each clip when saving frames (reduces boundary noise)")
    # Smoothing utilities (evaluation-only)
    parser.add_argument("--smooth_frames_csv", type=str, default=None,
                        help="Path to a frames CSV (e.g., dmd_frames.csv) to generate a smoothed-labels CSV for evaluation")
    parser.add_argument("--smooth_out", type=str, default=None, help="Output CSV path for smoothed labels")
    parser.add_argument("--smooth_method", type=str, default="majority", choices=["majority", "ema"],
                        help="Smoothing method over time: majority vote in a window, or EMA over time")
    parser.add_argument("--smooth_window_s", type=float, default=0.5, help="Temporal window in seconds for smoothing")
    parser.add_argument("--smooth_alpha", type=float, default=0.2, help="EMA alpha if method=ema (0<alpha<=1)")
    parser.add_argument("--smooth_fps", type=float, default=None, help="Fallback FPS if CSV lacks per-frame FPS/time")
    # CSV generation utility (RAF-DB official layout)
    parser.add_argument("--gen_csv_from_rafdb", nargs=2, metavar=("ROOT_DIR", "OUT_CSV"),
                        help="Parse RAF-DB annotation lists under ROOT_DIR and write OUT_CSV, then exit")
    parser.add_argument("--raf_split", type=str, default="train", choices=["train", "test"],
                        help="RAF-DB split to parse (train or test)")
    parser.add_argument("--raf_image_subdir", type=str, default=None,
                        help="Relative path under ROOT_DIR to images (e.g., 'Image/aligned'). If omitted, tries common defaults.")
    # Data args
    parser.add_argument("--affectnet_csv", type=str, default=None, help="CSV with image_path,label for AffectNet train")
    parser.add_argument("--affectnet_root", type=str, default=None, help="Root folder for AffectNet images")
    parser.add_argument("--rafdb_csv", type=str, default=None, help="CSV with image_path,label for RAF-DB train")
    parser.add_argument("--rafdb_root", type=str, default=None, help="Root folder for RAF-DB images")
    parser.add_argument("--dmd_csv", type=str, required=False, default=None, help="CSV with image_path,label for DMD train")
    parser.add_argument("--dmd_root", type=str, default=None, help="Root folder for DMD images")
    parser.add_argument("--class_map_json", type=str, default=None, help="Optional JSON mapping {label_name: id}")

    # Data/loader controls
    parser.add_argument("--num_classes", type=int, default=7)
    parser.add_argument("--eff_variant", type=str, default="b0", choices=["b0", "b1"]) 
    parser.add_argument("--img_size", type=int, default=112)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--epochs_pretrain", type=int, default=5)
    parser.add_argument("--epochs_main", type=int, default=10)
    parser.add_argument("--out_dir", type=str, default="./outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_amp", action="store_true", help="Disable mixed precision")
    parser.add_argument("--no_balance", action="store_true", help="Disable weighted sampling")
    # Inspection controls
    parser.add_argument("--inspect_batches", type=int, default=1, help="Number of batches to iterate for sanity checks per loader")
    # Back-compat: accept but ignore legacy training flags
    parser.add_argument("--pretrain_only", action="store_true", help="Deprecated no-op: training removed; flag is accepted for compatibility")
    # Generic dataset evaluation inputs (optional)
    parser.add_argument("--affectnet_eval_csv", type=str, default=None, help="Optional AffectNet eval CSV for reporting post-pretrain")
    parser.add_argument("--affectnet_eval_root", type=str, default=None, help="Optional AffectNet eval images root")
    parser.add_argument("--rafdb_eval_csv", type=str, default=None, help="Optional RAF-DB eval CSV for reporting post-pretrain")
    parser.add_argument("--rafdb_eval_root", type=str, default=None, help="Optional RAF-DB eval images root")

    args = parser.parse_args()

    # Autodiscovery of common local defaults if flags omitted
    def _maybe(path: str) -> Optional[str]:
        return path if path and os.path.exists(path) else None

    autodiscovered = []
    if not args.affectnet_csv:
        p = _maybe(os.path.join(os.getcwd(), "affectnet_train.csv"))
        if p:
            args.affectnet_csv = p
            autodiscovered.append(f"affectnet_csv={p}")
    if not args.affectnet_root:
        # Prefer Train subfolder root since CSV was generated from Train
        p = _maybe(os.path.join(os.getcwd(), "datasets", "affectNet", "Train"))
        if not p:
            p = _maybe(os.path.join(os.getcwd(), "datasets", "affectNet"))
        if p:
            args.affectnet_root = p
            autodiscovered.append(f"affectnet_root={p}")

    if not args.rafdb_csv:
        p = _maybe(os.path.join(os.getcwd(), "rafdb_train.csv"))
        if p:
            args.rafdb_csv = p
            autodiscovered.append(f"rafdb_csv={p}")
    if not args.rafdb_root:
        # Support both layouts:
        #   datasets/raf-db/train
        #   datasets/raf-db/DATASET/train
        raf_candidates = [
            os.path.join(os.getcwd(), "datasets", "raf-db", "train"),
            os.path.join(os.getcwd(), "datasets", "raf-db", "DATASET", "train"),
        ]
        for cand in raf_candidates:
            p = _maybe(cand)
            if p:
                args.rafdb_root = p
                autodiscovered.append(f"rafdb_root={p}")
                break

    if not args.class_map_json:
        # Prefer an affectnet_map.json in CWD if present
        p = _maybe(os.path.join(os.getcwd(), "affectnet_map.json"))
        if p:
            args.class_map_json = p
            autodiscovered.append(f"class_map_json={p}")

    if autodiscovered:
        print("Autodiscovered defaults:")
        for item in autodiscovered:
            print("  ", item)

    # Autodiscover eval CSVs if present
    eval_autodiscovered = []
    if not args.affectnet_eval_csv:
        p = _maybe(os.path.join(os.getcwd(), "affectnet_test.csv"))
        if p:
            args.affectnet_eval_csv = p
            eval_autodiscovered.append(f"affectnet_eval_csv={p}")
    if not args.rafdb_eval_csv:
        p = _maybe(os.path.join(os.getcwd(), "rafdb_test.csv"))
        if p:
            args.rafdb_eval_csv = p
            eval_autodiscovered.append(f"rafdb_eval_csv={p}")
    # Try to autodiscover eval roots for AffectNet/RAF-DB if csvs found
    if not args.affectnet_eval_root and args.affectnet_eval_csv:
        p = _maybe(os.path.join(os.getcwd(), "datasets", "affectNet", "Test"))
        if p:
            args.affectnet_eval_root = p
            eval_autodiscovered.append(f"affectnet_eval_root={p}")
    if not args.rafdb_eval_root and args.rafdb_eval_csv:
        # Support both layouts for eval/test
        raf_candidates = [
            os.path.join(os.getcwd(), "datasets", "raf-db", "test"),
            os.path.join(os.getcwd(), "datasets", "raf-db", "DATASET", "test"),
        ]
        for cand in raf_candidates:
            p = _maybe(cand)
            if p:
                args.rafdb_eval_root = p
                eval_autodiscovered.append(f"rafdb_eval_root={p}")
                break

    if eval_autodiscovered:
        print("Autodiscovered eval sets:")
        for item in eval_autodiscovered:
            print("  ", item)

    # ---------------- CSV generation mode ----------------
    if args.gen_csv_from_folders is not None:
        root_dir, out_csv = args.gen_csv_from_folders
        exts = {e.lower().strip() for e in args.extensions.split(',') if e.strip()}
        class_map = None
        if args.use_class_map:
            class_map = load_class_map(args.use_class_map)
        wrote_map_path = _gen_csv_from_folders(root_dir, out_csv, exts, class_map, args.save_class_map)
        print(f"Wrote CSV: {out_csv}")
        if wrote_map_path:
            print(f"Wrote class_map: {wrote_map_path}")
        return

    if args.gen_csv_from_rafdb is not None:
        root_dir, out_csv = args.gen_csv_from_rafdb
        img_subdir = args.raf_image_subdir
        _gen_csv_from_rafdb(root_dir, out_csv, split=args.raf_split, image_subdir=img_subdir)
        print(f"Wrote RAF-DB CSV: {out_csv}")
        return

    # ---------------- Export aligned crops mode ----------------
    if args.export_aligned_crops is not None:
        seed_everything(args.seed)
        device = torch_device(getattr(args, "device", "auto"))
        out_root = os.path.abspath(args.export_aligned_crops)
        os.makedirs(out_root, exist_ok=True)

        # Build datasets like pretraining to get the same samples/splits
        cfg = TrainConfig(
            affectnet_csv=args.affectnet_csv,
            affectnet_root=args.affectnet_root,
            rafdb_csv=args.rafdb_csv,
            rafdb_root=args.rafdb_root,
            dmd_csv=args.dmd_csv,
            dmd_root=args.dmd_root,
            class_map_json=args.class_map_json,
            num_classes=args.num_classes,
            eff_variant=args.eff_variant,
            img_size=args.img_size,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            epochs_pretrain=args.epochs_pretrain,
            epochs_main=args.epochs_main,
            seed=args.seed,
            out_dir=args.out_dir,
            amp=(not args.no_amp),
            balance_sampling=(not args.no_balance),
        )

        # We'll iterate raw samples rather than augmented tensors; set up a detector
        MTCNN = _try_import_mtcnn()
        mtcnn = None
        if MTCNN is not None:
            mtcnn = MTCNN(keep_all=False, device=('cuda' if torch.cuda.is_available() else 'cpu'))
        else:
            print("Warning: facenet_pytorch not available; exporting center-cropped resized images without landmarks.")

        def detect_and_crop(pil: Image.Image) -> Tuple[Image.Image, Optional[float], Optional[Tuple[int,int,int,int]], Optional[Tuple[int,int,int,int]], Optional[Tuple[int,int,int,int]]]:
            # Try tight crop from detection with margin 8-12%, then fit to cfg.img_size
            if mtcnn is not None:
                try:
                    res = mtcnn.detect(pil, landmarks=True)
                    boxes = probs = points = None
                    if isinstance(res, tuple):
                        if len(res) >= 1:
                            boxes = res[0]
                        if len(res) >= 2:
                            probs = res[1]
                        if len(res) >= 3:
                            points = res[2]
                    if boxes is not None and len(boxes) > 0:
                        idx = 0
                        if probs is not None:
                            idx = int(np.argmax(probs))
                        prob = float(probs[idx]) if probs is not None else None
                        pts5 = points[idx] if points is not None else None
                        if pts5 is not None:
                            aligned, pts_t = align_face_5pt(pil, np.array(pts5, dtype=np.float32), out_size=cfg.img_size, margin=0.10)
                            le, reye, mo = roi_boxes_from_landmarks(pts_t, out_size=cfg.img_size, eye_size=64, mouth_size=64)
                            return aligned, prob, le, reye, mo
                        # Fallback to bbox-based crop
                        x1, y1, x2, y2 = boxes[idx]
                        w = max(1.0, float(x2 - x1))
                        h = max(1.0, float(y2 - y1))
                        m = 0.10
                        dw = m * w
                        dh = m * h
                        nx1 = max(0.0, x1 - dw)
                        ny1 = max(0.0, y1 - dh)
                        nx2 = min(float(pil.width), x2 + dw)
                        ny2 = min(float(pil.height), y2 + dh)
                        cropped = pil.crop((nx1, ny1, nx2, ny2))
                        try:
                            cropped = ImageOps.fit(cropped, (cfg.img_size, cfg.img_size), method=PIL_RESAMPLE_BILINEAR, centering=(0.5, 0.5))
                        except Exception:
                            cropped = cropped.resize((cfg.img_size, cfg.img_size))
                        return cropped, prob, None, None, None
                except Exception:
                    pass
            # Fallback to center-fit
            try:
                fitted = ImageOps.fit(pil, (cfg.img_size, cfg.img_size), method=PIL_RESAMPLE_BILINEAR, centering=(0.5, 0.5))
            except Exception:
                fitted = pil.resize((cfg.img_size, cfg.img_size))
            return fitted, None, None, None, None

        # Helper to export samples from a (path,label) list
        # Keep original absolute path internally for previews (not in CSV)
        rows: List[Tuple[str, int, str, Optional[float], float, str, Optional[Tuple[int,int,int,int]], Optional[Tuple[int,int,int,int]], Optional[Tuple[int,int,int,int]]]] = []
        # Stats accumulators (TRAIN ONLY)
        train_sum = np.zeros(3, dtype=np.float64)
        train_sumsq = np.zeros(3, dtype=np.float64)
        train_count = 0

        def export_split(samples: List[Tuple[str, int]], tag: str):
            nonlocal train_sum, train_sumsq, train_count
            split_dir = os.path.join(out_root, tag)
            os.makedirs(split_dir, exist_ok=True)
            for path, label in progress(samples, total=len(samples), unit="img", desc=f"export {tag}", leave=False):
                try:
                    with Image.open(path) as im:
                        im = im.convert("RGB")
                        crop, prob, le, reye, mo = detect_and_crop(im)
                        bm = blur_metric(crop)
                        # save under class subdir
                        class_dir = os.path.join(split_dir, str(label))
                        os.makedirs(class_dir, exist_ok=True)
                        base = os.path.splitext(os.path.basename(path))[0]
                        out_path = os.path.join(class_dir, f"{base}.jpg")
                        crop.save(out_path, format="JPEG", quality=95)
                        rel = os.path.relpath(out_path, out_root).replace('\\', '/')
                        rows.append((rel, label, tag, prob if prob is not None else float('nan'), bm, path, le, reye, mo))
                        # accumulate stats from TRAIN only
                        if tag == 'train':
                            arr = np.array(crop.convert('RGB'), dtype=np.float32) / 255.0
                            # channel order RGB
                            train_sum += arr.reshape(-1, 3).sum(axis=0)
                            train_sumsq += (arr.reshape(-1, 3) ** 2).sum(axis=0)
                            train_count += arr.shape[0] * arr.shape[1]
                except Exception:
                    continue

        # Build pretraining datasets and splits to get (train/val)
        train_tf, val_tf = make_transforms(cfg.img_size)
        class_map = load_class_map(cfg.class_map_json)
        pretrain_datasets: List[Dataset] = []
        if cfg.affectnet_csv:
            pretrain_datasets.append(GenericFERDataset(
                cfg.affectnet_csv, cfg.affectnet_root, transform=val_tf, class_map=class_map,
                num_classes=cfg.num_classes, enable_alignment=True,
                crop_tight=True, margin_range=(0.08, 0.12), target_size=cfg.img_size
            ))
        if cfg.rafdb_csv:
            pretrain_datasets.append(GenericFERDataset(
                cfg.rafdb_csv, cfg.rafdb_root, transform=val_tf, class_map=class_map,
                num_classes=cfg.num_classes, enable_alignment=True,
                crop_tight=True, margin_range=(0.08, 0.12), target_size=cfg.img_size
            ))
        if len(pretrain_datasets) == 0:
            raise ValueError("No pretraining datasets provided for export.")
        concat = ConcatDataset(pretrain_datasets)
        total = len(concat)
        val_size = max(1000, int(0.02 * total))
        train_size = total - val_size
        train_subset, val_subset = random_split(concat, [train_size, val_size], generator=torch.Generator().manual_seed(123))

        # Materialize sample lists from Subsets
        def subset_samples(sub):
            result: List[Tuple[str, int]] = []
            base = sub.dataset
            # ConcatDataset
            datasets = base.datasets
            # Precompute offsets
            lengths = [len(d) for d in datasets]
            cum = np.cumsum([0] + lengths)
            for i in sub.indices:
                # find dataset index
                di = int(np.searchsorted(cum, i, side='right') - 1)
                local = i - cum[di]
                d = datasets[di]
                path, label = d.samples[local]
                result.append((path, label))
            return result

        export_split(subset_samples(train_subset), "train")
        export_split(subset_samples(val_subset), "val")

        # Optional test sets from eval CSVs
        if args.affectnet_eval_csv:
            ds_eval = GenericFERDataset(args.affectnet_eval_csv, args.affectnet_eval_root or cfg.affectnet_root,
                                        transform=val_tf, class_map=class_map, num_classes=cfg.num_classes,
                                        enable_alignment=True, crop_tight=True, margin_range=(0.08, 0.12), target_size=cfg.img_size)
            export_split(ds_eval.samples, "test")
        if args.rafdb_eval_csv:
            ds_eval = GenericFERDataset(args.rafdb_eval_csv, args.rafdb_eval_root or cfg.rafdb_root,
                                        transform=val_tf, class_map=class_map, num_classes=cfg.num_classes,
                                        enable_alignment=True, crop_tight=True, margin_range=(0.08, 0.12), target_size=cfg.img_size)
            export_split(ds_eval.samples, "test")

        # Write CSV
        csv_path = os.path.join(out_root, args.export_csv_name)
        with open(csv_path, 'w', newline='', encoding='utf-8') as fcsv:
            writer = csv.writer(fcsv)
            writer.writerow(["path", "label", "dataset", "landmarks_quality", "blur_metric", "left_eye_roi", "right_eye_roi", "mouth_roi"])
            def fmt_roi(box):
                if box is None:
                    return ""
                x,y,w,h = box
                return f"{x},{y},{w},{h}"
            for rel, label, tag, prob, bm, _orig, le, reye, mo in rows:
                writer.writerow([rel, label, tag, ("" if prob != prob else f"{prob:.6f}"), f"{bm:.6f}", fmt_roi(le), fmt_roi(reye), fmt_roi(mo)])
        print(f"Wrote metadata CSV: {csv_path} ({len(rows)} rows)")

        # Save normalization stats used
        # Compute train-only mean/std (per-channel RGB)
        if train_count > 0:
            mean = (train_sum / train_count).tolist()
            var = (train_sumsq / train_count) - np.square(train_sum / train_count)
            std = np.sqrt(np.maximum(var, 1e-12)).tolist()
        else:
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
        norm_stats = {
            'img_size': cfg.img_size,
            'channels': 'rgb',
            'computed_on': 'train',
            'count_pixels_per_channel': int(train_count),
            'mean': mean,
            'std': std,
        }
        with open(os.path.join(out_root, 'normalization_stats.json'), 'w', encoding='utf-8') as fjs:
            json.dump(norm_stats, fjs, indent=2)
        print(f"Saved normalization stats to: {os.path.join(out_root, 'normalization_stats.json')}")

        # ---------------- Preview popup with before/after comparisons ----------------
        try:
            if len(rows) > 0:
                k = min(6, len(rows))
                samples = random.sample(rows, k)
                # Build a vertical collage of paired before/after (112x112 each)
                pair_w, pair_h = cfg.img_size * 2 + 10, cfg.img_size  # 10px spacer
                grid = Image.new('RGB', (pair_w, pair_h * k + 5 * (k - 1)), color=COL_20)
                draw = ImageDraw.Draw(grid)
                y = 0
                for i, sample in enumerate(samples):
                    rel = sample[0]
                    orig_path = sample[5]
                    after_path = os.path.join(out_root, rel)
                    try:
                        with Image.open(orig_path) as im_before:
                            im_before = ImageOps.fit(im_before.convert('RGB'), (cfg.img_size, cfg.img_size), method=PIL_RESAMPLE_BILINEAR)
                    except Exception:
                        im_before = Image.new('RGB', (cfg.img_size, cfg.img_size), color=COL_60)
                    try:
                        with Image.open(after_path) as im_after:
                            im_after = ImageOps.fit(im_after.convert('RGB'), (cfg.img_size, cfg.img_size), method=PIL_RESAMPLE_BILINEAR)
                    except Exception:
                        im_after = Image.new('RGB', (cfg.img_size, cfg.img_size), color=COL_100)
                    # paste with margin
                    grid.paste(im_before, (0, y))
                    # spacer rectangle
                    spacer_x1 = cfg.img_size
                    spacer_x2 = cfg.img_size + 10
                    draw.rectangle([spacer_x1, y, spacer_x2, y + cfg.img_size], fill=(20, 20, 20))
                    grid.paste(im_after, (cfg.img_size + 10, y))
                    y += pair_h + 5

                preview_path = os.path.join(out_root, 'preview_before_after.jpg')
                grid.save(preview_path, format='JPEG', quality=95)
                print(f"Saved preview collage: {preview_path}")

                # Try to open with default viewer (popup)
                try:
                    if os.name == 'nt':
                        os.startfile(preview_path)  # type: ignore[attr-defined]
                    else:
                        opener = 'open' if sys.platform == 'darwin' else 'xdg-open'
                        subprocess.Popen([opener, preview_path])
                    print("Opened preview image in default viewer.")
                except Exception:
                    print("Note: could not auto-open preview image; please open it manually.")
        except Exception as e:
            print(f"Preview generation failed: {e}")

        return

    # ---------------- DMD face-frame export mode ----------------
    if args.dmd_export_faces is not None:
        seed_everything(args.seed)
        out_root = os.path.abspath(args.dmd_export_faces)
        os.makedirs(out_root, exist_ok=True)

        # Ensure cv2 available for video decoding
        if cv2 is None:
            raise RuntimeError("OpenCV (cv2) is required for --dmd_export_faces. Install with: pip install opencv-python")

        MTCNN = _try_import_mtcnn()
        mtcnn = None
        if MTCNN is not None:
            mtcnn = MTCNN(keep_all=False, device=('cuda' if torch.cuda.is_available() else 'cpu'))
        else:
            print("Warning: facenet_pytorch not available; exporting center-cropped resized frames without landmarks.")

        # Load JSON annotations if provided
        annotations = None
        if args.dmd_json and os.path.exists(args.dmd_json):
            try:
                with open(args.dmd_json, 'r', encoding='utf-8') as fj:
                    annotations = json.load(fj)
            except Exception as e:
                print(f"Warning: failed to read DMD JSON: {e}")

        dmd_root = args.dmd_root or os.path.join(os.getcwd(), 'datasets', 'dmd')
        if not os.path.isdir(dmd_root):
            raise FileNotFoundError(f"DMD root not found: {dmd_root}")

        # Helper to resolve label and split from session directory using json (best-effort)
        def resolve_label_split(session_dir: str) -> Tuple[Optional[int], str]:
            rel = os.path.relpath(session_dir, dmd_root).replace('\\', '/')
            split = 'train'
            label_val: Optional[int] = None
            # Heuristic 1: parent folder name numeric label
            try:
                parent = os.path.basename(os.path.dirname(session_dir))
                if parent.isdigit():
                    label_val = int(parent)
            except Exception:
                pass
            # Heuristic 2: JSON lookup
            try:
                if isinstance(annotations, dict):
                    # Try exact path key
                    entry = annotations.get(rel) or annotations.get('/' + rel)
                    if isinstance(entry, dict):
                        if label_val is None:
                            lv = entry.get('label')
                            if isinstance(lv, int):
                                label_val = lv
                        sp = entry.get('split')
                        if isinstance(sp, str):
                            split = sp
            except Exception:
                pass
            return label_val, split

        # Walk sessions and pick face videos for chosen modality
        video_exts = {'.mp4', '.avi', '.mov', '.mkv'}
        modality_pref = {
            'rgb': ['rgb', 'color', 'vis'],
            'ir': ['ir', 'infra'],
            'depth': ['depth']
        }
        def is_face_video(name: str) -> bool:
            n = name.lower()
            return 'face' in n
        def matches_modality(name: str, want: str) -> bool:
            if want == 'auto':
                return True
            n = name.lower()
            return any(tok in n for tok in modality_pref.get(want, []))

        candidates: List[Tuple[str, Optional[int], str]] = []  # (video_path, label, split)
        for root, dirs, files in os.walk(dmd_root):
            # session folder ends with s{number}
            base = os.path.basename(root)
            if not base.lower().startswith('s'):
                # not necessarily reliable; still scan
                pass
            for fn in files:
                ext = os.path.splitext(fn)[1].lower()
                if ext not in video_exts:
                    continue
                if not is_face_video(fn):
                    continue
                if not matches_modality(fn, args.dmd_modality):
                    continue
                session_dir = root
                label, split = resolve_label_split(session_dir)
                vid_path = os.path.join(root, fn)
                candidates.append((vid_path, label, split))

        if len(candidates) == 0:
            raise ValueError("No DMD face videos found. Check modality filter and path.")

        # Prefer RGB in auto mode
        if args.dmd_modality == 'auto':
            pri = ['rgb', 'ir', 'depth']
            def modality_of(name: str) -> str:
                n = name.lower()
                for key in pri:
                    if any(tok in n for tok in modality_pref[key]):
                        return key
                return 'rgb'
            # Group by modality, pick highest-priority set
            for wanted in pri:
                filtered = [c for c in candidates if modality_of(c[0]) == wanted]
                if filtered:
                    candidates = filtered
                    print(f"Auto-selected DMD modality: {wanted} ({len(candidates)} videos)")
                    break

        # Sampling setup
        target_fps = max(0.1, float(args.dmd_fps))
        max_frames = max(1, int(args.dmd_max_frames))

        rows_dmd: List[Tuple[str, int, str, int, str, float, Optional[float], Optional[Tuple[int,int,int,int]], Optional[Tuple[int,int,int,int]], Optional[Tuple[int,int,int,int]], str, float, float]] = []
        # Train-only normalization accumulators; optionally per-camera
        global_sum = np.zeros(3, dtype=np.float64)
        global_sumsq = np.zeros(3, dtype=np.float64)
        global_count = 0
        cam_sum: dict = {}
        cam_sumsq: dict = {}
        cam_count: dict = {}

        def color_mode_for_modality(mod: str) -> str:
            if args.dmd_color == 'rgb':
                return 'rgb'
            if args.dmd_color == 'gray':
                return 'gray'
            # auto
            return 'gray' if mod in ('ir', 'depth') else 'rgb'

        def extract_camera_id(filename: str) -> str:
            base = os.path.basename(filename)
            # regex if provided
            if args.dmd_camera_regex:
                try:
                    m = re.search(args.dmd_camera_regex, base)
                    if m:
                        gid = m.group(1) if m.groups() else m.group(0)
                        return str(gid)
                except Exception:
                    pass
            # heuristics
            m = re.search(r'(cam\d+|c\d+)', base, re.IGNORECASE)
            if m:
                return m.group(1)
            return 'cam0'

        def detect_and_crop_frame(pil: Image.Image) -> Tuple[Image.Image, Optional[float], Optional[Tuple[float, float, float, float]], Optional[Tuple[int,int,int,int]], Optional[Tuple[int,int,int,int]], Optional[Tuple[int,int,int,int]]]:
            if mtcnn is not None:
                try:
                    res = mtcnn.detect(pil, landmarks=True)
                    boxes = probs = points = None
                    if isinstance(res, tuple):
                        if len(res) >= 1:
                            boxes = res[0]
                        if len(res) >= 2:
                            probs = res[1]
                        if len(res) >= 3:
                            points = res[2]
                    if boxes is not None and len(boxes) > 0:
                        idx = 0
                        if probs is not None:
                            idx = int(np.argmax(probs))
                        prob = float(probs[idx]) if probs is not None else None
                        pts5 = points[idx] if points is not None else None
                        if pts5 is not None:
                            aligned, pts_t = align_face_5pt(pil, np.array(pts5, dtype=np.float32), out_size=args.img_size, margin=0.10)
                            le, reye, mo = roi_boxes_from_landmarks(pts_t, out_size=args.img_size, eye_size=64, mouth_size=64)
                            return aligned, prob, None, le, reye, mo
                        # fallback bbox-based
                        x1, y1, x2, y2 = boxes[idx]
                        w = max(1.0, float(x2 - x1))
                        h = max(1.0, float(y2 - y1))
                        m = 0.10
                        dw = m * w
                        dh = m * h
                        nx1 = max(0.0, x1 - dw)
                        ny1 = max(0.0, y1 - dh)
                        nx2 = min(float(pil.width), x2 + dw)
                        ny2 = min(float(pil.height), y2 + dh)
                        cropped = pil.crop((nx1, ny1, nx2, ny2))
                        try:
                            cropped = ImageOps.fit(cropped, (args.img_size, args.img_size), method=PIL_RESAMPLE_BILINEAR, centering=(0.5, 0.5))
                        except Exception:
                            cropped = cropped.resize((args.img_size, args.img_size))
                        return cropped, prob, (x1, y1, x2, y2), None, None, None
                except Exception:
                    pass
            try:
                fitted = ImageOps.fit(pil, (args.img_size, args.img_size), method=PIL_RESAMPLE_BILINEAR, centering=(0.5, 0.5))
            except Exception:
                fitted = pil.resize((args.img_size, args.img_size))
            return fitted, None, None, None, None, None

        # Export frames
        # Track near-duplicates per session (time window 1s)
        session_history: dict = {}

        for vid_path, label, split in progress(candidates, total=len(candidates), unit='vid', desc='dmd export', leave=True):
            # resolve label fallback: use -1 if missing
            if label is None:
                label = -1
            # Create out dir
            sess_dir = os.path.dirname(vid_path)
            # session id for naming
            session_id = os.path.basename(sess_dir)
            class_dir = os.path.join(out_root, split, str(label if label >= 0 else 'unknown'), session_id)
            os.makedirs(class_dir, exist_ok=True)
            cam_id = extract_camera_id(vid_path)

            cap = cv2.VideoCapture(vid_path)
            if not cap.isOpened():
                continue
            fps = cap.get(cv2.CAP_PROP_FPS)
            fps = fps if fps and fps > 0 else 30.0
            stride = max(1, int(round(fps / target_fps)))
            fcount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT) else 0
            duration = (fcount / float(fps)) if fcount > 0 else None
            frame_idx = 0
            saved = 0
            history = session_history.setdefault(session_id, [])  # list of (time_sec, phash)
            while True:
                ret = cap.grab()
                if not ret:
                    break
                if frame_idx % stride == 0:
                    ret2, frame = cap.retrieve()
                    if not ret2:
                        frame_idx += 1
                        continue
                    # BGR->RGB
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil = Image.fromarray(frame)
                    crop, prob, bbox, le, reye, mo = detect_and_crop_frame(pil)
                    # color handling
                    cmode = color_mode_for_modality(args.dmd_modality)
                    if cmode == 'gray':
                        crop_save = crop.convert('L').convert('RGB')  # save as 3-channel jpeg but grayscale content
                        arr_stats = np.array(crop.convert('L'), dtype=np.float32) / 255.0
                        # treat as single-channel for stats
                        ch_sums = np.array([arr_stats.sum()]*3, dtype=np.float64)
                        ch_sumsq = np.array([(arr_stats**2).sum()]*3, dtype=np.float64)
                        ch_count = arr_stats.size
                    else:
                        crop_save = crop
                        arr = np.array(crop.convert('RGB'), dtype=np.float32) / 255.0
                        flat = arr.reshape(-1, 3)
                        ch_sums = flat.sum(axis=0).astype(np.float64)
                        ch_sumsq = (flat**2).sum(axis=0).astype(np.float64)
                        ch_count = arr.shape[0] * arr.shape[1]

                    # QA filters
                    # 1) face presence/size: drop if bbox height < 64 px
                    if bbox is not None:
                        _, y1b, _, y2b = bbox
                        if (y2b - y1b) < 64:
                            frame_idx += 1
                            continue

                    # 2) blur: variance of Laplacian < 80 -> drop
                    lap_var = laplacian_variance(crop)
                    if lap_var < 80.0:
                        frame_idx += 1
                        continue

                    # 3) exposure: mean gray < 20 or > 235 -> drop
                    mean_gray = float(np.array(crop.convert('L')).mean())
                    if mean_gray < 20.0 or mean_gray > 235.0:
                        frame_idx += 1
                        continue

                    # 4) extreme occlusions via low skin coverage (< 0.5) -> drop
                    skin_frac = skin_coverage_fraction(crop)
                    if skin_frac < 0.5:
                        frame_idx += 1
                        continue

                    # 5) near-duplicates via pHash within 1 second
                    time_sec = frame_idx / float(fps)
                    ph = compute_phash(crop)
                    dup = False
                    for tprev, hp in history:
                        if abs(time_sec - tprev) <= 1.0 and hamming_distance(ph, hp) <= 6:
                            dup = True
                            break
                    if dup:
                        frame_idx += 1
                        continue

                    # Boundary exclusion
                    if args.dmd_exclude_boundary_s and args.dmd_exclude_boundary_s > 0:
                        excl = float(args.dmd_exclude_boundary_s)
                        if time_sec < excl or (duration is not None and time_sec > (duration - excl)):
                            frame_idx += 1
                            continue

                    bm = lap_var  # use Laplacian variance as blur metric
                    out_name = f"f{frame_idx:06d}.jpg"
                    out_path = os.path.join(class_dir, out_name)
                    try:
                        crop_save.save(out_path, format='JPEG', quality=95)
                    except Exception:
                        pass
                    rel = os.path.relpath(out_path, out_root).replace('\\', '/')
                    rows_dmd.append((rel, label, session_id, frame_idx, split, bm, prob, le, reye, mo, cam_id, float(fps), float(time_sec)))
                    # record history
                    history.append((time_sec, ph))
                    # prune old history beyond 1.5s to limit growth
                    if len(history) > 0:
                        cutoff = time_sec - 1.5
                        history = [item for item in history if item[0] >= cutoff]
                        session_history[session_id] = history
                    # accumulate train-only stats per camera
                    if split == 'train':
                        global_sum += ch_sums
                        global_sumsq += ch_sumsq
                        global_count += ch_count
                        cam_sum.setdefault(cam_id, np.zeros(3, dtype=np.float64))
                        cam_sumsq.setdefault(cam_id, np.zeros(3, dtype=np.float64))
                        cam_count.setdefault(cam_id, 0)
                        cam_sum[cam_id] += ch_sums
                        cam_sumsq[cam_id] += ch_sumsq
                        cam_count[cam_id] += ch_count

                    saved += 1
                    if saved >= max_frames:
                        break
                frame_idx += 1
            cap.release()

        # Write CSV
        csv_path = os.path.join(out_root, 'dmd_frames.csv')
        with open(csv_path, 'w', newline='', encoding='utf-8') as fcsv:
            writer = csv.writer(fcsv)
            writer.writerow(["path", "label", "session", "frame_idx", "split", "blur_metric", "landmarks_quality", "modality", "left_eye_roi", "right_eye_roi", "mouth_roi", "camera_id", "fps", "time_sec"])
            def fmt_roi(box):
                if box is None:
                    return ""
                x,y,w,h = box
                return f"{x},{y},{w},{h}"
            for rel, label, session_id, fidx, split, bm, prob, le, reye, mo, cam_id, fps_val, tsec in rows_dmd:
                writer.writerow([rel, label, session_id, fidx, split, f"{bm:.6f}", ("" if prob is None else f"{prob:.6f}"), args.dmd_modality, fmt_roi(le), fmt_roi(reye), fmt_roi(mo), cam_id, f"{fps_val:.3f}", f"{tsec:.3f}"]) 
        print(f"Wrote DMD frames CSV: {csv_path} ({len(rows_dmd)} rows)")

        # Save normalization stats
        # Train-only normalization stats (global and per-camera)
        if global_count > 0:
            g_mean = (global_sum / global_count).tolist()
            g_var = (global_sumsq / global_count) - np.square(global_sum / global_count)
            g_std = np.sqrt(np.maximum(g_var, 1e-12)).tolist()
        else:
            g_mean = [0.485, 0.456, 0.406]
            g_std = [0.229, 0.224, 0.225]
        per_cam = {}
        for cid in cam_sum.keys():
            cnt = cam_count.get(cid, 0)
            if cnt > 0:
                m = (cam_sum[cid] / cnt).tolist()
                v = (cam_sumsq[cid] / cnt) - np.square(cam_sum[cid] / cnt)
                s = np.sqrt(np.maximum(v, 1e-12)).tolist()
                per_cam[cid] = {'mean': m, 'std': s, 'count_pixels_per_channel': int(cnt)}
        norm_stats = {
            'img_size': args.img_size,
            'channels': color_mode_for_modality(args.dmd_modality),
            'computed_on': 'train',
            'global': {
                'mean': g_mean,
                'std': g_std,
                'count_pixels_per_channel': int(global_count),
            },
            'per_camera': per_cam,
            'modality': args.dmd_modality,
        }
        with open(os.path.join(out_root, 'normalization_stats.json'), 'w', encoding='utf-8') as fjs:
            json.dump(norm_stats, fjs, indent=2)

        # Quick preview collage
        try:
            if len(rows_dmd) > 0:
                k = min(6, len(rows_dmd))
                samples = random.sample(rows_dmd, k)
                w = args.img_size
                h = args.img_size
                grid = Image.new('RGB', (w * k + 5 * (k - 1), h), color=COL_20)
                x = 0
                for rel, *_rest in samples:
                    apath = os.path.join(out_root, rel)
                    try:
                        with Image.open(apath) as im:
                            im = ImageOps.fit(im.convert('RGB'), (w, h), method=PIL_RESAMPLE_BILINEAR)
                    except Exception:
                        im = Image.new('RGB', (w, h), color=COL_60)
                    grid.paste(im, (x, 0))
                    x += w + 5
                preview_path = os.path.join(out_root, 'preview_dmd_samples.jpg')
                grid.save(preview_path, format='JPEG', quality=95)
                print(f"Saved DMD preview: {preview_path}")
                try:
                    if os.name == 'nt':
                        os.startfile(preview_path)  # type: ignore[attr-defined]
                    else:
                        opener = 'open' if sys.platform == 'darwin' else 'xdg-open'
                        subprocess.Popen([opener, preview_path])
                except Exception:
                    pass
        except Exception as e:
            print(f"Preview generation failed: {e}")

        return

    # ---------------- Smoothing utility (evaluation-only) ----------------
    if args.smooth_frames_csv is not None and args.smooth_out is not None:
        in_path = args.smooth_frames_csv
        out_path = args.smooth_out
        if not os.path.isfile(in_path):
            raise FileNotFoundError(f"Frames CSV not found: {in_path}")

        # Load rows
        if pd is None:
            raise RuntimeError("Pandas is required for smoothing: pip install pandas")
        try:
            df = pd.read_csv(in_path)
        except Exception as e:
            raise RuntimeError(f"Failed to read CSV: {e}")
        # Ensure required columns
        required = ['session', 'frame_idx', 'label']
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing column '{col}' in {in_path}")
        # Determine time stamps
        if 'time_sec' in df.columns:
            df['time_sec'] = df['time_sec'].astype(float)
        else:
            fps_val = float(args.smooth_fps) if args.smooth_fps else None
            if fps_val is None and 'fps' in df.columns:
                # use per-row fps if present
                try:
                    fps_val = float(df['fps'].iloc[0])
                except Exception:
                    fps_val = 30.0
            if fps_val is None:
                fps_val = 30.0
            df['time_sec'] = df['frame_idx'].astype(float) / fps_val

        method = args.smooth_method
        window_s = max(0.01, float(args.smooth_window_s))
        alpha = max(1e-6, min(1.0, float(args.smooth_alpha)))

        def smooth_session(g: 'pd.DataFrame') -> 'pd.Series':
            g = g.sort_values('time_sec').reset_index(drop=True)
            labels = g['label'].astype(int).to_numpy()
            times = g['time_sec'].to_numpy()
            uniq = np.unique(labels)
            if method == 'majority':
                res = np.empty_like(labels)
                q = deque()  # stores indices currently in window
                left = 0
                for i in range(len(labels)):
                    t = times[i]
                    # include i
                    # advance left boundary
                    while left < len(labels) and times[left] < t - window_s/2:
                        left += 1
                    # window indices in [left, i] then extend to right side
                    right = i
                    # extend forward until time > t+win/2
                    j = i
                    while j+1 < len(labels) and times[j+1] <= t + window_s/2:
                        j += 1
                    window_labels = labels[left:j+1]
                    if window_labels.size == 0:
                        res[i] = labels[i]
                    else:
                        c = Counter(window_labels.tolist())
                        # pick most common; tie -> original
                        most, count = None, -1
                        for k, v in c.items():
                            if v > count:
                                most, count = k, v
                        res[i] = most if most is not None else labels[i]
                return pd.Series(res, index=g.index)
            else:  # ema
                # per-class EMA over one-hot; choose argmax
                scores = np.zeros((len(uniq), len(labels)), dtype=np.float64)
                prev = np.zeros(len(uniq), dtype=np.float64)
                for i, y in enumerate(labels):
                    onehot = (uniq == y).astype(np.float64)
                    prev = alpha * onehot + (1 - alpha) * prev
                    scores[:, i] = prev
                idx = scores.argmax(axis=0)
                smoothed = uniq[idx]
                return pd.Series(smoothed, index=g.index)

        df['label_smoothed'] = df.groupby('session', group_keys=False).apply(smooth_session)
        os.makedirs(os.path.dirname(os.path.abspath(out_path)) or '.', exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"Wrote smoothed labels CSV: {out_path}")
        return

    # ---------------- Feature extractor/export modes ----------------
    if args.export_feature_extractor is not None or args.extract_features_csv is not None:
        seed_everything(args.seed)
        device = torch_device(getattr(args, "device", "auto"))
        # Log device
        try:
            if isinstance(device, torch.device) and device.type == "cuda":
                num = torch.cuda.device_count()
                names = [torch.cuda.get_device_name(i) for i in range(num)]
                print(f"Using device: cuda (GPUs={num}) -> {names}")
            elif isinstance(device, torch.device) and device.type == "mps":
                print("Using device: mps (Apple Silicon)")
            elif isinstance(device, torch.device):
                print(f"Using device: {device}")
            else:
                print(f"Using device object: {device}")
        except Exception:
            print(f"Using device: {device}")

        # Build backbone
        backbone = EfficientNetFeatureExtractor(args.eff_variant, pretrained=True).to(device)
        if args.load_backbone_ckpt and os.path.exists(args.load_backbone_ckpt):
            ckpt = torch.load(args.load_backbone_ckpt, map_location=device)
            state = ckpt.get('state_dict', ckpt if isinstance(ckpt, dict) else None)
            if isinstance(state, dict):
                missing, unexpected = backbone.load_state_dict(state, strict=False)
                if missing or unexpected:
                    print(f"Loaded backbone with missing={len(missing)} unexpected={len(unexpected)} keys")
            else:
                print("Warning: could not find state_dict in checkpoint; using ImageNet weights.")

        print(f"Backbone variant={args.eff_variant} feature_dim={backbone.feature_dim}")

        if args.export_feature_extractor is not None:
            # Save a simple checkpoint with variant and state_dict
            out_path = args.export_feature_extractor
            os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
            torch.save({
                'type': 'efficientnet_backbone',
                'variant': args.eff_variant,
                'state_dict': backbone.state_dict(),
                'feature_dim': backbone.feature_dim,
            }, out_path)
            print(f"Exported feature extractor to: {out_path}")

        if args.extract_features_csv is not None and args.extract_features_out is not None:
            # Build dataset/loader with val transforms
            _, val_tf = make_transforms(args.img_size)
            class_map = load_class_map(args.class_map_json)
            ds = GenericFERDataset(args.extract_features_csv, args.extract_features_root, transform=val_tf,
                                   class_map=class_map, num_classes=args.num_classes, enable_alignment=True)
            num_workers = get_num_workers()
            pin_mem = isinstance(device, torch.device) and getattr(device, 'type', 'cpu') == 'cuda'
            dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_mem)
            backbone.eval()
            feats_list: List[np.ndarray] = []
            labels_list: List[int] = []
            paths_list: List[str] = [p for p, _ in ds.samples]
            with torch.no_grad():
                iterable = progress(dl, total=len(dl), unit="batch", desc="extract", leave=False)
                for images, labels in iterable:
                    images = images.to(device, non_blocking=True)
                    try:
                        with torch.autocast(device_type='cuda', enabled=(isinstance(device, torch.device) and device.type == 'cuda')):
                            feats = backbone(images)
                    except Exception:
                        feats = backbone(images)
                    feats = feats.detach().cpu().numpy()
                    feats_list.append(feats)
                    labels_list.extend(labels.detach().cpu().numpy().tolist())
            feats_all = np.concatenate(feats_list, axis=0) if feats_list else np.zeros((0, backbone.feature_dim), dtype=np.float32)
            labels_all = np.array(labels_list, dtype=np.int64)
            out_npz = args.extract_features_out
            os.makedirs(os.path.dirname(os.path.abspath(out_npz)) or ".", exist_ok=True)
            np.savez_compressed(out_npz, features=feats_all, labels=labels_all, paths=np.array(paths_list, dtype=object))
            print(f"Saved features to: {out_npz}  shape={feats_all.shape}")

        # If either export or extract was requested, we are done
        if args.export_feature_extractor is not None or (args.extract_features_csv is not None and args.extract_features_out is not None):
            return

    cfg = TrainConfig(
        affectnet_csv=args.affectnet_csv,
        affectnet_root=args.affectnet_root,
        rafdb_csv=args.rafdb_csv,
        rafdb_root=args.rafdb_root,
        dmd_csv=args.dmd_csv,
        dmd_root=args.dmd_root,
        class_map_json=args.class_map_json,
        num_classes=args.num_classes,
        eff_variant=args.eff_variant,
        img_size=args.img_size,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs_pretrain=args.epochs_pretrain,
        epochs_main=args.epochs_main,
        seed=args.seed,
        out_dir=args.out_dir,
        amp=(not args.no_amp),
        balance_sampling=(not args.no_balance),
    )

    seed_everything(cfg.seed)
    device = torch_device(getattr(args, "device", "auto"))
    if getattr(args, "pretrain_only", False):
        print("Note: --pretrain_only specified. In this data-only build, training is disabled; when training is re-added, this flag will run only the AffectNet/RAF-DB pretraining phase and skip the DMD finetune phase.")
    try:
        if isinstance(device, torch.device) and device.type == "cuda":
            num = torch.cuda.device_count()
            names = [torch.cuda.get_device_name(i) for i in range(num)]
            print(f"Using device: cuda (GPUs={num}) -> {names}")
        elif isinstance(device, torch.device) and device.type == "mps":
            print("Using device: mps (Apple Silicon)")
        elif isinstance(device, torch.device):
            print(f"Using device: {device}")
        else:
            print(f"Using device object: {device}")
    except Exception:
        print(f"Using device: {device}")

    # Data
    pretrain_train_loader, pretrain_val_loader, dmd_train_loader, dmd_val_loader = build_loaders(cfg, device)

    # Inspection-only pass over a few batches to validate pipeline
    def _peek(loader, name: str, max_batches: int):
        if loader is None:
            return
        print(f"\nInspecting loader: {name}")
        bcount = 0
        t0 = time.time()
        for images, labels in loader:
            bcount += 1
            print(f"  batch {bcount}: images={tuple(images.shape)} dtype={images.dtype} min={float(images.min()):.3f} max={float(images.max()):.3f}")
            try:
                # quick label stats for this batch
                labs = labels.detach().cpu().numpy().tolist()
                c = Counter(labs)
                print(f"  labels: {dict(c)}")
            except Exception:
                pass
            if bcount >= max_batches:
                break
        t1 = time.time()
        print(f"  iterated {bcount} batch(es) in {t1 - t0:.2f}s")

    _peek(pretrain_train_loader, "pretrain_train", args.inspect_batches)
    _peek(pretrain_val_loader, "pretrain_val", 1)
    if dmd_train_loader is not None:
        _peek(dmd_train_loader, "dmd_train", min(1, args.inspect_batches))
    if dmd_val_loader is not None:
        _peek(dmd_val_loader, "dmd_val", 1)

    print("\nData loading and preprocessing check complete.")


if __name__ == "__main__":
    main()

