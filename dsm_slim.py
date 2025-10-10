# DO NOT RUN
# script requires datasets to run
# code stripped of debug and cli functions, made easier to read

import os
import json
import random
import io
import csv
from dataclasses import dataclass
from typing import Optional, List, Tuple

import numpy as np
import multiprocessing as mp
from collections import Counter
from facenet_pytorch import MTCNN as _MTCNN
import cv2

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

# Pillow resampling compatibility
try:
    from PIL.Image import Resampling as _PILResampling
    PIL_RESAMPLE_BILINEAR: _Any = _PILResampling.BILINEAR
except Exception:
    # Fallback for older Pillow; value 2 corresponds to bilinear
    PIL_RESAMPLE_BILINEAR: _Any = getattr(Image, 'BILINEAR', 2)


# /////////////////////////////
# Utilities
# /////////////////////////////

def seed_everything(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


_DML_WARNED = False


def torch_device(prefer: Optional[str] = "auto") -> torch.device:
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
                print("DirectML not available; falling back to CUDA/MPS/CPU.")
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


def safe_fit(img: Image.Image, size: Tuple[int, int]) -> Image.Image:
    try:
        return ImageOps.fit(img, size, method=PIL_RESAMPLE_BILINEAR, centering=(0.5, 0.5))
    except Exception:
        return img.resize(size)


def load_class_map(path: Optional[str]) -> Optional[dict]:
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# /////////////////////////////
# Dataset definitions
# /////////////////////////////


class CsvImageDataset(Dataset):

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
                    # Assume success environment: skip unknown labels
                    continue
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


# /////////////////////////////
# Cleaning and preprocessing for generic FER
# /////////////////////////////


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


def blur_metric(img: Image.Image) -> float:
# Sobel gradient magnitude variance (higher is sharper)
    t = TF.to_tensor(img.convert("L")).unsqueeze(0)
    sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32, device=t.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32, device=t.device).view(1, 1, 3, 3)
    with torch.no_grad():
        gx = F.conv2d(t, sobel_x, padding=1)
        gy = F.conv2d(t, sobel_y, padding=1)
        mag = (gx.pow(2) + gy.pow(2)).sqrt()
        score = mag.var().item() * 1000.0
    return float(score)


def is_blurry(img: Image.Image, edge_thresh: float = 4.0) -> bool:
    return blur_metric(img) < edge_thresh


def laplacian_variance(img: Image.Image) -> float:
    try:
        if cv2 is None:
            return blur_metric(img)
    except Exception:
        return blur_metric(img)
    gray = np.array(img.convert('L'), dtype=np.uint8)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(lap.var())

class GenericFERDataset(CsvImageDataset):

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


# /////////////////////////////
# CSV generation from folder-per-class
# /////////////////////////////


def build_weighted_sampler(labels: List[int]) -> WeightedRandomSampler:
    counts = Counter(labels)
    num_samples = len(labels)
    # Inverse frequency weights
    weights = [num_samples / (counts[l] * len(counts)) for l in labels]
    return WeightedRandomSampler(weights, num_samples=num_samples, replacement=True)


# /////////////////////////////
# DMD export
# /////////////////////////////


def export_dmd_faces(
    dmd_root: str,
    out_root: str,
    modality: str = 'auto',
    fps: float = 4.0,
    max_frames: int = 400,
    img_size: int = 112,
    annotations_json: Optional[str] = None,
    exclude_boundary_s: Optional[float] = None,
    seed: int = 42,
):
    
    os.makedirs(out_root, exist_ok=True)
    random.seed(seed)

    # Detector
    MTCNN = _try_import_mtcnn()
    mtcnn = None
    if MTCNN is not None:
        mtcnn = MTCNN(keep_all=False, device=('cuda' if torch.cuda.is_available() else 'cpu'))

    # Optional annotations
    annotations = None
    if annotations_json and os.path.exists(annotations_json):
        try:
            with open(annotations_json, 'r', encoding='utf-8') as fj:
                annotations = json.load(fj)
        except Exception:
            annotations = None

    def resolve_label_split(session_dir: str) -> Tuple[Optional[int], str]:
        split = 'train'
        label_val: Optional[int] = None
        try:
            parent = os.path.basename(os.path.dirname(session_dir))
            if parent.isdigit():
                label_val = int(parent)
        except Exception:
            pass
        try:
            if isinstance(annotations, dict):
                rel = os.path.relpath(session_dir, dmd_root).replace('\\', '/')
                entry = annotations.get(rel) or annotations.get('/' + rel)
                if isinstance(entry, dict):
                    lv = entry.get('label')
                    if isinstance(lv, int):
                        label_val = lv
                    sp = entry.get('split')
                    if isinstance(sp, str):
                        split = sp
        except Exception:
            pass
        return label_val, split

    video_exts = {'.mp4', '.avi', '.mov', '.mkv'}
    modality_pref = {
        'rgb': ['rgb', 'color', 'vis'],
        'ir': ['ir', 'infra'],
        'depth': ['depth']
    }
    def is_face_video(name: str) -> bool:
        return 'face' in name.lower()
    def matches_modality(name: str, want: str) -> bool:
        if want == 'auto':
            return True
        n = name.lower()
        return any(tok in n for tok in modality_pref.get(want, []))

    candidates: List[Tuple[str, Optional[int], str]] = []
    for root, _dirs, files in os.walk(dmd_root):
        for fn in files:
            ext = os.path.splitext(fn)[1].lower()
            if ext not in video_exts:
                continue
            if not is_face_video(fn) or not matches_modality(fn, modality):
                continue
            session_dir = root
            label, split = resolve_label_split(session_dir)
            vid_path = os.path.join(root, fn)
            candidates.append((vid_path, label, split))

    if modality == 'auto' and candidates:
        pri = ['rgb', 'ir', 'depth']
        def modality_of(name: str) -> str:
            n = name.lower()
            for key in pri:
                if any(tok in n for tok in modality_pref[key]):
                    return key
            return 'rgb'
        for wanted in pri:
            filtered = [c for c in candidates if modality_of(c[0]) == wanted]
            if filtered:
                candidates = filtered
                break

    rows = []
    os.makedirs(out_root, exist_ok=True)

    def detect_and_crop(pil: Image.Image):
        if mtcnn is not None:
            try:
                res = mtcnn.detect(pil, landmarks=True)
                boxes = probs = points = None
                if isinstance(res, tuple):
                    if len(res) >= 1: boxes = res[0]
                    if len(res) >= 2: probs = res[1]
                    if len(res) >= 3: points = res[2]
                if boxes is not None and len(boxes) > 0:
                    idx = 0
                    if probs is not None:
                        idx = int(np.argmax(probs))
                    prob = float(probs[idx]) if probs is not None else None
                    pts5 = points[idx] if points is not None else None
                    if pts5 is not None:
                        aligned, pts_t = align_face_5pt(pil, np.array(pts5, dtype=np.float32), out_size=img_size, margin=0.10)
                        le, reye, mo = roi_boxes_from_landmarks(pts_t, out_size=img_size, eye_size=64, mouth_size=64)
                        return aligned, prob, None, le, reye, mo
                    x1, y1, x2, y2 = boxes[idx]
                    w = max(1.0, float(x2 - x1)); h = max(1.0, float(y2 - y1))
                    m = 0.10
                    dw = m * w; dh = m * h
                    nx1 = max(0.0, x1 - dw); ny1 = max(0.0, y1 - dh)
                    nx2 = min(float(pil.width), x2 + dw); ny2 = min(float(pil.height), y2 + dh)
                    cropped = pil.crop((nx1, ny1, nx2, ny2))
                    cropped = safe_fit(cropped, (img_size, img_size))
                    return cropped, prob, (x1, y1, x2, y2), None, None, None
            except Exception:
                pass
        return safe_fit(pil, (img_size, img_size)), None, None, None, None, None

    for vid_path, label, split in candidates:
        if label is None:
            label = -1
        sess_dir = os.path.dirname(vid_path)
        session_id = os.path.basename(sess_dir)
        class_dir = os.path.join(out_root, split, str(label if label >= 0 else 'unknown'), session_id)
        os.makedirs(class_dir, exist_ok=True)

        cap = cv2.VideoCapture(vid_path)
        if not cap.isOpened():
            continue
        cap_fps = cap.get(cv2.CAP_PROP_FPS)
        cap_fps = cap_fps if cap_fps and cap_fps > 0 else 30.0
        stride = max(1, int(round(cap_fps / fps)))
        fcount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT) else 0
        duration = (fcount / float(cap_fps)) if fcount > 0 else None
        frame_idx = 0
        saved = 0
        history: List[Tuple[float, int]] = []
        while True:
            ret = cap.grab()
            if not ret:
                break
            if frame_idx % stride == 0:
                ret2, frame = cap.retrieve()
                if not ret2:
                    frame_idx += 1
                    continue
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil = Image.fromarray(frame)
                crop, prob, bbox, le, reye, mo = detect_and_crop(pil)

                # QA filters
                if bbox is not None:
                    _, y1b, _, y2b = bbox
                    if (y2b - y1b) < 64:
                        frame_idx += 1
                        continue
                lap_var = laplacian_variance(crop)
                if lap_var < 80.0:
                    frame_idx += 1
                    continue
                mean_gray = float(np.array(crop.convert('L')).mean())
                if mean_gray < 20.0 or mean_gray > 235.0:
                    frame_idx += 1
                    continue
                skin_frac = skin_coverage_fraction(crop)
                if skin_frac < 0.5:
                    frame_idx += 1
                    continue
                time_sec = frame_idx / float(cap_fps)
                ph = compute_phash(crop)
                if any(abs(time_sec - tprev) <= 1.0 and hamming_distance(ph, hp) <= 6 for tprev, hp in history):
                    frame_idx += 1
                    continue
                if exclude_boundary_s and ((time_sec < exclude_boundary_s) or (duration is not None and time_sec > (duration - exclude_boundary_s))):
                    frame_idx += 1
                    continue

                out_name = f"f{frame_idx:06d}.jpg"
                out_path = os.path.join(class_dir, out_name)
                try:
                    crop.save(out_path, format='JPEG', quality=95)
                except Exception:
                    pass
                rel = os.path.relpath(out_path, out_root).replace('\\', '/')
                rows.append([rel, label, session_id, frame_idx, split, f"{lap_var:.6f}", ("" if prob is None else f"{prob:.6f}"), modality, le, reye, mo, f"{cap_fps:.3f}", f"{time_sec:.3f}"])
                history.append((time_sec, ph))
                # prune history older than 1.5s
                cutoff = time_sec - 1.5
                history = [item for item in history if item[0] >= cutoff]
                saved += 1
                if saved >= max_frames:
                    break
            frame_idx += 1
        cap.release()

    # Write CSV
    csv_path = os.path.join(out_root, 'dmd_frames.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(["path", "label", "session", "frame_idx", "split", "blur_metric", "landmarks_quality", "modality", "left_eye_roi", "right_eye_roi", "mouth_roi", "fps", "time_sec"])
        def fmt_roi(box):
            if box is None:
                return ""
            x,y,w,h = box
            return f"{x},{y},{w},{h}"
        for rel, label, session_id, fidx, split, bm, prob, mod, le, reye, mo, fps_val, tsec in rows:
            writer.writerow([rel, label, session_id, fidx, split, bm, prob, mod, fmt_roi(le), fmt_roi(reye), fmt_roi(mo), fps_val, tsec])
    return csv_path, len(rows)

# /////////////////////
# DMD Helpers
# /////////////////////

def compute_phash(img: Image.Image) -> int:
    dct = None
    try:
        if cv2 is not None:
            gray = np.array(img.convert('L').resize((32, 32), PIL_RESAMPLE_BILINEAR), dtype=np.float32)
            dct = cv2.dct(gray)
    except Exception:
        dct = None
    if dct is None:
        a = np.array(img.convert('L').resize((32, 32), PIL_RESAMPLE_BILINEAR), dtype=np.float32)
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
    try:
        if cv2 is None:
            raise RuntimeError
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
    except Exception:
        g = np.array(img.convert('L'))
        mask = (g > 60) & (g < 200)
        return float(mask.mean()) if mask.size else 0.0


def _reference_five_point_112(margin: float = 0.10) -> np.ndarray:
    ref = np.array([
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ], dtype=np.float32)
    out = 112.0
    center = np.array([out / 2.0, out / 2.0], dtype=np.float32)
    scale = max(0.0, 1.0 - float(margin))
    ref_scaled = center + (ref - center) * scale
    return ref_scaled.astype(np.float32)


def align_face_5pt(img: Image.Image, pts5: np.ndarray, out_size: int = 112, margin: float = 0.10):
    try:
        if cv2 is None:
            raise RuntimeError
        pts_src = np.asarray(pts5, dtype=np.float32)
        if pts_src.shape != (5, 2):
            raise RuntimeError
        ref = _reference_five_point_112(margin=margin)
        M, _ = cv2.estimateAffinePartial2D(pts_src, ref, method=cv2.LMEDS)
        M = np.asarray(M, dtype=np.float32) if M is not None else None
        if M is None:
            raise RuntimeError
        img_np = np.array(img.convert('RGB'))
        aligned_np = cv2.warpAffine(img_np, M, (out_size, out_size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        pts_aug = np.hstack([pts_src, np.ones((5, 1), dtype=np.float32)])
        pts_t = (M @ pts_aug.T).T[:, :2]
        return Image.fromarray(aligned_np), pts_t
    except Exception:
        return safe_fit(img, (out_size, out_size)), None


def roi_boxes_from_landmarks(pts_t: Optional[np.ndarray], out_size: int = 112, eye_size: int = 64, mouth_size: int = 64):
    if pts_t is None or not isinstance(pts_t, np.ndarray) or pts_t.shape != (5, 2):
        return None, None, None
    def clamp_box(cx, cy, bw, bh, W, H):
        x = int(round(cx - bw / 2))
        y = int(round(cy - bh / 2))
        x = max(0, min(W - bw, x))
        y = max(0, min(H - bh, y))
        return (x, y, bw, bh)
    W = H = out_size
    left_eye = pts_t[0]
    right_eye = pts_t[1]
    mouth_c = (pts_t[3] + pts_t[4]) / 2.0
    le = clamp_box(left_eye[0], left_eye[1], eye_size, eye_size, W, H)
    re = clamp_box(right_eye[0], right_eye[1], eye_size, eye_size, W, H)
    mo = clamp_box(mouth_c[0], mouth_c[1], mouth_size, mouth_size, W, H)
    return le, re, mo

# /////////////////////////////
# Feature extractor
# /////////////////////////////


class EfficientNetFeatureExtractor(nn.Module):

    def __init__(self, pretrained: bool = True):
        super().__init__()
        m = torchvision.models.efficientnet_b0(
            weights=(torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
        )
        # Remove classifier entirely to avoid dropout; return pooled features
        in_feats = m.classifier[1].in_features
        m.classifier = nn.Sequential(nn.Identity())
        self.backbone = m
        self.feature_dim = in_feats

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


# /////////////////////////////
# Data loading config
# /////////////////////////////


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
    img_size: int = 112

    # Loader/Batching
    batch_size: int = 64
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
                n = len(ds)
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


if __name__ == "__main__":
    print("dsm.py slim build: import and use the library APIs (datasets, transforms, loaders, backbone).")
