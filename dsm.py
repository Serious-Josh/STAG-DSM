#DEBUG FULL DROWSINESS BUILD

#CSCI 49500 - FER DSM Project
#Basharat, Derek, Josh, Madison

import argparse
import csv
import io
import json
import math
import multiprocessing as mp
import os
import random
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, cast

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter, ImageOps
from sklearn.metrics import confusion_matrix as confusion_matrix, precision_recall_fscore_support as prfs
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import ConcatDataset, Subset, WeightedRandomSampler, random_split
import torchvision
from torchvision import transforms
from torchvision.transforms import functional as TF
from tqdm.auto import tqdm

def ensure_list_metric(x):
    try:
        return np.asarray(x).reshape(-1).tolist()
    except Exception:
        try:
            return [float(x)]
        except Exception:
            return [x]

def safe_int(x: Any) -> int:
    try:
        if x is None:
            return 0
        # unwrap simple single-element containers
        if isinstance(x, (list, tuple)) and len(x) == 1:
            x = x[0]
        try:
            return int(x)  # type: ignore[arg-type]
        except Exception:
            return int(float(x))  # type: ignore[arg-type]
    except Exception:
        return 0
try:
    from PIL.Image import Resampling
    PIL_RESAMPLE_BILINEAR: Any = Resampling.BILINEAR
except Exception:
    PIL_RESAMPLE_BILINEAR: Any = getattr(Image, 'BILINEAR', 2)

# Stable global mapping for DMD OpenLABEL action names -> integer ids.
# This can be extended at runtime if new action names are encountered.
DMD_ACTION_NAME_TO_ID: Dict[str, int] = {
    'neutral': 0,
    "eyes_opened": 1,
    "eyes_closed": 2,
    "eyes_opening": 3,
    "eyes_closing": 4,
    "blinking": 5,
    "yawn_no_hand": 6,
    "yawn_with_hand": 7,
    "occl_face": 8,
    "occl_body": 9,
    "occl_hands": 10,
}

# ---- DMD OpenLABEL type → 11-class id (canonical) ----
# 0 none, 1 eyes_opened, 2 eyes_closed, 3 eyes_opening, 4 eyes_closing,
# 5 blinking, 6 yawn_no_hand, 7 yawn_with_hand, 8 occl_face, 9 occl_body, 10 occl_hands
DMD_TYPE_TO_ID: Dict[str, int] = {
    # Eyes state
    "eyes_state/open": 1,
    "eyes_state/close": 2,
    "eyes_state/opening": 1,
    "eyes_state/closing": 2,
    "eyes_state/undefined": 0,   # seen in some files

    # Blinks
    "blinks/blinking": 5,

    # Yawning (exact strings from dataset; keep lowercase)
    "yawning/yawning with hand": 7,
    "yawning/yawning without hand": 6,

    # Occlusions (appear in some splits)
    "occlusion/face occlusion": 8,
    "occlusion/body occlusion": 9,
    "occlusion/hands occlusion": 10,
}

# Priority when multiple labels overlap in the same frame (higher wins)
DMD_PRIORITY: Dict[int, int] = {
    8: 400, 9: 400, 10: 400,        # occlusions dominate
    7: 300, 6: 300,                 # yawns
    5: 200,                         # blink
    4: 100, 3: 100, 2: 100, 1: 100, # eyes state
    0: 0,
}

# AffectNet primary-8 name mapping (default)
AFFECTNET_PRIMARY8_NAME_TO_ID: Dict[str, int] = {
    'neutral': 0,
    'happy': 1,
    'sad': 2,
    'surprise': 3,
    'fear': 4,
    'disgust': 5,
    'anger': 6,
    'contempt': 7,
}


def affectnet_clean(labels_csv: str, images_root: str, cache_dir: Optional[str]=None, seven_class: bool=False) -> Tuple[str, Dict[Any, int], List[int], Dict[int, int]]:
    cache_dir = cache_dir or os.path.join(os.getcwd(), 'outputs', 'affectnet_cache')
    os.makedirs(cache_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(labels_csv))[0]
    K = 7 if seven_class else 8
    out_csv = os.path.join(cache_dir, f'{base}_clean_{K}.csv')
    map_json = os.path.join(cache_dir, 'affectnet_map.json')
    meta_json = os.path.join(cache_dir, f'{base}_clean_{K}.meta.json')

    # Cache reuse
    try:
        src_mtime = os.path.getmtime(labels_csv)
        if os.path.isfile(out_csv) and os.path.isfile(meta_json):
            meta = json.load(open(meta_json, 'r', encoding='utf-8'))
            if float(meta.get('source_mtime', -1)) == float(src_mtime) and int(meta.get('K', 0)) == K:
                # quick histogram read
                counts: Dict[int,int] = {}
                uniq: List[int] = []
                with open(out_csv, 'r', encoding='utf-8') as f:
                    for line in f:
                        parts = line.strip().split(',')
                        if len(parts) >= 2 and parts[1].strip().isdigit():
                            lab = int(parts[1].strip())
                            counts[lab] = counts.get(lab, 0) + 1
                uniq = sorted(counts.keys())
                mapping = json.load(open(map_json, 'r', encoding='utf-8')) if os.path.isfile(map_json) else {}
                print(f'AffectNet cleaner (cache): using {out_csv}')
                print(f'AffectNet: K={K} | unique={uniq} | counts={counts}')
                return (out_csv, mapping, uniq, counts)
    except Exception:
        pass

    df = pd.read_csv(labels_csv)
    # Drop junk columns
    junk = [c for c in df.columns if c.lower().startswith('unnamed') or c.lower().startswith('rel')]
    if junk:
        df = df.drop(columns=junk)
    # Identify path/label columns
    path_cols = ['path', 'file', 'image', 'subDirectory_filePath', 'subdirectory_file_path', 'filename']
    label_cols = ['label', 'expression', 'expr']
    pcol = None
    for c in path_cols:
        if c in df.columns:
            pcol = c; break
    if pcol is None:
        # fallback: first object column
        for c in df.columns:
            if df[c].dtype == object:
                pcol = c; break
    lcol = None
    for c in label_cols:
        if c in df.columns:
            lcol = c; break
    if lcol is None:
        # fallback: any column that looks label-like
        for c in df.columns:
            if c != pcol and (str(df[c].dtype).startswith('int') or str(df[c].dtype).startswith('float') or df[c].dtype == object):
                lcol = c; break
    if pcol is None or lcol is None:
        raise ValueError(f'Cannot locate path/label columns in {labels_csv}. Columns={list(df.columns)}')

    # Normalize labels (accept strings or ints)
    mapping8 = AFFECTNET_PRIMARY8_NAME_TO_ID.copy()
    if seven_class:
        mapping7 = {k: v if v < 7 else None for k, v in mapping8.items()}
        mapping7 = {k: v for k, v in mapping7.items() if v is not None}
        # remap to contiguous 0..6 preserving order
        order = ['neutral','happy','sad','surprise','fear','disgust','anger']
        mapping7 = {name: i for i, name in enumerate(order)}
        name_to_id = mapping7
    else:
        name_to_id = mapping8

    def label_to_id(val: Any) -> Optional[int]:
        if val is None:
            return None
        if isinstance(val, str):
            key = val.strip().lower()
            if key == '':
                return None
            # Many CSVs may store integers as strings
            if key.isdigit():
                try:
                    iv = int(key)
                    if seven_class:
                        return iv if 0 <= iv <= 6 else None
                    else:
                        return iv if 0 <= iv <= 7 else None
                except Exception:
                    return None
            return name_to_id.get(key)
        try:
            iv = int(val)
            if seven_class:
                return iv if 0 <= iv <= 6 else None
            else:
                return iv if 0 <= iv <= 7 else None
        except Exception:
            return None

    images_root_abs = os.path.abspath(images_root)
    def relpath(p: str) -> str:
        p = str(p or '')
        p = p.replace('\\', '/').lstrip('/')
        try:
            if os.path.isabs(p):
                p_rel = os.path.relpath(p, images_root_abs)
            else:
                p_rel = p
        except Exception:
            p_rel = p
        return p_rel.replace('\\', '/').strip()

    rows: List[Tuple[str,int]] = []
    counts: Dict[int,int] = {}
    for _, r in df.iterrows():
        lab = label_to_id(r[lcol])
        if lab is None:
            continue
        relp = relpath(r[pcol])
        if not relp:
            continue
        rows.append((relp, int(lab)))
        counts[int(lab)] = counts.get(int(lab), 0) + 1

    uniq = sorted(set(l for _, l in rows))
    # assert range
    if any(x < 0 or x >= K for x in uniq):
        bad = [x for x in uniq if (x < 0 or x >= K)]
        raise ValueError(f'AffectNet clean produced out-of-range labels for K={K}: {bad}')

    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        for rel, lab in rows:
            w.writerow([rel, int(lab)])

    # Save mapping JSON
    mapping_json: Dict[Any, int] = name_to_id.copy()
    with open(map_json, 'w', encoding='utf-8') as f:
        json.dump(mapping_json, f, indent=2)

    # Save meta for cache
    try:
        json.dump({'source': os.path.abspath(labels_csv), 'source_mtime': os.path.getmtime(labels_csv), 'K': K}, open(meta_json, 'w', encoding='utf-8'), indent=2)
    except Exception:
        pass

    print(f'AffectNet cleaner: wrote {out_csv} K={K} unique={uniq} counts={counts}')
    return (out_csv, mapping_json, uniq, counts)

def seed_everything(seed: int=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
 

# Pick a device: 'auto' prefers CUDA, then MPS (Apple), else CPU.
def torch_device(prefer: Optional[str]='auto') -> torch.device:
    # Select device: explicit 'cuda'/'mps' or 'auto' ↁEcuda ↁEmps ↁEcpu
    try:
        pref = (prefer or 'auto').lower()
    except Exception:
        pref = 'auto'

    def mps_available() -> bool:
        try:
            mps = getattr(torch.backends, 'mps', None)
            return bool(mps) and bool(getattr(mps, 'is_available', lambda: False)())
        except Exception:
            return False

    # DirectML path removed
    if pref == 'cuda' and torch.cuda.is_available():
        return torch.device('cuda')
    if pref == 'mps' and mps_available():
        return torch.device('mps')
    # no DirectML branch
    if pref == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda')
        if mps_available():
            return torch.device('mps')
    return torch.device('cpu')

def get_num_workers() -> int:
    try:
        return max(2, mp.cpu_count() // 2)
    except Exception:
        return 2

def progress(iterable, total: Optional[int]=None, unit: str='it', desc: Optional[str]=None, leave: bool=False):
    if tqdm is not None:
        try:
            return tqdm(iterable, total=total, unit=unit, desc=desc, leave=leave)
        except Exception:
            return iterable
    return iterable

def load_class_map(path: Optional[str]) -> Optional[dict]:
    if not path:
        return None
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

# CSV-backed dataset -> (tensor,label) pairs
class CsvImageDataset(Dataset):
    # Simple CSV dataset that yields (RGB tensor, label)

    def __init__(self, csv_file: str, root_dir: Optional[str]=None, transform=None, class_map: Optional[dict]=None, has_header: bool=False, sep: str=','):
        super().__init__()
        self.root_dir = root_dir
        self.transform = transform
        self.samples: List[Tuple[str, int]] = []
        self.class_map = class_map
        with open(csv_file, 'r', encoding='utf-8') as f:
            lines = f.read().strip().splitlines()
        if has_header and lines:
            lines = lines[1:]
        for line in lines:
            if not line:
                continue
            parts = line.split(sep)
            if len(parts) < 2:
                continue
            path_raw, label_raw = (parts[0].strip(), parts[1].strip())
            if self.root_dir and (not os.path.isabs(path_raw)):
                path = os.path.join(self.root_dir, path_raw)
            else:
                path = path_raw
            if self.class_map and (not label_raw.isdigit()):
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
            im = im.convert('RGB')
            if self.transform is not None:
                im = self.transform(im)
        if isinstance(im, Image.Image):
            im = TF.to_tensor(im)
        return (im, label)

class YuNetDetector:
    # Adapter over OpenCV YuNet (FaceDetectorYN).
    # detect(img_bgr) returns a list of detections sorted by score.

    def __init__(self, model_path: str='face_detection_yunet.onnx', input_size: Tuple[int, int]=(320, 320), conf_threshold: float=0.6, nms_threshold: float=0.3, top_k: int=5000):
        if cv2 is None:
            raise RuntimeError('OpenCV is required for YuNetDetector')
        if not hasattr(cv2, 'FaceDetectorYN_create'):
            raise RuntimeError('This OpenCV build lacks FaceDetectorYN. Install OpenCV >= 4.6 (opencv-python>=4.6.0).')
        mp = model_path
        if not os.path.isabs(mp):
            cwd_try = os.path.abspath(mp)
            file_try = os.path.join(os.path.dirname(os.path.abspath(__file__)), mp)
            if os.path.isfile(cwd_try):
                mp = cwd_try
            elif os.path.isfile(file_try):
                mp = file_try
        self.model_path = mp
        if not os.path.isfile(self.model_path):
            raise FileNotFoundError(f'YuNet ONNX not found: {self.model_path}')
        self.input_size = (int(input_size[0]), int(input_size[1]))
        self.conf_threshold = float(conf_threshold)
        self.nms_threshold = float(nms_threshold)
        self.top_k = int(top_k)
        self.detector = getattr(cv2, 'FaceDetectorYN_create')(self.model_path, '', self.input_size, self.conf_threshold, self.nms_threshold, self.top_k)

    def set_input_size(self, w: int, h: int):
        if (w, h) != tuple(self.input_size):
            self.detector.setInputSize((int(w), int(h)))
            self.input_size = (int(w), int(h))

    def detect(self, img_bgr: np.ndarray) -> List[dict]:
        if img_bgr is None:
            return []
        h, w = img_bgr.shape[:2]
        if h <= 1 or w <= 1:
            return []
        self.set_input_size(w, h)
        faces, _ = self.detector.detect(img_bgr)
        if faces is None or len(faces) == 0:
            return []
        results: List[dict] = []
        for f in faces:
            f = f.flatten().tolist()
            x, y, bw, bh = (f[0], f[1], f[2], f[3])
            kps = [(f[4], f[5]), (f[6], f[7]), (f[8], f[9]), (f[10], f[11]), (f[12], f[13])]
            score = float(f[14])
            results.append({'bbox': (float(x), float(y), float(bw), float(bh)), 'kps': [(float(px), float(py)) for px, py in kps], 'score': score})
        results.sort(key=lambda d: d.get('score', 0.0), reverse=True)
        return results

def is_image_corrupted(path: str) -> bool:
    # Best-effort integrity check via Pillow verify() + full load
    try:
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            im.convert('RGB').load()
        return False
    except Exception:
        return True

def is_occluded_or_too_dark_bright(img: Image.Image, dark_thresh: float=0.15, bright_thresh: float=0.15) -> bool:
    # Drop images with extreme dark/bright proportions
    im = np.array(img.convert('L'))
    total = im.size
    too_dark = (im < 10).sum() / total
    too_bright = (im > 245).sum() / total
    return too_dark > dark_thresh or too_bright > bright_thresh

def is_blurry(img: Image.Image, edge_thresh: float=4.0) -> bool:
    # Sobel variance threshold; lower implies blurrier images
    t = TF.to_tensor(img.convert('L')).unsqueeze(0)
    sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32, device=t.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32, device=t.device).view(1, 1, 3, 3)
    with torch.no_grad():
        gx = F.conv2d(t, sobel_x, padding=1)
        gy = F.conv2d(t, sobel_y, padding=1)
        mag = (gx.pow(2) + gy.pow(2)).sqrt()
        score = mag.var().item() * 1000.0
    return score < edge_thresh

def blur_metric(img: Image.Image) -> float:
    # Return the Sobel-based sharpness score used for QA
    t = TF.to_tensor(img.convert('L')).unsqueeze(0)
    sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32, device=t.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32, device=t.device).view(1, 1, 3, 3)
    with torch.no_grad():
        gx = F.conv2d(t, sobel_x, padding=1)
        gy = F.conv2d(t, sobel_y, padding=1)
        mag = (gx.pow(2) + gy.pow(2)).sqrt()
        score = mag.var().item() * 1000.0
    return float(score)

def laplacian_variance(img: Image.Image) -> float:
    # Classic OpenCV blur metric (variance of Laplacian)
    if cv2 is None:
        return blur_metric(img)
    gray = np.array(img.convert('L'), dtype=np.uint8)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(lap.var())

def compute_phash(img: Image.Image) -> int:
    # 64-bit perceptual hash for near-duplicate detection
    if cv2 is not None:
        try:
            gray = np.array(img.convert('L').resize((32, 32), PIL_RESAMPLE_BILINEAR), dtype=np.float32)
            dct = cv2.dct(gray)
        except Exception:
            dct = None
    else:
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
        h = h << 1 | int(b)
    return int(h)

def hamming_distance(a: int, b: int) -> int:
    # Bitwise Hamming distance between two integer hashes
    return int((a ^ b).bit_count())

def skin_coverage_fraction(img: Image.Image) -> float:
    # YCrCb + HSV heuristics to approximate skin-pixel fraction
    if cv2 is None:
        g = np.array(img.convert('L'))
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

def reference_five_point_112(margin: float=0.1) -> np.ndarray:
    ref = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)
    out = 112.0
    center = np.array([out / 2.0, out / 2.0], dtype=np.float32)
    scale = max(0.0, 1.0 - float(margin))
    ref_scaled = center + (ref - center) * scale
    return ref_scaled.astype(np.float32)

def align_face_5pt(img: Image.Image, pts5: np.ndarray, out_size: int=112, margin: float=0.1):
    if cv2 is None:
        try:
            pil = ImageOps.fit(img, (out_size, out_size), method=PIL_RESAMPLE_BILINEAR, centering=(0.5, 0.5))
        except Exception:
            pil = img.resize((out_size, out_size))
        return (pil, None)
    pts_src = np.asarray(pts5, dtype=np.float32)
    if pts_src.shape != (5, 2):
        try:
            pil = ImageOps.fit(img, (out_size, out_size), method=PIL_RESAMPLE_BILINEAR, centering=(0.5, 0.5))
        except Exception:
            pil = img.resize((out_size, out_size))
        return (pil, None)
    ref = reference_five_point_112(margin=margin)
    M, inliers = cv2.estimateAffinePartial2D(pts_src, ref, method=cv2.LMEDS)
    M = np.asarray(M, dtype=np.float32) if M is not None else None
    if M is None:
        try:
            pil = ImageOps.fit(img, (out_size, out_size), method=PIL_RESAMPLE_BILINEAR, centering=(0.5, 0.5))
        except Exception:
            pil = img.resize((out_size, out_size))
        return (pil, None)
    img_np = np.array(img.convert('RGB'))
    aligned_np = cv2.warpAffine(img_np, M, (out_size, out_size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    pts_aug = np.hstack([pts_src, np.ones((5, 1), dtype=np.float32)])
    pts_t = (M @ pts_aug.T).T[:, :2]
    return (Image.fromarray(aligned_np), pts_t)

def roi_boxes_from_landmarks(pts_t: Optional[np.ndarray], out_size: int=112, eye_size: int=64, mouth_size: int=64):

    def clamp_box(cx, cy, bw, bh, W, H):
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
    le = clamp_box(left_eye[0], left_eye[1], eye_size, eye_size, W, H)
    re = clamp_box(right_eye[0], right_eye[1], eye_size, eye_size, W, H)
    mo = clamp_box(mouth_c[0], mouth_c[1], mouth_size, mouth_size, W, H)
    return (le, re, mo)

GENFER_CLEAN_CACHE: dict = {}

class GenericFERDataset(CsvImageDataset):
    # Cleans samples and optionally applies YuNet-based alignment per image.

    def __init__(self, csv_file: str, root_dir: Optional[str]=None, transform=None, class_map: Optional[dict]=None, has_header: bool=False, sep: str=',', num_classes: Optional[int]=None, enable_alignment: bool=True, crop_tight: bool=False, margin_range: Tuple[float, float]=(0.08, 0.12), target_size: Optional[int]=None):
        super().__init__(csv_file, root_dir, transform, class_map, has_header, sep)
        self.enable_alignment: bool = bool(enable_alignment)
        self.detector: Optional[YuNetDetector] = None
        if self.enable_alignment:
            try:
                self.detector = YuNetDetector(model_path=os.environ.get('YUNET_ONNX', 'face_detection_yunet.onnx'), input_size=(320, 320), conf_threshold=0.6, nms_threshold=0.3, top_k=5000)
            except Exception:
                self.detector = None
        self.crop_tight = bool(crop_tight)
        self.margin_range = margin_range
        self.target_size = target_size
        # Robust: cache cleaned samples per (csv, root, num_classes) to avoid duplicate work/logs
        key = (os.path.abspath(str(csv_file)), os.path.abspath(str(root_dir)) if root_dir else None, int(num_classes) if num_classes is not None else None)
        if key in GENFER_CLEAN_CACHE:
            self.samples = GENFER_CLEAN_CACHE[key]['samples']
            # Suppress duplicate print; cleaning already logged on first creation
            return
        cleaned: List[Tuple[str, int]] = []
        dropped_corrupt = dropped_occluded = dropped_blurry = 0
        for path, label in self.samples:
            if num_classes is not None and (label < 0 or label >= num_classes):
                continue
            if is_image_corrupted(path):
                dropped_corrupt += 1
                continue
            try:
                with Image.open(path) as im:
                    im = im.convert('RGB')
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
        # Save to cache and print once
        GENFER_CLEAN_CACHE[key] = {'samples': self.samples, 'stats': (dropped_corrupt, dropped_occluded, dropped_blurry)}
        if dropped_corrupt or dropped_occluded or dropped_blurry:
            print(f'[GenericFERDataset] Cleaned: {len(self.samples)} kept | corrupt {dropped_corrupt}, occluded {dropped_occluded}, blurry {dropped_blurry}')

    def __getstate__(self):
        # Make dataset picklable for DataLoader workers on Windows by dropping non-picklable detector
        state = self.__dict__.copy()
        state['detector'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Detector will be lazily re-created in _align_crop if needed

    def align_crop(self, img: Image.Image) -> Image.Image:
        # 1) If requested, try a bbox-based tight crop with a small random margin
        # Lazily (re)create detector in worker if alignment was requested
        if self.detector is None and self.enable_alignment:
            try:
                self.detector = YuNetDetector(model_path=os.environ.get('YUNET_ONNX', 'face_detection_yunet.onnx'), input_size=(320, 320), conf_threshold=0.6, nms_threshold=0.3, top_k=5000)
            except Exception:
                self.detector = None
        if self.crop_tight and self.detector is not None:
            try:
                img_bgr = cv2.cvtColor(np.array(img.convert('RGB')), cv2.COLOR_RGB2BGR)
                dets = self.detector.detect(img_bgr)
                if dets:
                    best = dets[0]
                    x, y, w, h = best['bbox']
                    x1, y1, x2, y2 = (x, y, x + w, y + h)
                    w = max(1.0, float(w))
                    h = max(1.0, float(h))
                    m = float(random.uniform(self.margin_range[0], self.margin_range[1]))
                    dw = m * w
                    dh = m * h
                    nx1 = max(0.0, x1 - dw)
                    ny1 = max(0.0, y1 - dh)
                    nx2 = min(float(img.width), x2 + dw)
                    ny2 = min(float(img.height), y2 + dh)
                    cropped = img.crop((nx1, ny1, nx2, ny2))
                    if isinstance(self.target_size, int) and self.target_size > 0:
                        try:
                            cropped = ImageOps.fit(cropped, (self.target_size, self.target_size), method=PIL_RESAMPLE_BILINEAR, centering=(0.5, 0.5))
                        except Exception:
                            cropped = cropped.resize((self.target_size, self.target_size))
                    return cropped
            except Exception:
                pass
        if self.detector is not None:
            # 2) Otherwise, if we have landmarks, do 5-point alignment to 112x112
            try:
                img_bgr = cv2.cvtColor(np.array(img.convert('RGB')), cv2.COLOR_RGB2BGR)
                dets = self.detector.detect(img_bgr)
                if dets:
                    best = dets[0]
                    pts5 = np.array(best['kps'], dtype=np.float32)
                    out_sz = int(self.target_size) if isinstance(self.target_size, int) and self.target_size > 0 else 112
                    aligned, _ = align_face_5pt(img, pts5, out_size=out_sz, margin=0.1)
                    return aligned
            except Exception:
                pass
        # 3) Final fallback: center-fit to target size
        out_sz = int(self.target_size) if isinstance(self.target_size, int) and self.target_size > 0 else 112
        try:
            pil = ImageOps.fit(img, (out_sz, out_sz), method=PIL_RESAMPLE_BILINEAR, centering=(0.5, 0.5))
        except Exception:
            pil = img.resize((out_sz, out_sz))
        return pil

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        with Image.open(path) as im:
            im = im.convert('RGB')
            im = self.align_crop(im)
            if self.transform is not None:
                im = self.transform(im)
        if isinstance(im, Image.Image):
            im = TF.to_tensor(im)
        return (im, label)

class RandomTightCrop:

    def __init__(self, max_pixels: int=8, output_size: Optional[int]=None):
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

    def __init__(self, p: float=0.2, k_range: Tuple[int, int]=(3, 7)):
        self.p = p
        self.k_range = k_range

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() >= self.p:
            return img
        kmin, kmax = self.k_range
        k = random.randint(max(3, kmin), max(3, kmax))
        if k % 2 == 0:
            k += 1
        orientation = random.choice(['h', 'v'])
        size = (k, k)
        kernel = [0.0] * (k * k)
        if orientation == 'h':
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

    def __init__(self, p: float=0.2, quality: Tuple[int, int]=(40, 80)):
        self.p = p
        self.quality = quality

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() >= self.p:
            return img
        q = int(random.randint(self.quality[0], self.quality[1]))
        buf = io.BytesIO()
        try:
            img.save(buf, format='JPEG', quality=q)
            buf.seek(0)
            return Image.open(buf).convert('RGB')
        except Exception:
            return img

class RandomGamma:

    def __init__(self, p: float=0.5, gamma_range: Tuple[float, float]=(0.9, 1.1)):
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

    def __init__(self, p: float=0.1):
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() >= self.p:
            return img
        w, h = img.size
        draw = ImageDraw.Draw(img)
        kind = random.choice(['sunglasses', 'mask'])
        if kind == 'sunglasses':
            y_center = int(0.33 * h)
            band_h = int(random.uniform(0.15, 0.25) * h)
            y1 = max(0, y_center - band_h // 2)
            y2 = min(h, y1 + band_h)
            draw.rectangle([0, y1, w, y2], fill=(0, 0, 0))
        else:
            y1 = int(0.55 * h)
            y2 = min(h, int(y1 + random.uniform(0.25, 0.4) * h))
            draw.rectangle([0, y1, w, y2], fill=(0, 0, 0))
        return img

class AddGaussianNoise:

    def __init__(self, p: float=0.1, sigma_range: Tuple[float, float]=(3.0, 8.0)):
        self.p = p
        self.sigma_range = sigma_range

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        if random.random() >= self.p:
            return t
        sigma = random.uniform(self.sigma_range[0], self.sigma_range[1]) / 255.0
        noise = torch.randn_like(t) * sigma
        t = t + noise
        return t.clamp(0.0, 1.0)

def make_transforms(img_size: int=112):
    train_tf = transforms.Compose([transforms.Resize((img_size, img_size), antialias=True), transforms.RandomHorizontalFlip(p=0.5), transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1), fill=0), RandomTightCrop(max_pixels=8, output_size=img_size), transforms.ColorJitter(brightness=0.2, contrast=0.2), RandomGamma(p=0.5, gamma_range=(0.9, 1.1)), transforms.RandomGrayscale(p=0.1), MotionBlur(p=0.2, k_range=(3, 7)), RandomJPEG(p=0.2, quality=(40, 80)), RandomOcclusionBand(p=0.1), transforms.ToTensor(), AddGaussianNoise(p=0.1, sigma_range=(3.0, 8.0)), transforms.RandomErasing(p=0.25, scale=(0.08, 0.12), value=0), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    val_tf = transforms.Compose([transforms.Resize((img_size, img_size), antialias=True), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    return (train_tf, val_tf)

def gen_csv_from_folders(root_dir: str, out_csv: str, exts: set, class_map_in: Optional[dict], save_map_path: Optional[str]) -> Optional[str]:
    root_dir = os.path.abspath(root_dir)
    subdirs = [d for d in sorted(os.listdir(root_dir)) if os.path.isdir(os.path.join(root_dir, d))]
    if not subdirs:
        raise ValueError(f'No class subfolders found under: {root_dir}')
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
                continue
            label = class_map[cname]
            for r, _, files in os.walk(cdir):
                for fn in files:
                    ext = os.path.splitext(fn)[1].lower()
                    if ext in exts:
                        rel_path = os.path.relpath(os.path.join(r, fn), root_dir)
                        fcsv.write(f'{rel_path},{label}\n')
                        count += 1
    print(f'[CSV GEN] {count} samples written from {len(subdirs)} classes')
    return wrote_map_path

def gen_csv_from_rafdb(root_dir: str, out_csv: str, split: str='train', image_subdir: Optional[str]=None):
    root_dir = os.path.abspath(root_dir)
    anno_dir = os.path.join(root_dir, 'EmoLabel')
    if split == 'train':
        list_file = os.path.join(anno_dir, 'list_train.txt')
    else:
        list_file = os.path.join(anno_dir, 'list_test.txt')
    if not os.path.isfile(list_file):
        raise FileNotFoundError(f'RAF-DB annotation file not found: {list_file}')
    if image_subdir is None:
        candidates = ['Image/aligned', 'Image/original', 'image/aligned', 'image/original', 'Aligned', 'Original']
        img_root = None
        for c in candidates:
            p = os.path.join(root_dir, c)
            if os.path.isdir(p):
                img_root = p
                break
        if img_root is None:
            raise FileNotFoundError('Could not locate RAF-DB images. Provide --raf_image_subdir.')
    else:
        img_root = os.path.join(root_dir, image_subdir)
        if not os.path.isdir(img_root):
            raise FileNotFoundError(f'RAF-DB images directory not found: {img_root}')
    rows = []
    with open(list_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            img_name, label_str = (parts[0], parts[1])
            try:
                label = int(label_str)
            except Exception:
                continue
            if label >= 1:
                label -= 1
            img_path = os.path.join(img_root, img_name)
            rel_path = os.path.relpath(img_path, img_root)
            rows.append((rel_path.replace('\\', '/'), label))
    if not rows:
        raise ValueError('No RAF-DB samples parsed. Check paths and annotation format.')
    with open(out_csv, 'w', encoding='utf-8') as fcsv:
        for p, l in rows:
            fcsv.write(f'{p},{l}\n')
    print(f"[RAF-DB CSV GEN] {len(rows)} samples from split='{split}' using base '{os.path.relpath(img_root, root_dir)}'")

def build_weighted_sampler(labels: List[int]) -> WeightedRandomSampler:
    counts = Counter(labels)
    num_samples = len(labels)
    weights = [num_samples / (counts[l] * len(counts)) for l in labels]
    return WeightedRandomSampler(weights, num_samples=num_samples, replacement=True)

class EfficientNetFeatureExtractor(nn.Module):

    def __init__(self, variant: str='b0', pretrained: bool=True):
        super().__init__()
        variant = (variant or 'b0').lower()
        if variant == 'b0':
            m = torchvision.models.efficientnet_b0(weights=torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
        elif variant == 'b1':
            m = torchvision.models.efficientnet_b1(weights=torchvision.models.EfficientNet_B1_Weights.IMAGENET1K_V1 if pretrained else None)
        else:
            raise ValueError('Supported variants: b0, b1')
        # Keep classifier Identity but we will call features/avgpool directly
        # Type-only narrowing for lin head's in_features
        in_feats = cast(int, m.classifier[1].in_features)
        m.classifier = nn.Sequential(nn.Identity())
        self.backbone = m
        self.feature_dim = in_feats

        # ROI branch: project last conv map to 512 then pool per ROI
        self.roi_proj = nn.Conv2d(in_feats, 512, kernel_size=1, bias=False)
        self.roi_proj_bn = nn.BatchNorm2d(512)
        # Shared lightweight MLP for ROI vectors (512 -> 128)
        # Use LayerNorm to be robust to small batch sizes (e.g., batch size 1)
        self.roi_mlp = nn.Sequential(
            nn.LayerNorm(512),
            nn.Dropout(p=0.2),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
        )
        # Fusion: [global(in_feats) || eyes(128) || mouth(128)] -> in_feats
        self.fusion = nn.Linear(in_feats + 256, in_feats)

        # Precompute default canonical ROI boxes using reference landmarks on 112x112
        ref_pts = reference_five_point_112(margin=0.1)
        leye, reye, mouth = roi_boxes_from_landmarks(ref_pts, out_size=112, eye_size=64, mouth_size=64)
        # If detector is not available, roi_boxes_from_landmarks may return (None, None, None)
        # Fall back to simple centered boxes
        if leye is None or reye is None or mouth is None:
            def center_box(cx, cy, bw, bh, W=112, H=112):
                x = int(round(cx - bw / 2)); y = int(round(cy - bh / 2))
                x = max(0, min(W - bw, x)); y = max(0, min(H - bh, y))
                return (x, y, bw, bh)
            leye = center_box(112 * 0.35, 112 * 0.46, 64, 64)
            reye = center_box(112 * 0.65, 112 * 0.46, 64, 64)
            mouth = center_box(112 * 0.50, 112 * 0.82, 64, 64)
        self.register_buffer('roi_leye', torch.tensor(leye, dtype=torch.float32))
        self.register_buffer('roi_reye', torch.tensor(reye, dtype=torch.float32))
        self.register_buffer('roi_mouth', torch.tensor(mouth, dtype=torch.float32))

    def pool_roi(self, fmap: torch.Tensor, box_112: torch.Tensor) -> torch.Tensor:
        # fmap: [N, C, Hf, Wf] from last conv; box_112: [4] (x,y,w,h) in 112x112 space
        N, C, Hf, Wf = fmap.shape
        # map box from 112x112 to fmap coords
        scale_x = Wf / 112.0
        scale_y = Hf / 112.0
        x, y, w, h = box_112
        x1_t = torch.clamp(torch.round(x * scale_x).long(), 0, Wf - 1)
        y1_t = torch.clamp(torch.round(y * scale_y).long(), 0, Hf - 1)
        x2_t = torch.clamp(torch.round((x + w) * scale_x).long(), 0, Wf)
        y2_t = torch.clamp(torch.round((y + h) * scale_y).long(), 0, Hf)
        # ensure at least 1x1
        x2_t = torch.maximum(x2_t, x1_t + 1)
        y2_t = torch.maximum(y2_t, y1_t + 1)
        # convert to Python ints for safe slicing across PyTorch versions
        x1 = int(x1_t.item()); x2 = int(x2_t.item())
        y1 = int(y1_t.item()); y2 = int(y2_t.item())
        # average pool over the rectangular slice
        # compute mean over H and W for each sample/channel
        roi = fmap[:, :, y1:y2, x1:x2]
        roi_mean = roi.mean(dim=(2, 3))  # [N, C]
        return roi_mean

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Accept [B, C, H, W] or [B, T, C, H, W]; default img size 112
        has_time = (x.dim() == 5)
        if has_time:
            B, T, C, H, W = x.shape
            x = x.reshape(B * T, C, H, W)
        else:
            B, T = x.shape[0], 1
        # Conv features
        feats_map = self.backbone.features(x)  # [N, C=feature_dim, Hf, Wf]
        # Global pooled vector
        g = self.backbone.avgpool(feats_map)
        g = torch.flatten(g, 1)  # [N, in_feats]
        # ROI features
        z = self.roi_proj_bn(self.roi_proj(feats_map))
        z = F.relu(z, inplace=True)
        # eyes: average of left/right eye pooled vectors (C=512)
        leye = self.pool_roi(z, cast(torch.Tensor, self.roi_leye))
        reye = self.pool_roi(z, cast(torch.Tensor, self.roi_reye))
        eyes_512 = 0.5 * (leye + reye)
        mouth_512 = self.pool_roi(z, cast(torch.Tensor, self.roi_mouth))
        # ROI MLP to 128 dims each
        eyes_128 = self.roi_mlp(eyes_512)
        mouth_128 = self.roi_mlp(mouth_512)
        # Fuse
        fused = torch.cat([g, eyes_128, mouth_128], dim=1)
        fused = self.fusion(fused)
        # Temporal pooling if time dimension present
        if T > 1:
            fused = fused.view(B, T, -1)
            # keep last up to 8 frames
            if fused.size(1) > 8:
                fused = fused[:, -8:, :]
            fused = fused.mean(dim=1)
        return fused

# Softmax head removed: ArcFace is the only classification head.

class ArcFaceHead(nn.Module):

    def __init__(self, in_dim: int, num_classes: int, m: float=0.5, s: float=64.0, eps: float=1e-07):
        super().__init__()
        self.W = nn.Parameter(torch.randn(num_classes, in_dim))
        nn.init.xavier_uniform_(self.W)
        self.m = float(m)
        self.s = float(s)
        self.eps = float(eps)

    def forward(self, x: torch.Tensor, labels: Optional[torch.Tensor]=None) -> torch.Tensor:
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

@dataclass
class TrainConfig:
    affectnet_csv: Optional[str] = None
    affectnet_root: Optional[str] = None
    rafdb_csv: Optional[str] = None
    rafdb_root: Optional[str] = None
    dmd_csv: Optional[str] = None
    dmd_root: Optional[str] = None
    class_map_json: Optional[str] = None
    num_classes: int = 7
    eff_variant: str = 'b0'
    img_size: int = 112
    batch_size: int = 64
    lr: float = 0.0003
    weight_decay: float = 0.0001
    epochs_pretrain: int = 5
    epochs_main: int = 15
    seed: int = 42
    out_dir: str = './outputs'
    amp: bool = True
    balance_sampling: bool = True

def build_loaders(cfg: TrainConfig, device: Optional[torch.device]=None):
    train_tf, val_tf = make_transforms(cfg.img_size)
    # Auto-clean AffectNet CSVs on load if necessary
    def maybe_prepare_affectnet() -> None:
        try:
            seven = bool(int(os.environ.get('AFFECTNET_7CLASS', '0')))
        except Exception:
            seven = False
        cache_dir = os.environ.get('AFFECTNET_CACHE_DIR', os.path.join(os.getcwd(), 'outputs', 'affectnet_cache'))
        # Determine labels CSV to clean
        labels_csv = None
        if cfg.affectnet_csv and os.path.isfile(cfg.affectnet_csv):
            # Peek to decide if already cleaned (two columns, no header)
            try:
                df = pd.read_csv(cfg.affectnet_csv)
                if df.shape[1] <= 2 and {'label'}.issubset(set(map(str.lower, df.columns.tolist()))):
                    # Might still be raw; prefer explicit labels_csv flag instead
                    pass
                # If more than 2 columns, very likely raw labels.csv
                if df.shape[1] > 2:
                    labels_csv = cfg.affectnet_csv
            except Exception:
                # If can't parse, assume it's already cleaned two-column
                labels_csv = None
        # If not set, try discover labels.csv under affectnet_root
        if labels_csv is None and cfg.affectnet_root:
            cand = os.path.join(cfg.affectnet_root, 'labels.csv')
            if os.path.isfile(cand):
                labels_csv = cand
        if labels_csv is None:
            # Also check CWD for labels.csv
            cand2 = os.path.join(os.getcwd(), 'labels.csv')
            if os.path.isfile(cand2):
                labels_csv = cand2
        if labels_csv:
            images_root = cfg.affectnet_root or os.path.join(os.getcwd(), 'datasets', 'affectNet')
            cleaned_csv, mapping, uniq, counts = affectnet_clean(labels_csv, images_root, cache_dir=cache_dir, seven_class=seven)
            cfg.affectnet_csv = cleaned_csv
            # K consistent with mapping length
            K = 7 if seven else 8
            cfg.num_classes = K
            print(f'AffectNet: K={K} | unique={uniq} | counts={counts}')
    maybe_prepare_affectnet()
    class_map = load_class_map(cfg.class_map_json)
    # Build separate train/val datasets to avoid mutating shared transforms
    pretrain_datasets_train: List[Dataset] = []
    pretrain_datasets_val: List[Dataset] = []
    if cfg.affectnet_csv:
        pretrain_datasets_train.append(GenericFERDataset(cfg.affectnet_csv, cfg.affectnet_root, transform=train_tf, class_map=class_map, num_classes=cfg.num_classes, enable_alignment=True, crop_tight=True, margin_range=(0.08, 0.12), target_size=cfg.img_size))
        pretrain_datasets_val.append(GenericFERDataset(cfg.affectnet_csv, cfg.affectnet_root, transform=val_tf, class_map=class_map, num_classes=cfg.num_classes, enable_alignment=True, crop_tight=True, margin_range=(0.08, 0.12), target_size=cfg.img_size))
    if cfg.rafdb_csv:
        pretrain_datasets_train.append(GenericFERDataset(cfg.rafdb_csv, cfg.rafdb_root, transform=train_tf, class_map=class_map, num_classes=cfg.num_classes, enable_alignment=True, crop_tight=True, margin_range=(0.08, 0.12), target_size=cfg.img_size))
        pretrain_datasets_val.append(GenericFERDataset(cfg.rafdb_csv, cfg.rafdb_root, transform=val_tf, class_map=class_map, num_classes=cfg.num_classes, enable_alignment=True, crop_tight=True, margin_range=(0.08, 0.12), target_size=cfg.img_size))
    if len(pretrain_datasets_train) == 0:
        raise ValueError('No pretraining datasets provided. Provide AffectNet and/or RAF-DB CSVs.')
    pretrain_train = ConcatDataset(pretrain_datasets_train)
    pretrain_val_base = ConcatDataset(pretrain_datasets_val)
    total = len(pretrain_train)
    if total <= 1:
        train_size, val_size = total, 0
    else:
        desired_val = max(1000, int(0.02 * total))
        val_size = min(total - 1, max(1, desired_val))
        train_size = total - val_size
    pretrain_train_ds, pretrain_val_indices = random_split(pretrain_train, [train_size, val_size], generator=torch.Generator().manual_seed(123))
    # Ensure validation uses the val transforms by indexing into the val-base dataset
    pretrain_val_ds = Subset(pretrain_val_base, cast(Any, pretrain_val_indices).indices if hasattr(pretrain_val_indices, 'indices') else [])
    dmd_train_loader = None
    dmd_val_loader = None
    dmd_train_ds = None
    dmd_val_ds = None
    if cfg.dmd_csv:
        dname = os.path.basename(str(cfg.dmd_csv)).lower()
        dmd_has_header = ('dmd_frames' in dname)
        dmd_train_full = CsvImageDataset(cfg.dmd_csv, cfg.dmd_root, transform=train_tf, class_map=class_map, has_header=dmd_has_header)
        dmd_val_full = CsvImageDataset(cfg.dmd_csv, cfg.dmd_root, transform=val_tf, class_map=class_map, has_header=dmd_has_header)
        total_dmd = len(dmd_train_full)
        if total_dmd <= 1:
            train_dmd, val_dmd = total_dmd, 0
        else:
            desired = max(200, int(0.1 * total_dmd))
            val_dmd = min(total_dmd - 1, max(1, desired))
            train_dmd = total_dmd - val_dmd
        dmd_train_ds, dmd_val_indices = random_split(dmd_train_full, [train_dmd, val_dmd], generator=torch.Generator().manual_seed(456))
        dmd_val_ds = Subset(dmd_val_full, cast(Any, dmd_val_indices).indices if hasattr(dmd_val_indices, 'indices') else [])

    def labels_from_dataset(ds: Any) -> List[int]:
        lbls = []
        if isinstance(ds, Subset):
            for i in ds.indices:
                base = ds.dataset
                if isinstance(base, ConcatDataset):
                    idx = i
                    for d in base.datasets:
                        n = len(cast(Any, d))
                        if idx < n:
                            if hasattr(d, 'samples'):
                                try:
                                    lab = cast(Any, d).samples[idx][1]
                                    lbls.append(lab)
                                except Exception:
                                    try:
                                        x, lab = d[idx]
                                        lbls.append(lab)
                                    except Exception:
                                        pass
                            else:
                                try:
                                    x, lab = d[idx]
                                    lbls.append(lab)
                                except Exception:
                                    pass
                            break
                        idx -= n
                else:
                    if hasattr(base, 'samples'):
                        try:
                            lab = cast(Any, base).samples[i][1]
                            lbls.append(lab)
                        except Exception:
                            try:
                                x, lab = base[i]
                                lbls.append(lab)
                            except Exception:
                                pass
                    else:
                        try:
                            x, lab = base[i]
                            lbls.append(lab)
                        except Exception:
                            pass
        else:
            try:
                n = len(ds)
            except Exception:
                return lbls
            for i in range(n):
                try:
                    if hasattr(ds, 'samples'):
                        lab = cast(Any, ds).samples[i][1]
                    else:
                        x, lab = ds[i]
                    lbls.append(lab)
                except Exception:
                    break
        return lbls
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
    num_workers = get_num_workers()
    try:
        dev = device if device is not None else torch_device()
        pin_mem = isinstance(dev, torch.device) and dev.type == 'cuda'
    except Exception:
        pin_mem = False
    # Sanity: verify labels fall into [0, num_classes)
    try:
        lbls_check = labels_from_dataset(pretrain_train_ds)
        uniq = sorted(set(int(x) for x in lbls_check))
        if any((x < 0 or x >= int(cfg.num_classes)) for x in uniq):
            raise ValueError(f"AffectNet/Pretrain labels out of range for num_classes={cfg.num_classes}. Unique={uniq}")
        cnt = Counter(lbls_check)
        print(f"AffectNet: K={int(cfg.num_classes)} | unique={uniq} | counts={dict(cnt)}")
    except Exception as e:
        print(f"Label-space check warning: {e}")
    pretrain_train_loader = DataLoader(cast(Dataset, pretrain_train_ds), batch_size=cfg.batch_size, shuffle=pretrain_sampler is None, sampler=pretrain_sampler, num_workers=num_workers, pin_memory=pin_mem)
    pretrain_val_loader = DataLoader(cast(Dataset, pretrain_val_ds), batch_size=cfg.batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_mem)
    if cfg.dmd_csv:
        dmd_train_loader = DataLoader(cast(Dataset, dmd_train_ds), batch_size=cfg.batch_size, shuffle=dmd_sampler is None, sampler=dmd_sampler, num_workers=num_workers, pin_memory=pin_mem, drop_last=False)
        dmd_val_loader = DataLoader(cast(Dataset, dmd_val_ds), batch_size=cfg.batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_mem, drop_last=False)
    return (pretrain_train_loader, pretrain_val_loader, dmd_train_loader, dmd_val_loader)

 

def autodiscover_defaults() -> dict:

    def maybe(path: str) -> Optional[str]:
        return path if path and os.path.exists(path) else None
    info: dict = {'affectnet_csv': None, 'affectnet_root': None, 'rafdb_csv': None, 'rafdb_root': None, 'class_map_json': None, 'affectnet_eval_csv': None, 'affectnet_eval_root': None, 'rafdb_eval_csv': None, 'rafdb_eval_root': None, 'dmd_csv': None, 'dmd_root': None}
    p = maybe(os.path.join(os.getcwd(), 'affectnet_train.csv'))
    if p:
        info['affectnet_csv'] = p
    p = maybe(os.path.join(os.getcwd(), 'rafdb_train.csv'))
    if p:
        info['rafdb_csv'] = p
    p = maybe(os.path.join(os.getcwd(), 'datasets', 'affectNet', 'Train')) or maybe(os.path.join(os.getcwd(), 'datasets', 'affectNet'))
    if p:
        info['affectnet_root'] = p
    for cand in [os.path.join(os.getcwd(), 'datasets', 'raf-db', 'train'), os.path.join(os.getcwd(), 'datasets', 'raf-db', 'DATASET', 'train')]:
        p = maybe(cand)
        if p:
            info['rafdb_root'] = p
            break
    p = maybe(os.path.join(os.getcwd(), 'affectnet_map.json'))
    if p:
        info['class_map_json'] = p
    p = maybe(os.path.join(os.getcwd(), 'affectnet_test.csv'))
    if p:
        info['affectnet_eval_csv'] = p
    p = maybe(os.path.join(os.getcwd(), 'rafdb_test.csv'))
    if p:
        info['rafdb_eval_csv'] = p
    p = maybe(os.path.join(os.getcwd(), 'datasets', 'affectNet', 'Test'))
    if p:
        info['affectnet_eval_root'] = p
    for cand in [os.path.join(os.getcwd(), 'datasets', 'raf-db', 'test'), os.path.join(os.getcwd(), 'datasets', 'raf-db', 'DATASET', 'test')]:
        p = maybe(cand)
        if p:
            info['rafdb_eval_root'] = p
            break
    # DMD autodiscovery (optional)
    # Prefer exported DMD crops if present
    dmd_export_root = maybe(os.path.join(os.getcwd(), 'outputs', 'dmd_export'))
    if dmd_export_root:
        # Prefer frames CSV, else fallback to a simple CSV in export root
        for name in ['dmd_frames.csv', 'dmd.csv']:
            p = maybe(os.path.join(dmd_export_root, name))
            if p:
                info['dmd_csv'] = p
                break
        info['dmd_root'] = dmd_export_root
    else:
        # Fallbacks: CSV in CWD
        for name in ['dmd_frames.csv', 'dmd_train.csv', 'dmd.csv']:
            p = maybe(os.path.join(os.getcwd(), name))
            if p:
                info['dmd_csv'] = p
                break
        # Root preference: datasets/dmd (if only source videos) else none
        for cand in [os.path.join(os.getcwd(), 'datasets', 'dmd')]:
            p = maybe(cand)
            if p:
                info['dmd_root'] = p
                break
    return info

def make_default_cfg() -> TrainConfig:
    d = autodiscover_defaults()
    return TrainConfig(affectnet_csv=d['affectnet_csv'], affectnet_root=d['affectnet_root'], rafdb_csv=d['rafdb_csv'], rafdb_root=d['rafdb_root'], dmd_csv=d.get('dmd_csv'), dmd_root=d.get('dmd_root'), class_map_json=d['class_map_json'], num_classes=7, eff_variant='b0', img_size=112, batch_size=64, lr=0.0003, weight_decay=0.0001, epochs_pretrain=5, epochs_main=15, seed=42, out_dir='./outputs', amp=True, balance_sampling=True)

def run_training():
    cfg = make_default_cfg()
    class_map_meta = load_class_map(cfg.class_map_json)
    seed_everything(cfg.seed)
    device = torch_device()
    # DMD export can be lengthy; allow skipping via --skip_dmd (env propagated from main)
    if os.environ.get('SKIP_DMD', ''):
        print('Skipping DMD export and loading (--skip_dmd).')
        cfg.dmd_csv = None
        cfg.dmd_root = None
    else:
        # Always (re)generate DMD export (frames + CSV) so fine-tune uses fresh data
        try:
            dmd_out_root = os.path.join(os.getcwd(), 'outputs', 'dmd_export')
            run_dmd_export(dmd_out_root)
            cfg.dmd_csv = os.path.join(dmd_out_root, 'dmd_frames.csv')
            cfg.dmd_root = dmd_out_root
        except Exception as e:
            print(f'Warning: DMD export failed or skipped: {e}')
    # If a cleaned CSV exists in CWD, prefer it
    if not cfg.affectnet_csv:
        p_clean = os.path.join(os.getcwd(), 'affectnet_train.csv')
        if os.path.isfile(p_clean):
            cfg.affectnet_csv = p_clean
    # Ensure num_classes matches chosen primary set (7 by default; 8 if contempt included)
    # If mapping json exists, infer K
    try:
        cmap = load_class_map(os.path.join(os.getcwd(), 'affectnet_map.json'))
        if isinstance(cmap, dict) and len(cmap) in (7, 8):
            cfg.num_classes = int(len(cmap))
    except Exception:
        pass
    skip_generic = bool(os.environ.get('SKIP_GENERIC', ''))
    if not skip_generic:
        pretrain_train_loader, pretrain_val_loader, dmd_train_loader, dmd_val_loader = build_loaders(cfg, device)
        train_loader = pretrain_train_loader
        val_loader = pretrain_val_loader
        if train_loader is None or val_loader is None:
            raise RuntimeError('Training requested but train/val loaders are unavailable.')
    else:
        pretrain_train_loader = pretrain_val_loader = None
        dmd_train_loader = dmd_val_loader = None
        train_loader = None
        val_loader = None
    # Preview a batch from each loader (parity with main())
    def peek(loader, name: str, max_batches: int=1):
        if loader is None:
            return
        try:
            print(f'\nInspecting loader: {name}')
            bcount = 0
            t0 = time.time()
            for images, labels in loader:
                bcount += 1
                try:
                    print(f'  batch {bcount}: images={tuple(images.shape)} dtype={images.dtype} min={float(images.min()):.3f} max={float(images.max()):.3f}')
                except Exception:
                    pass
                try:
                    labs = labels.detach().cpu().numpy().tolist()
                    c = Counter(labs)
                    print(f'  labels: {dict(c)}')
                except Exception:
                    pass
                if bcount >= max_batches:
                    break
            t1 = time.time()
            print(f'  iterated {bcount} batch(es) in {t1 - t0:.2f}s')
        except Exception:
            pass
    if not skip_generic:
        peek(pretrain_train_loader, 'pretrain_train', 1)
        peek(pretrain_val_loader, 'pretrain_val', 1)
        if dmd_train_loader is not None:
            peek(dmd_train_loader, 'dmd_train', 1)
        if dmd_val_loader is not None:
            peek(dmd_val_loader, 'dmd_val', 1)
    backbone = EfficientNetFeatureExtractor('b0', pretrained=True).to(device)

    # ArcFace stability settings
    arcface_scale = 15.0
    arcface_margin = 0.15
    safe_start = bool(int(os.environ.get('ARC_FACE_SAFE_START', '0')))

    def make_head(name: str='arcface') -> nn.Module:
        margin_init = 0.0 if safe_start else arcface_margin
        return ArcFaceHead(cast(int, cast(Any, backbone).feature_dim), cfg.num_classes, m=float(margin_init), s=float(arcface_scale))
    head_name = 'arcface'

    # Always start a fresh training run: do not auto-resume model/optimizer
    # Optionally, read previous best to decide whether to overwrite best checkpoint
    def read_prev_best(out_dir: str, preferred: str) -> float:
        try:
            best_path = os.path.join(out_dir, f'best_{preferred}.pt')
            if os.path.isfile(best_path):
                c = torch.load(best_path, map_location='cpu')
                return float(c.get('best_acc', 0.0))
        except Exception:
            return 0.0
        return 0.0
    start_epoch = 0
    best_acc = read_prev_best(cfg.out_dir, head_name)
    if best_acc > 0:
        print(f"Starting fresh training. Previous best_acc found: {best_acc:.4f} (for best overwrite threshold only)")
    head_m = make_head('arcface').to(device)
    optimizer = torch.optim.Adam(list(backbone.parameters()) + list(head_m.parameters()), lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    # Diagnostics: print trainable parameter counts for backbone and head
    def param_stats(m: nn.Module) -> Tuple[int, int]:
        total = sum(p.numel() for p in m.parameters())
        trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
        return (int(total), int(trainable))
    try:
        b_tot, b_tr = param_stats(backbone)
        h_tot, h_tr = param_stats(head_m)
        print(f"[DEBUG] Backbone params: total={b_tot}, trainable={b_tr}")
        print(f"[DEBUG] Head params: total={h_tot}, trainable={h_tr}")
        print(f"[DEBUG] ArcFace settings: scale={arcface_scale}, margin={(0.0 if safe_start else arcface_margin)}")
    except Exception:
        pass

    # Diagnostics flags
    feature_shift_done = False
    current_epoch = 0

    def run_epoch(loader, train_mode: bool) -> Tuple[float, float]:
        nonlocal feature_shift_done
        if train_mode:
            backbone.train()
            head_m.train()
            # Print BatchNorm layers training/eval counts
            try:
                bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)
            except Exception:
                bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)
            try:
                bn_train = bn_eval = 0
                for m in backbone.modules():
                    if isinstance(m, bn_types):
                        if m.training:
                            bn_train += 1
                        else:
                            bn_eval += 1
                print(f"[DEBUG] BN layers: training={bn_train}, eval={bn_eval}")
            except Exception:
                pass
        else:
            backbone.eval()
            head_m.eval()
        total_loss = 0.0
        total_correct = 0
        total = 0
        batch_idx = 0
        for images, labels in progress(loader, total=len(loader), unit='batch', desc='train' if train_mode else 'val', leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.set_grad_enabled(train_mode):
                feats = backbone(images)
                # label-conditioned logits for loss
                logits_train = head_m(feats, labels)
                loss = criterion(logits_train, labels)
            if train_mode:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                # Feature movement diagnostic for first epoch, first batch
                if (not feature_shift_done) and (current_epoch == 0) and (batch_idx == 0):
                    try:
                        feats_before = feats.detach().clone()
                    except Exception:
                        feats_before = None
                    optimizer.step()
                    try:
                        with torch.no_grad():
                            feats_after = backbone(images)
                            if feats_before is not None and isinstance(feats_after, torch.Tensor):
                                diff = (feats_after.detach() - feats_before).float().reshape(feats_after.size(0), -1)
                                l2 = torch.linalg.vector_norm(diff, dim=1).mean().item()
                                print(f"[DEBUG] Feature shift L2={l2:.6e}")
                                if l2 < 1e-6:
                                    print("[WARN] Backbone features not changing; possible frozen backbone or BN stuck in eval.")
                    except Exception:
                        pass
                    finally:
                        feature_shift_done = True
                else:
                    optimizer.step()
            total_loss += float(loss.detach().cpu().item()) * images.size(0)
            # label-free logits for prediction/metrics
            logits_pred = head_m(feats, None)
            preds = torch.argmax(logits_pred, dim=1)
            total_correct += int((preds == labels).sum().detach().cpu().item())
            total += int(images.size(0))
            batch_idx += 1
        return (total_loss / max(1, total), total_correct / max(1, total))
    epochs = int(cfg.epochs_main)
    os.makedirs(cfg.out_dir, exist_ok=True)
    best_path = os.path.join(cfg.out_dir, f'best_{head_name}.pt')
    last_path = os.path.join(cfg.out_dir, f'last_{head_name}.pt')
    run_best_acc = 0.0
    # One-time sanity on label-free logits CE vs log(K)
    def sanity_ce_once(loader) -> None:
        try:
            images, labels = next(iter(loader))
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.no_grad():
                feats = backbone(images)
                logits_pred = head_m(feats, None)
                ce = nn.CrossEntropyLoss()(logits_pred, labels).item()
            exp = math.log(max(2, int(cfg.num_classes)))
            print(f"Sanity: label-free CE={ce:.3f} (expected ~log(K)={exp:.3f}) for K={int(cfg.num_classes)}")
            if ce < 0.2:
                print('Warning: very low CE suggests label/logit mismatch or degenerate head.')
        except Exception as e:
            print(f'Sanity CE skipped: {e}')

    printed_sanity = False
    for epoch in range(start_epoch, epochs):
        # Record epoch index for diagnostics
        current_epoch = epoch - start_epoch
        # If safe-start is enabled, switch margin on epoch 3 (1-based)
        if safe_start and (epoch - start_epoch) == 2:
            try:
                # Avoid type-checker complaint on nn.Module.__setattr__ stubs
                object.__setattr__(head_m, 'm', float(arcface_margin))
                object.__setattr__(head_m, 's', float(arcface_scale))
                print(f"[DEBUG] ArcFace switch at epoch {epoch + 1}: scale={arcface_scale}, margin={arcface_margin}")
            except Exception:
                pass
        # Print LR at start of epoch
        try:
            lrs = [pg.get('lr', None) for pg in optimizer.param_groups]
            lr0 = lrs[0] if lrs else None
            if lr0 is not None:
                print(f"[DEBUG] epoch {epoch + 1}: lr={lr0}")
        except Exception:
            pass
        if skip_generic:
            break
        tr_loss, tr_acc = run_epoch(train_loader, True)
        if not printed_sanity:
            try:
                sanity_ce_once(val_loader)
            except Exception:
                pass
            printed_sanity = True
        va_loss, va_acc = run_epoch(val_loader, False)
        print(f'Epoch {epoch + 1}/{epochs}  train loss {tr_loss:.4f} acc {tr_acc:.4f} | val loss {va_loss:.4f} acc {va_acc:.4f}')
        torch.save({'head_type': 'arcface', 'epoch': epoch + 1, 'best_acc': max(best_acc, va_acc), 'backbone_state': backbone.state_dict(), 'head_state': head_m.state_dict(), 'eff_variant': 'b0', 'feature_dim': backbone.feature_dim, 'optimizer': optimizer.state_dict(), 'num_classes': int(cfg.num_classes), 'class_map': class_map_meta, 'preprocess': {'img_size': int(cfg.img_size), 'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}}, last_path)
        if va_acc > best_acc:
            best_acc = va_acc
            torch.save({'head_type': 'arcface', 'epoch': epoch + 1, 'best_acc': best_acc, 'backbone_state': backbone.state_dict(), 'head_state': head_m.state_dict(), 'eff_variant': 'b0', 'feature_dim': backbone.feature_dim, 'num_classes': int(cfg.num_classes), 'class_map': class_map_meta, 'preprocess': {'img_size': int(cfg.img_size), 'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}}, best_path)
        run_best_acc = max(run_best_acc, va_acc)
    if not skip_generic:
        print(f'Training complete. Best val acc (this run) {run_best_acc:.4f}. Checkpoints saved to {cfg.out_dir}')

    # Optional second stage: fine-tune on DMD if a processed DMD CSV is available
    if cfg.dmd_csv:
        print('\nStarting DMD fine-tuning stage...')
        img_size = cfg.img_size
        train_tf, val_tf = make_transforms(img_size)
        class_map = load_class_map(cfg.class_map_json)
        dname = os.path.basename(str(cfg.dmd_csv)).lower()
        dmd_has_header = ('dmd_frames' in dname)
        # Use CsvImageDataset for processed DMD crops to avoid extra cleaning drops
        ds_full = CsvImageDataset(cfg.dmd_csv, cfg.dmd_root, transform=train_tf, class_map=class_map, has_header=dmd_has_header)
        total = len(ds_full)
        train_n = 0
        val_n = 0
        if total < 1:
            print('DMD fine-tune skipped (dataset empty).')
            return
        if total == 1:
            train_n, val_n = 1, 0
        else:
            desired = max(1, int(0.1 * total))
            val_n = min(total - 1, desired)
            train_n = total - val_n
        train_ds, val_ds = random_split(ds_full, [train_n, val_n], generator=torch.Generator().manual_seed(456))
        if hasattr(val_ds, 'dataset') and hasattr(val_ds.dataset, 'transform'):
            cast(Any, val_ds.dataset).transform = val_tf


        # === Remap labels to 0..K-1 based on union of labels present in train+val ===
        # Build index lists for the two Subset objects produced by random_split
        train_indices = list(getattr(train_ds, 'indices', []))
        val_indices   = list(getattr(val_ds, 'indices', []))

        # Compute union of labels actually present
        labs_union = sorted(set(int(ds_full.samples[i][1]) for i in (train_indices + val_indices)))
        class_to_idx = {cls: i for i, cls in enumerate(labs_union)}

        # In-place relabel the underlying ds_full.samples for just the selected indices
        for i in (train_indices + val_indices):
            rp, rl = ds_full.samples[i]
            ds_full.samples[i] = (rp, int(class_to_idx[int(rl)]))

        # Recreate Subset views so they see the remapped labels
        train_ds = Subset(ds_full, train_indices)
        val_ds   = Subset(ds_full, val_indices)

        # K is the dynamic number of classes; keep it for head construction later
        K = len(labs_union)
        print(f"[DMD] classes={K} unique={labs_union}")


        # Weighted sampler + ultra-rare drop
        def labels_from_subset(sub):
            lbls = []
            base = sub.dataset
            idxs = getattr(sub, 'indices', [])
            if hasattr(base, 'samples'):
                for i in idxs:
                    try:
                        lbls.append(cast(Any, base).samples[i][1])
                    except Exception:
                        pass
            else:
                for i in idxs:
                    try:
                        x, lab = base[i]
                        lbls.append(lab)
                    except Exception:
                        pass
            return lbls
        # Drop ultra-rare classes (<3 samples) from train
        labs = labels_from_subset(train_ds)

    
        sampler = build_weighted_sampler(labels_from_subset(train_ds))
        num_workers = get_num_workers()
        pin_mem = isinstance(device, torch.device) and getattr(device, 'type', 'cpu') == 'cuda'
        # Choose batch size to accommodate very small datasets; allow batch_size=1 safely (LayerNorm in ROI MLP)
        if train_n >= 2:
            batch_size_ft = min(cfg.batch_size, train_n)
        else:
            batch_size_ft = 1
        # Ensure we do not drop the last batch for small DMD datasets
        train_loader = DataLoader(train_ds, batch_size=batch_size_ft, shuffle=(sampler is None), sampler=sampler, num_workers=num_workers, pin_memory=pin_mem, drop_last=False)
        val_loader = DataLoader(val_ds, batch_size=max(1, min(cfg.batch_size, max(1, val_n))), shuffle=False, num_workers=num_workers, pin_memory=pin_mem, drop_last=False)
        # Detailed DMD fine-tune loader summary
        try:
            train_labels = labels_from_subset(train_ds)
            val_labels = labels_from_subset(val_ds)
            print("\n[DMD] Dataset summary:")
            print(f"  total={total}  train_n={train_n}  val_n={val_n}")
            print(f"  train label hist: {dict(Counter(train_labels))}")
            print(f"  val   label hist: {dict(Counter(val_labels))}")
            print(f"  batch_size={batch_size_ft}  train_batches={len(train_loader)}  val_batches={len(val_loader)}")
            # Extra logging per request
            K = (max(train_labels) + 1) if train_labels else 0
            print(f"[DMD] train_n={train_n}, val_n={val_n}, classes={K}")
            print(f"[DMD] train_hist={dict(Counter(train_labels))}")
            print(f"[DMD] sampler=WeightedRandomSampler(inv_freq), label_smoothing=0.05")
        except Exception:
            pass
        # Group-wise split by session/video for no leakage
        def group_key_from_path(rel: str) -> str:
            rel = str(rel or '').replace('\\', '/')
            parts = [p for p in rel.split('/') if p]
            # Expect: split/label/session/file
            if len(parts) >= 3:
                return parts[2]
            # fallback: filename stem
            stem = os.path.splitext(os.path.basename(rel))[0]
            return stem.split('_')[0]
        total = len(ds_full)
        desired = max(1, int(0.1 * total))
        val_n = min(total - 1, desired) if total > 1 else 0
        val_ratio = (float(val_n) / float(total)) if total > 0 else 0.1
        groups = {}
        for idx, (p, lab) in enumerate(ds_full.samples):
            g = group_key_from_path(p)
            groups.setdefault(g, []).append(idx)
        uniq_groups = list(groups.keys())
        rnd = random.Random(cfg.seed)
        rnd.shuffle(uniq_groups)
        n_val_groups = max(1, int(round(val_ratio * len(uniq_groups)))) if len(uniq_groups) > 1 else min(1, len(uniq_groups))
        val_groups = set(uniq_groups[:n_val_groups])
        train_groups = set(uniq_groups[n_val_groups:])
        assert train_groups.isdisjoint(val_groups)
        print(f"[DMD][split] train_groups={len(train_groups)} val_groups={len(val_groups)} overlap={len(train_groups & val_groups)}")
        train_indices = [i for g in train_groups for i in groups[g]]
        val_indices = [i for g in val_groups for i in groups[g]]

        # Remap labels to contiguous 0..K-1 based on union of train and val
        labs_union = sorted(set(int(ds_full.samples[i][1]) for i in train_indices + val_indices))
        class_to_idx = {cls: i for i, cls in enumerate(labs_union)}
        for i in (train_indices + val_indices):
            rp, rl = ds_full.samples[i]
            ds_full.samples[i] = (rp, int(class_to_idx[int(rl)]))
        K = len(labs_union)
        print(f"[DMD] classes={K} unique={labs_union}")

        # Build subsets
        train_ds = Subset(ds_full, train_indices)
        val_ds = Subset(ds_full, val_indices)

        # New head sized for remapped DMD classes
        num_classes_dmd = K
        head_ft = ArcFaceHead(cast(int, cast(Any, backbone).feature_dim), num_classes_dmd).to(device)
        try:
            object.__setattr__(head_ft, 'out_features', int(num_classes_dmd))
        except Exception:
            pass
        assert getattr(head_ft, 'out_features', num_classes_dmd) == num_classes_dmd
        criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
        out_dir = cfg.out_dir
        os.makedirs(out_dir, exist_ok=True)
        best_acc_ft = 0.0
        # Phase A: head-only
        for p in backbone.parameters():
            p.requires_grad = False
        opt_head = torch.optim.AdamW(head_ft.parameters(), lr=1e-4, weight_decay=1e-4)
        dmd_eval_banner_done = False
        def run_epoch_ft(loader, train_mode: bool) -> Tuple[float, float, Optional[float], Optional[List[int]], Optional[List[int]]]:
            nonlocal dmd_eval_banner_done
            if train_mode:
                backbone.train(); head_ft.train()
            else:
                backbone.eval(); head_ft.eval()
            total_loss = 0.0; total_correct = 0; total_sz = 0
            preds_all: List[int] = []
            labels_all: List[int] = []
            for images, labels in progress(loader, total=len(loader), unit='batch', desc=('train' if train_mode else 'val'), leave=True):
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                with torch.set_grad_enabled(train_mode):
                    feats = backbone(images)
                    if train_mode:
                        # label-conditioned for training
                        logits_train = head_ft(feats, labels)
                        loss = criterion(logits_train, labels)
                    else:
                        # label-free logits for evaluation
                        logits_free = head_ft(feats, None)
                        if not train_mode and not dmd_eval_banner_done:
                            try:
                                print(f"[DMD][eval] using_label_free_logits=True  K={num_classes_dmd}  head_out_features={getattr(head_ft,'out_features', head_ft.W.shape[0])}")
                            except Exception:
                                pass
                            dmd_eval_banner_done = True
                        loss = criterion(logits_free, labels)
                if train_mode:
                    opt_head.zero_grad(set_to_none=True); loss.backward(); opt_head.step()
                total_loss += float(loss.detach().cpu().item()) * images.size(0)
                # label-free logits for prediction/metrics
                logits_pred = head_ft(feats, None)
                preds = torch.argmax(logits_pred, dim=1)
                total_correct += int((preds == labels).sum().detach().cpu().item())
                total_sz += int(images.size(0))
                if not train_mode:
                    try:
                        preds_all.extend(preds.detach().cpu().tolist())
                        labels_all.extend(labels.detach().cpu().tolist())
                    except Exception:
                        pass
            macro_f1 = None
            if not train_mode and labels_all:
                try:
                    K = int(num_classes_dmd)
                    # compute macro-F1
                    f1s = []
                    for c in range(K):
                        tp = sum(1 for p, t in zip(preds_all, labels_all) if p == c and t == c)
                        fp = sum(1 for p, t in zip(preds_all, labels_all) if p == c and t != c)
                        fn = sum(1 for p, t in zip(preds_all, labels_all) if p != c and t == c)
                        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
                        f1s.append(f1)
                    macro_f1 = float(sum(f1s) / max(1, len(f1s)))
                except Exception:
                    macro_f1 = None
            return (total_loss / max(1, total_sz), total_correct / max(1, total_sz), macro_f1, (preds_all if not train_mode else None), (labels_all if not train_mode else None))
        for epoch in range(3):
            tr_loss, tr_acc, mf1, pa, la = run_epoch_ft(train_loader, True)
            va_loss, va_acc, va_f1, preds_all, labels_all = run_epoch_ft(val_loader, False)
            # Per-epoch detailed log for DMD head-only phase
            try:
                extra = '' if va_f1 is None else f" macroF1 {va_f1:.4f}"
                print(f"[DMD][HeadOnly] Epoch {epoch + 1}/3  train loss {tr_loss:.4f} acc {tr_acc:.4f} | val loss {va_loss:.4f} acc {va_acc:.4f}{extra} | batches train={len(train_loader)} val={len(val_loader)}")
                # Extended evaluation report
                try:
                    if prfs is not None and confusion_matrix is not None and preds_all is not None and labels_all is not None:
                        Kloc = int(num_classes_dmd)
                        prec_arr, rec_arr, f1_arr, sup_arr = prfs(labels_all, preds_all, labels=list(range(Kloc)), zero_division=0)
                        prec_arr = ensure_list_metric(prec_arr); rec_arr = ensure_list_metric(rec_arr); f1_arr = ensure_list_metric(f1_arr); sup_arr = ensure_list_metric(sup_arr)
                        print('class_id | precision | recall | f1 | support')
                        for cid in range(Kloc):
                            print(f'{cid:8d} | {prec_arr[cid]:9.4f} | {rec_arr[cid]:6.4f} | {f1_arr[cid]:.4f} | {safe_int(sup_arr[cid])}')
                        cm = confusion_matrix(labels_all, preds_all, labels=list(range(Kloc)))
                        print('Confusion matrix (rows=true, cols=pred):')
                        print(cm)
                        if plt is not None:
                            plt.figure(figsize=(6,5))
                            plt.imshow(cm, interpolation='nearest', cmap='Blues')
                            plt.title('Confusion Matrix')
                            plt.colorbar()
                            plt.xlabel('Predicted')
                            plt.ylabel('True')
                            plt.tight_layout()
                            os.makedirs('./outputs', exist_ok=True)
                            img_path = os.path.join('./outputs', f'confusion_matrix_epoch{epoch + 1}.png')
                            plt.savefig(img_path)
                            plt.close()
                except Exception:
                    pass
            except Exception:
                pass
            best_acc_ft = max(best_acc_ft, va_acc)
            torch.save({'head_type': 'arcface', 'epoch': epoch + 1, 'phase': 'head_only', 'best_acc': best_acc_ft, 'backbone_state': backbone.state_dict(), 'head_state': head_ft.state_dict(), 'eff_variant': 'b0', 'feature_dim': backbone.feature_dim, 'num_classes': int(num_classes_dmd), 'class_map': class_map, 'preprocess': {'img_size': int(img_size), 'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}}, os.path.join(out_dir, 'last_finetune_arcface.pt'))
        # Phase B: unfreeze backbone
        for p in backbone.parameters():
            p.requires_grad = True
        optimizer_ft = torch.optim.AdamW([
            {'params': backbone.parameters(), 'lr': 1e-5, 'weight_decay': 1e-4},
            {'params': head_ft.parameters(), 'lr': 1e-4, 'weight_decay': 1e-4},
        ])
        for epoch in range(12):
            # train
            backbone.train(); head_ft.train()
            total_loss = 0.0; total_correct = 0; total_sz = 0
            for images, labels in progress(train_loader, total=len(train_loader), unit='batch', desc='train', leave=True):
                images = images.to(device, non_blocking=True); labels = labels.to(device, non_blocking=True)
                feats = backbone(images)
                # label-conditioned logits for loss
                logits_train = head_ft(feats, labels)
                loss = criterion(logits_train, labels)
                optimizer_ft.zero_grad(set_to_none=True); loss.backward(); optimizer_ft.step()
                total_loss += float(loss.detach().cpu().item()) * images.size(0)
                # label-free logits for prediction/metrics
                logits_pred = head_ft(feats, None)
                preds = torch.argmax(logits_pred, dim=1); total_correct += int((preds == labels).sum().detach().cpu().item()); total_sz += int(images.size(0))
            va_loss, va_acc, va_f1, preds_all, labels_all = run_epoch_ft(val_loader, False)
            # Per-epoch detailed log for DMD unfreeze phase
            try:
                extra = '' if va_f1 is None else f" macroF1 {va_f1:.4f}"
                print(f"[DMD][Unfreeze] Epoch {epoch + 1}/12  train loss {total_loss/max(1,total_sz):.4f} acc {total_correct/max(1,total_sz):.4f} | val loss {va_loss:.4f} acc {va_acc:.4f}{extra} | batches train={len(train_loader)} val={len(val_loader)}")
                # Extended evaluation report
                try:
                    if prfs is not None and confusion_matrix is not None and preds_all is not None and labels_all is not None:
                        Kloc = int(num_classes_dmd)
                        prec_arr, rec_arr, f1_arr, sup_arr = prfs(labels_all, preds_all, labels=list(range(Kloc)), zero_division=0)
                        prec_arr = ensure_list_metric(prec_arr); rec_arr = ensure_list_metric(rec_arr); f1_arr = ensure_list_metric(f1_arr); sup_arr = ensure_list_metric(sup_arr)
                        print('class_id | precision | recall | f1 | support')
                        for cid in range(Kloc):
                            print(f'{cid:8d} | {prec_arr[cid]:9.4f} | {rec_arr[cid]:6.4f} | {f1_arr[cid]:.4f} | {safe_int(sup_arr[cid])}')
                        cm = confusion_matrix(labels_all, preds_all, labels=list(range(Kloc)))
                        print('Confusion matrix (rows=true, cols=pred):')
                        print(cm)
                        if plt is not None:
                            plt.figure(figsize=(6,5))
                            plt.imshow(cm, interpolation='nearest', cmap='Blues')
                            plt.title('Confusion Matrix')
                            plt.colorbar()
                            plt.xlabel('Predicted')
                            plt.ylabel('True')
                            plt.tight_layout()
                            os.makedirs('./outputs', exist_ok=True)
                            img_path = os.path.join('./outputs', f'confusion_matrix_epoch{3 + epoch + 1}.png')
                            plt.savefig(img_path)
                            plt.close()
                except Exception:
                    pass
            except Exception:
                pass
            best_acc_ft = max(best_acc_ft, va_acc)
            # save
            last_path = os.path.join(out_dir, 'last_finetune_arcface.pt')
            best_path = os.path.join(out_dir, 'best_finetune_arcface.pt')
            torch.save({'head_type': 'arcface', 'epoch': 3 + epoch + 1, 'best_acc': best_acc_ft, 'backbone_state': backbone.state_dict(), 'head_state': head_ft.state_dict(), 'eff_variant': 'b0', 'feature_dim': backbone.feature_dim, 'optimizer': optimizer_ft.state_dict(), 'num_classes': int(num_classes_dmd), 'class_map': class_map, 'preprocess': {'img_size': int(img_size), 'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}}, last_path)
            if va_acc >= best_acc_ft:
                torch.save({'head_type': 'arcface', 'epoch': 3 + epoch + 1, 'best_acc': best_acc_ft, 'backbone_state': backbone.state_dict(), 'head_state': head_ft.state_dict(), 'eff_variant': 'b0', 'feature_dim': backbone.feature_dim, 'num_classes': int(num_classes_dmd), 'class_map': class_map, 'preprocess': {'img_size': int(img_size), 'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}}, best_path)
        print(f'DMD fine-tune complete. Best val acc {best_acc_ft:.4f}. Checkpoints saved to {out_dir}')


# --- AffectNet CSV cleaning/generation ---------------------------------------------------------

def gen_csv_from_affectnet_labels(labels_csv: str, images_root: str, out_csv: str, keep_classes: Iterable[int], map_unknown: Optional[int]=None, out_map_json: Optional[str]='affectnet_map.json') -> int:
    # Load labels.csv and emit a cleaned CSV with contiguous labels 0..K-1
    df = pd.read_csv(labels_csv)
    # heuristics for columns
    path_cols = ['path', 'file', 'image', 'subDirectory_filePath', 'subdirectory_file_path', 'filename']
    label_cols = ['label', 'expression', 'expr']
    pcol = None
    for c in path_cols:
        if c in df.columns:
            pcol = c; break
    if pcol is None:
        # fallback: first string-like column
        for c in df.columns:
            if df[c].dtype == object:
                pcol = c; break
    lcol = None
    for c in label_cols:
        if c in df.columns:
            lcol = c; break
    if lcol is None:
        # fallback: first numeric column
        for c in df.columns:
            if str(df[c].dtype).startswith('int') or str(df[c].dtype).startswith('float'):
                lcol = c; break
    if pcol is None or lcol is None:
        raise ValueError(f'Cannot locate path/label columns in {labels_csv}. Columns={list(df.columns)}')
    # normalize labels
    df['_label_raw'] = df[lcol].astype(int)
    keep_set = set(int(x) for x in keep_classes)
    total_in = len(df)
    unique_before = sorted(df['_label_raw'].unique().tolist())
    # filter rows
    def keep(lbl: int) -> bool:
        if lbl < 0:
            return False
        if lbl in keep_set:
            return True
        return (map_unknown is not None) and (lbl == map_unknown)
    df = df[df['_label_raw'].map(lambda x: keep(int(x)))].copy()
    # build mapping old->new using the order of sorted keep_classes
    ordered = [int(x) for x in keep_classes]
    mapping = { old: i for i, old in enumerate(ordered) }
    df['_label_new'] = df['_label_raw'].map(lambda x: mapping.get(int(x), None))
    df = df[df['_label_new'].map(lambda x: x is not None)]
    # path normalization and relativization
    images_root_abs = os.path.abspath(images_root)
    out_rows: List[Tuple[str,int]] = []
    def relpath(p: str) -> str:
        if not isinstance(p, str):
            p = str(p)
        p = p.replace('\\', '/').lstrip('/')
        # If p is absolute, make relative to images_root when possible
        try:
            if os.path.isabs(p):
                p_rel = os.path.relpath(p, images_root_abs)
            else:
                p_rel = p
        except Exception:
            p_rel = p
        return p_rel.replace('\\', '/').strip()
    for _, row in df.iterrows():
        sub = relpath(row[pcol])
        out_rows.append((sub, int(row['_label_new'])))
    os.makedirs(os.path.dirname(out_csv) or '.', exist_ok=True)
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        for rel, lab in out_rows:
            w.writerow([rel, int(lab)])
    total_out = len(out_rows)
    unique_after = sorted(set(l for _, l in out_rows))
    print(f'AffectNet cleaner: in={total_in} out={total_out} unique_before={unique_before} unique_after={unique_after}')
    print(f'AffectNet mapping (old->new): {mapping}')
    if out_map_json:
        try:
            with open(out_map_json, 'w', encoding='utf-8') as f:
                json.dump(mapping, f, indent=2)
        except Exception:
            pass
    return int(len(mapping))

def run_dmd_export(out_root: Optional[str]=None) -> None:
    seed_everything(42)
    dmd_root = os.path.join(os.getcwd(), 'datasets', 'dmd')
    if not os.path.isdir(dmd_root):
        raise FileNotFoundError(f'DMD root not found: {dmd_root}')
    out_root = out_root or os.path.abspath(os.path.join('.', 'outputs', 'dmd_export'))
    os.makedirs(out_root, exist_ok=True)

    # Permissive quality thresholds (env-overridable)
    blur_cutoff = float(os.environ.get('DMD_BLUR_CUTOFF', '25.0'))
    exposure_min = float(os.environ.get('DMD_EXPOSURE_MIN', '5.0'))
    exposure_max = float(os.environ.get('DMD_EXPOSURE_MAX', '250.0'))
    skin_min = float(os.environ.get('DMD_SKIN_MIN_FRAC', '0.10'))

    sample_every = int(os.environ.get('DMD_SAMPLE_EVERY', '1'))
    sf = os.environ.get('DMD_SAMPLE_FPS', '').strip()
    sample_fps = float(sf) if sf else None
    exclude_boundary_s = 0.5
    img_size = 112

    face_stream_keys = [k.strip().lower() for k in os.environ.get('DMD_FACE_STREAM_KEY', 'rgb_face,driver_front,face,driver_face').split(',') if k.strip()]

    try:
        yunet = YuNetDetector(model_path=os.environ.get('YUNET_ONNX', 'face_detection_yunet.onnx'), input_size=(320, 320), conf_threshold=0.6, nms_threshold=0.3, top_k=5000)
    except Exception as e:
        raise RuntimeError(f'YuNet unavailable: {e}')

    video_exts = {'.mp4', '.avi', '.mov', '.mkv'}

    # Helpers
    def norm_name(s: str) -> str:
        s = os.path.basename(s).lower()
        return re.sub(r'[^a-z0-9]+', '', s)

    def list_files_ci(folder: str, exts: Iterable[str]) -> List[str]:
        out: List[str] = []
        try:
            for fn in os.listdir(folder):
                ext = os.path.splitext(fn)[1].lower()
                if ext in exts:
                    out.append(os.path.join(folder, fn))
        except Exception:
            pass
        out.sort(key=lambda p: os.path.basename(p).lower())
        return out

    def json_face_video(json_obj: dict, folder: str) -> Optional[str]:
        try:
            ol = json_obj.get('openlabel') or json_obj
            streams = ol.get('streams') or {}
            items = list(streams.items()) if isinstance(streams, dict) else []
            def score(item):
                k, v = item
                nm = str(k).lower()
                best = -1
                for i, key in enumerate(face_stream_keys):
                    if key and key in nm:
                        best = max(best, 100 - i)
                return best
            items.sort(key=score, reverse=True)
            uris: List[str] = []
            for k, v in items:
                if isinstance(v, dict):
                    for kk in ('uri', 'URL', 'url'):
                        u = v.get(kk)
                        if isinstance(u, str) and u.strip():
                            uris.append(u)
            if not uris:
                return None
            vids = list_files_ci(folder, video_exts)
            vids_by_norm = { norm_name(x): x for x in vids }
            for uri in uris:
                base = os.path.basename(uri)
                cand = os.path.join(folder, base)
                if os.path.isfile(cand):
                    return cand
                stem = os.path.splitext(base)[0].lower()
                for v in vids:
                    if os.path.splitext(os.path.basename(v))[0].lower() == stem:
                        return v
                n = norm_name(base)
                if n in vids_by_norm:
                    return vids_by_norm[n]
            return None
        except Exception:
            return None

    def infer_split_from(json_obj: Optional[dict], folder: str) -> str:
        sp = None
        try:
            if isinstance(json_obj, dict):
                sp = (json_obj.get('split') or (json_obj.get('openlabel') or {}).get('split'))
        except Exception:
            sp = None
        if isinstance(sp, str) and sp:
            return sp.lower()
        p = folder.lower()
        if ('/test/' in ('/' + p + '/')) or ('\\test\\' in ('\\' + p + '\\')):
            return 'test'
        if ('/val/' in ('/' + p + '/')) or ('\\val\\' in ('\\' + p + '\\')):
            return 'val'
        return 'train'

    def parse_openlabel_actions(json_obj: dict, fps: float) -> List[Tuple[int, int, int]]:
        """
        Build intervals from OpenLABEL actions using the action 'type' string
        mapped to our 11-class IDs via DMD_TYPE_TO_ID.
        Returns a list of (frame_start, frame_end, class_id).
        """
        ol = json_obj.get('openlabel') or json_obj
        actions = ol.get('actions') or {}
        out: List[Tuple[int, int, int]] = []
        if not isinstance(actions, dict) or not actions:
            return out

        def to_frames(lo, hi) -> Optional[Tuple[int, int]]:
            try:
                if lo is None or hi is None:
                    return None
                if isinstance(lo, (float, int)) and isinstance(hi, (float, int)):
                    if isinstance(lo, float) or isinstance(hi, float):
                        s0 = float(lo); s1 = float(hi)
                        a = int(max(0, np.floor(s0 * fps)))
                        b = int(max(a, np.floor(s1 * fps)))
                        return (a, b)
                    a = int(lo); b = max(int(lo), int(hi))
                    return (a, b)
            except Exception:
                return None
            return None

        for aid, act in actions.items():
            try:
                if not isinstance(act, dict):
                    continue
                typ = (act.get('type') or '').strip().lower()
                cid = DMD_TYPE_TO_ID.get(typ, 0)  # unknown → neutral (0)

                intervals = (act.get('frame_intervals') or
                            act.get('intervals') or
                            act.get('time_intervals') or [])
                for it in intervals:
                    lo = hi = None
                    if isinstance(it, (list, tuple)) and len(it) >= 2:
                        lo, hi = it[0], it[1]
                    elif isinstance(it, dict):
                        lo = it.get('frame_start') if 'frame_start' in it else (it.get('start') or it.get('t0'))
                        hi = it.get('frame_end')   if 'frame_end'   in it else (it.get('end')   or it.get('t1'))
                    rng = to_frames(lo, hi)
                    if rng is None:
                        continue
                    a, b = rng
                    if b < a:
                        a, b = b, a
                    out.append((int(a), int(b), int(cid)))
            except Exception:
                continue

        out.sort(key=lambda t: (t[0], t[1], t[2]))
        return out

    def sample_frames(intervals: Sequence[Tuple[int, int, int]], fps: float) -> List[int]:
        TARGET_FPS = 2.0
        MIN_SPACING = 3
        frames: List[int] = []
        for a, b, cid in intervals:
            if b < a:
                a, b = b, a
            length = max(0, b - a + 1)
            if length <= 0:
                continue
            if sample_every and sample_every > 0:
                step = max(1, int(sample_every))
            elif sample_fps is not None and sample_fps > 0:
                step = max(1, int(round(max(fps, 1.0) / float(sample_fps))))
            else:
                step = max(1, int(round(max(fps, 1.0) / TARGET_FPS)))
            if length <= step:
                frames.append(int(a + length // 2))
            else:
                cand = list(range(int(a), int(b) + 1, int(step)))
                filtered: List[int] = []
                last = None
                for f in cand:
                    if last is None or (int(f) - int(last)) >= MIN_SPACING:
                        filtered.append(int(f))
                        last = int(f)
                frames.extend(filtered)
        return sorted(set(int(x) for x in frames if x >= 0))

    def choose_label_and_interval(
        fidx: int,
        intervals: Sequence[Tuple[int, int, int]]
    ) -> Optional[Tuple[int, Tuple[int, int, int]]]:
        """
        Pick a single label for frame fidx from intervals of (a, b, cid),
        preferring higher DMD_PRIORITY, then longer duration, then cid.
        Returns (label_id, (a, b, cid)) or None if nothing active.
        """
        best_key = None  # (priority, duration, cid)
        best_iv = None
        for a, b, cid in intervals:
            if a <= fidx <= b:
                pr = DMD_PRIORITY.get(int(cid), 0)
                dur = int(b - a + 1)
                key = (pr, dur, int(cid))
                if best_key is None or key > best_key:
                    best_key = key
                    best_iv = (a, b, int(cid))
        if best_iv is None:
            return None
        return (best_iv[2], best_iv)


    # Detection/cropping helper (reuse relaxed behavior)
    def detect_and_crop_frame(bgr: np.ndarray):
        if bgr is None or bgr.size == 0:
            return (None, None, None, None, None, None)
        try:
            dets = yunet.detect(bgr)
            if dets:
                best = dets[0]
                prob = float(best.get('score', 0.0))
                x, y, w, h = best['bbox']
                bbox = (float(x), float(y), float(x + w), float(y + h))
                pts5 = np.array(best['kps'], dtype=np.float32)
                pil_rgb = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                if pts5.shape == (5, 2):
                    aligned, pts_t = align_face_5pt(pil_rgb, pts5, out_size=img_size, margin=0.1)
                    le, reye, mo = roi_boxes_from_landmarks(pts_t, out_size=img_size, eye_size=64, mouth_size=64)
                    return (aligned, prob, bbox, le, reye, mo)
                x1, y1, x2, y2 = bbox
                bw = max(1.0, float(x2 - x1))
                bh = max(1.0, float(y2 - y1))
                m = 0.1
                dw = m * bw
                dh = m * bh
                nx1 = max(0.0, x1 - dw)
                ny1 = max(0.0, y1 - dh)
                nx2 = min(float(bgr.shape[1]), x2 + dw)
                ny2 = min(float(bgr.shape[0]), y2 + dh)
                pil_rgb = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                cropped = pil_rgb.crop((nx1, ny1, nx2, ny2))
                try:
                    cropped = ImageOps.fit(cropped, (img_size, img_size), method=PIL_RESAMPLE_BILINEAR, centering=(0.5, 0.5))
                except Exception:
                    cropped = cropped.resize((img_size, img_size))
                return (cropped, prob, bbox, None, None, None)
        except Exception:
            pass
        pil_rgb = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        try:
            fitted = ImageOps.fit(pil_rgb, (img_size, img_size), method=PIL_RESAMPLE_BILINEAR, centering=(0.5, 0.5))
        except Exception:
            fitted = pil_rgb.resize((img_size, img_size))
        return (fitted, None, None, None, None, None)

    # Iterate session folders, form pairs, sample and export
    video_exts_local = video_exts
    all_folders = set()
    for root, dirs, files in os.walk(dmd_root):
        try:
            if isinstance(dirs, list):
                dirs[:] = [d for d in dirs if str(d).lower() != 'backup']
        except Exception:
            pass
        if os.path.basename(root).lower() == 'backup':
            continue
        vids = [f for f in files if os.path.splitext(f)[1].lower() in video_exts_local]
        jsons = [f for f in files if os.path.splitext(f)[1].lower() == '.json']
        if vids or jsons:
            all_folders.add(root)

    folders_with_index_pairing = 0
    total_rows = 0
    hist_by_split: Dict[str, Counter] = {'train': Counter(), 'val': Counter(), 'test': Counter()}
    rows_csv: List[Tuple[str, int, str, int, str, float, Optional[float], Optional[Tuple[int,int,int,int]], Optional[Tuple[int,int,int,int]], Optional[Tuple[int,int,int,int]], str, float, float]] = []
    global_sum = np.zeros(3, dtype=np.float64)
    global_sumsq = np.zeros(3, dtype=np.float64)
    global_count = 0

    # Caps to avoid dominance
    PER_INTERVAL_CAP = 80
    PER_VIDEO_CLASS_CAP = 1500
    TRAIN_MIN_PER_CLASS = 300

    for folder in sorted(all_folders, key=lambda p: p.lower()):
        vids = list_files_ci(folder, video_exts_local)
        js = list_files_ci(folder, {'.json'})
        if not vids or not js:
            continue
        # First pass: streams-based pairing
        used_v = set(); used_j = set(); pairs: List[Tuple[str, str]] = []
        for jpath in js:
            try:
                j = json.load(open(jpath, 'r', encoding='utf-8'))
            except Exception:
                continue
            ref = json_face_video(j, folder)
            if ref and ref in vids and ref not in used_v:
                pairs.append((ref, jpath))
                used_v.add(ref); used_j.add(jpath)
        # Second pass: index pairing for remainder
        remain_v = [v for v in vids if v not in used_v]
        remain_j = [j for j in js if j not in used_j]
        idx_pairs = min(len(remain_v), len(remain_j))
        if idx_pairs > 0:
            folders_with_index_pairing += 1
        for k in range(idx_pairs):
            pairs.append((remain_v[k], remain_j[k]))

        print(f'DMD folder: {folder} -> videos={len(vids)} jsons={len(js)} pairs={len(pairs)}')

        # Process pairs
        # Per-video class cap tracker
        per_video_class_counts: Dict[Tuple[str, int], int] = {}
        for (vid_path, json_path) in pairs:
            try:
                meta = json.load(open(json_path, 'r', encoding='utf-8'))
            except Exception:
                print(f'  Skipping invalid JSON: {json_path}')
                continue
            split = infer_split_from(meta, folder)
            cap = cv2.VideoCapture(vid_path)
            if not cap.isOpened():
                print(f'  Warning: cannot open video: {vid_path}')
                continue
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            fps = fps if fps and fps > 0 else 30.0
            fcount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT) else 0
            duration = (fcount / float(fps)) if fcount > 0 else None
            intervals = parse_openlabel_actions(meta, float(fps))
            if not intervals:
                print(f'  Warning: no actions found in {json_path}; skipping')
                cap.release()
                continue
            frame_indices = sample_frames(intervals, float(fps))
            if exclude_boundary_s and duration is not None and duration > 0:
                lo = float(exclude_boundary_s)
                hi = max(0.0, float(duration) - float(exclude_boundary_s))
                frame_indices = [fi for fi in frame_indices if (float(fi) / float(fps)) >= lo and (float(fi) / float(fps)) <= hi]

            print(
                f"  Pair: video={vid_path} json={os.path.basename(json_path)} "
                f"actions={len(intervals)} sampled_frames={len(frame_indices)}"
            )

            if not frame_indices:
                cap.release()
                continue

            session_id = os.path.basename(folder)
            history: List[Tuple[float, int]] = []
            per_interval_counts: Dict[Tuple[int, int, int], int] = {}

            # ---- DMD export local toggles (no env) ----
            DMD_INCLUDE_NEG = True    # allow unlabeled frames to map to neutral
            DMD_NONE_ID     = 0       # neutral/none class ID

            # Initialize counters once before the loop
            frame_counter = 0
            total_frames = len(frame_indices)
            prev_idx = -1


            for fidx in frame_indices:
                frame_counter += 1
                if frame_counter % 100 == 0:
                    print(f"[DMD][export] {vid_path}: {frame_counter}/{total_frames} frames processed...")

                # ---- fast seek / decode ----
                fi = int(fidx)
                if prev_idx >= 0 and fi > prev_idx:
                    to_skip = fi - prev_idx - 1
                    for _ in range(to_skip):
                        cap.grab()
                else:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, fi)

                ok, frame = cap.read()
                prev_idx = fi
                if not ok or frame is None:
                    continue

                # labeling with in-script negative fallback
                # Genuinely hell code right here, so many problems stemmed from this specific block of code
                li = choose_label_and_interval(fi, intervals)  # Optional[(label_id, (a,b,cid))]
                if not li or li[0] is None:
                    label_id = int(DMD_NONE_ID)
                    interval_key = (-1, -1, -1)  # all-int key to satisfy Dict[Tuple[int,int,int],int]
                else:
                    label_id, interval_key = li  # interval_key is (a,b,cid) -> all int

                if label_id in (8, 9, 10):
                    continue


                # Per-interval cap enforced via a simple counter keyed by (start,end,label)
                # Approximate by counting per label across this video segment
                key_vc = (vid_path, int(label_id))
                c_vc = per_video_class_counts.get(key_vc, 0)
                if c_vc >= PER_VIDEO_CLASS_CAP:
                    continue
                # Interval cap
                c_int = per_interval_counts.get(interval_key, 0)
                if c_int >= PER_INTERVAL_CAP:
                    continue
                crop, prob, bbox, le, reye, mo = detect_and_crop_frame(frame)
                if crop is None or not isinstance(crop, Image.Image):
                    continue
                if bbox is not None:
                    _, y1b, _, y2b = bbox
                    if (y2b - y1b) < 64:
                        continue
                lap_var = laplacian_variance(crop)
                if lap_var < blur_cutoff:
                    continue
                mg = float(np.array(crop.convert('L')).mean())
                if mg < exposure_min or mg > exposure_max:
                    continue
                if skin_coverage_fraction(crop) < skin_min:
                    continue
                time_sec = float(fidx) / float(fps)
                ph = compute_phash(crop)
                if any(abs(time_sec - tprev) <= 1.0 and hamming_distance(ph, hp) <= 6 for tprev, hp in history):
                    continue
                arr = np.array(crop.convert('RGB'), dtype=np.float32) / 255.0
                flat = arr.reshape(-1, 3)
                global_sum += flat.sum(axis=0).astype(np.float64)
                global_sumsq += (flat ** 2).sum(axis=0).astype(np.float64)
                global_count += int(arr.shape[0] * arr.shape[1])
                class_dir = os.path.join(out_root, split, str(label_id), session_id)
                os.makedirs(class_dir, exist_ok=True)
                out_name = f'f{int(fidx):06d}.jpg'
                out_path = os.path.join(class_dir, out_name)
                try:
                    crop.save(out_path, format='JPEG', quality=95)
                except Exception:
                    continue
                rel = os.path.relpath(out_path, out_root).replace('\\', '/')
                rows_csv.append((rel, int(label_id), session_id, int(fidx), split, float(lap_var), prob, le, reye, mo, 'cam0', float(fps), float(time_sec)))
                total_rows += 1
                hist_by_split.setdefault(split, Counter())
                hist_by_split[split][int(label_id)] += 1
                history.append((time_sec, ph))
                history = [it for it in history if (time_sec - it[0]) <= 1.0]
                per_video_class_counts[key_vc] = per_video_class_counts.get(key_vc, 0) + 1
                per_interval_counts[interval_key] = per_interval_counts.get(interval_key, 0) + 1
            cap.release()

    # Soft rebalance: oversample train classes to at least TRAIN_MIN_PER_CLASS by duplicating samples
    rows_before = len(rows_csv)
    try:
        train_rows = [r for r in rows_csv if r[4] == 'train']
        by_label: Dict[int, List[Tuple]] = defaultdict(list)
        for r in train_rows:
            by_label[int(r[1])].append(r)
        # oversample
        labels_all = sorted(set(int(r[1]) for r in train_rows))
        for lab in labels_all:
            cur = len(by_label.get(lab, []))
            while cur < TRAIN_MIN_PER_CLASS and by_label.get(lab):
                src = random.choice(by_label[lab])
                # duplicate row (reuse path)
                rows_csv.append(src)
                hist_by_split['train'][lab] += 1
                cur += 1
    except Exception:
        pass
    rows_after = len(rows_csv)


    # Write CSV
    csv_path = os.path.join(out_root, 'dmd_frames.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(['path', 'label', 'session', 'frame_idx', 'split', 'blur_metric', 'landmarks_quality', 'modality', 'left_eye_roi', 'right_eye_roi', 'mouth_roi', 'camera_id', 'fps', 'time_sec'])
        def fmt_roi(box):
            if box is None:
                return ''
            x, y, w, h = box
            return f'{x},{y},{w},{h}'
        for rel, label, session_id, fidx, split, bm, prob, le, reye, mo, cam_id, fps_val, tsec in rows_csv:
            writer.writerow([rel, label, session_id, fidx, split, f'{bm:.6f}', ('' if prob is None else f'{prob:.6f}'), 'rgb', fmt_roi(le), fmt_roi(reye), fmt_roi(mo), cam_id, f'{fps_val:.3f}', f'{tsec:.3f}'])

    # Normalization stats
    if global_count > 0:
        g_mean = (global_sum / global_count).tolist()
        g_var = (global_sumsq / global_count) - np.square(global_sum / global_count)
        g_std = np.sqrt(np.maximum(g_var, 1e-12)).tolist()
    else:
        g_mean = [0.485, 0.456, 0.406]
        g_std = [0.229, 0.224, 0.225]
    with open(os.path.join(out_root, 'normalization_stats.json'), 'w', encoding='utf-8') as fjs:
        json.dump({'img_size': img_size, 'channels': 'rgb', 'computed_on': 'all', 'global': {'mean': g_mean, 'std': g_std, 'count_pixels_per_channel': int(global_count)}}, fjs, indent=2)

    # Class map (OpenLABEL names -> ids)
    if DMD_ACTION_NAME_TO_ID:
        with open(os.path.join(out_root, 'class_map.json'), 'w', encoding='utf-8') as fcm:
            json.dump(DMD_ACTION_NAME_TO_ID, fcm, indent=2, ensure_ascii=False)

    # Diagnostics
    print(f'Pairing summary: folders_with_index_pairing={int(folders_with_index_pairing)} (evidence of non-basename pairing)')
    print(f'DMD export wrote total rows: {total_rows}')
    for split, hist in hist_by_split.items():
        if hist:
            print(f'  Split={split} histogram: {dict(hist)}')
    # Summary diagnostics for acceptance
    try:
        hist_train = dict(hist_by_split.get('train', Counter()))
        hist_val = dict(hist_by_split.get('val', Counter()))
        classes_K = len(set(list(hist_train.keys()) + list(hist_val.keys())))
        print(f"[DMD][export] rows_before={rows_before} rows_after={rows_after} classes={classes_K} hist_train={hist_train} hist_val={hist_val}")
    except Exception:
        pass
    print(f'DMD export complete. CSV: {csv_path}')

def main():
    parser = argparse.ArgumentParser(description='FER: minimal training CLI')
    parser.add_argument('--train', action='store_true', help='Run training on available datasets (ArcFace head)')
    parser.add_argument('--skip_dmd', action='store_true', help='Skip DMD export and loading (faster generic data testing)')
    parser.add_argument('--skip_generic', action='store_true', help='Skip generic FER stage (train only on DMD)')
    parser.add_argument('--arcface_safe_start', action='store_true', help='Epochs 1-2: margin-free cosine; from epoch 3 use ArcFace (s=15, m=0.15)')
    # Optional AffectNet cleaning flags
    parser.add_argument('--affectnet_primary8', action='store_true', help='Use primary 8-class AffectNet (include contempt as class 7)')
    parser.add_argument('--affectnet_labels_csv', type=str, default=None, help='Path to AffectNet labels.csv (train) to clean into affectnet_train.csv')
    parser.add_argument('--affectnet_labels_csv_test', type=str, default=None, help='Path to AffectNet labels.csv (test) to clean into affectnet_test.csv')
    args = parser.parse_args()
    # If training is requested, run optional AffectNet cleaning first (if flags provided), then train.
    if getattr(args, 'train', False):
        try:
            if getattr(args, 'affectnet_labels_csv', None):
                labels_csv = args.affectnet_labels_csv
                images_root = os.path.join(os.getcwd(), 'datasets', 'affectNet', 'Train')
                if not os.path.exists(images_root):
                    alt = os.path.join(os.getcwd(), 'datasets', 'affectNet')
                    if os.path.exists(alt):
                        images_root = alt
                keep = list(range(8 if getattr(args, 'affectnet_primary8', False) else 7))
                out_csv = os.path.join(os.getcwd(), 'affectnet_train.csv')
                gen_csv_from_affectnet_labels(labels_csv, images_root, out_csv, keep_classes=keep, map_unknown=None, out_map_json=os.path.join(os.getcwd(), 'affectnet_map.json'))
                if getattr(args, 'affectnet_labels_csv_test', None):
                    labels_csv_t = args.affectnet_labels_csv_test
                    images_root_t = os.path.join(os.getcwd(), 'datasets', 'affectNet', 'Test')
                    if os.path.exists(labels_csv_t) and os.path.exists(images_root_t):
                        out_csv_t = os.path.join(os.getcwd(), 'affectnet_test.csv')
                        gen_csv_from_affectnet_labels(labels_csv_t, images_root_t, out_csv_t, keep_classes=keep, map_unknown=None, out_map_json=os.path.join(os.getcwd(), 'affectnet_map.json'))
        except Exception as e:
            print(f'AffectNet cleaning skipped: {e}')
        # AffectNet flags
        if getattr(args, 'affectnet_7class', False):
            os.environ['AFFECTNET_7CLASS'] = '1'
        if getattr(args, 'affectnet_cache_dir', None):
            os.environ['AFFECTNET_CACHE_DIR'] = args.affectnet_cache_dir
        # Honor --skip_dmd: set env for run_training()
        if getattr(args, 'skip_dmd', False):
            os.environ['SKIP_DMD'] = '1'
        if getattr(args, 'skip_generic', False):
            os.environ['SKIP_GENERIC'] = '1'
        if getattr(args, 'arcface_safe_start', False):
            os.environ['ARC_FACE_SAFE_START'] = '1'
        run_training()
        return
    for name, val in [('gen_csv_from_folders', None), ('extensions', '.jpg,.jpeg,.png,.bmp'), ('save_class_map', None), ('use_class_map', None), ('export_aligned_crops', None), ('export_csv_name', 'aligned_metadata.csv'), ('export_feature_extractor', None), ('extract_features_csv', None), ('extract_features_root', None), ('extract_features_out', None), ('load_backbone_ckpt', None), ('dmd_export_faces', None), ('dmd_json', None), ('dmd_modality', 'rgb'), ('dmd_fps', 2.0), ('dmd_max_frames', 300), ('dmd_color', 'auto'), ('dmd_camera_regex', None), ('dmd_exclude_boundary_s', 0.5), ('smooth_frames_csv', None), ('smooth_out', None), ('smooth_method', 'majority'), ('smooth_window_s', 0.5), ('smooth_alpha', 0.2), ('smooth_fps', None), ('gen_csv_from_rafdb', None), ('raf_split', 'train'), ('raf_image_subdir', None), ('affectnet_csv', None), ('affectnet_root', None), ('rafdb_csv', None), ('rafdb_root', None), ('dmd_csv', None), ('dmd_root', None), ('class_map_json', None), ('out_dir', './outputs')]:
        if not hasattr(args, name):
            setattr(args, name, val)

    def maybe(path: str) -> Optional[str]:
        return path if path and os.path.exists(path) else None
    autodiscovered = []
    if not getattr(args, 'affectnet_csv', None):
        p = maybe(os.path.join(os.getcwd(), 'affectnet_train.csv'))
        if p:
            setattr(args, 'affectnet_csv', p)
            autodiscovered.append(f'affectnet_csv={p}')
    if not getattr(args, 'affectnet_root', None):
        p = maybe(os.path.join(os.getcwd(), 'datasets', 'affectNet', 'Train'))
        if not p:
            p = maybe(os.path.join(os.getcwd(), 'datasets', 'affectNet'))
        if p:
            setattr(args, 'affectnet_root', p)
            autodiscovered.append(f'affectnet_root={p}')
    if not getattr(args, 'rafdb_csv', None):
        p = maybe(os.path.join(os.getcwd(), 'rafdb_train.csv'))
        if p:
            setattr(args, 'rafdb_csv', p)
            autodiscovered.append(f'rafdb_csv={p}')
    if not getattr(args, 'rafdb_root', None):
        raf_candidates = [os.path.join(os.getcwd(), 'datasets', 'raf-db', 'train'), os.path.join(os.getcwd(), 'datasets', 'raf-db', 'DATASET', 'train')]
        for cand in raf_candidates:
            p = maybe(cand)
            if p:
                setattr(args, 'rafdb_root', p)
                autodiscovered.append(f'rafdb_root={p}')
                break
    if not getattr(args, 'class_map_json', None):
        p = maybe(os.path.join(os.getcwd(), 'affectnet_map.json'))
        if p:
            setattr(args, 'class_map_json', p)
            autodiscovered.append(f'class_map_json={p}')
    if autodiscovered:
        print('Autodiscovered defaults:')
        for item in autodiscovered:
            print('  ', item)
    eval_autodiscovered = []
    if not getattr(args, 'affectnet_eval_csv', None):
        p = maybe(os.path.join(os.getcwd(), 'affectnet_test.csv'))
        if p:
            setattr(args, 'affectnet_eval_csv', p)
            eval_autodiscovered.append(f'affectnet_eval_csv={p}')
    if not getattr(args, 'rafdb_eval_csv', None):
        p = maybe(os.path.join(os.getcwd(), 'rafdb_test.csv'))
        if p:
            setattr(args, 'rafdb_eval_csv', p)
            eval_autodiscovered.append(f'rafdb_eval_csv={p}')
    if not getattr(args, 'affectnet_eval_root', None) and getattr(args, 'affectnet_eval_csv', None):
        p = maybe(os.path.join(os.getcwd(), 'datasets', 'affectNet', 'Test'))
        if p:
            setattr(args, 'affectnet_eval_root', p)
            eval_autodiscovered.append(f'affectnet_eval_root={p}')
    if not getattr(args, 'rafdb_eval_root', None) and getattr(args, 'rafdb_eval_csv', None):
        raf_candidates = [os.path.join(os.getcwd(), 'datasets', 'raf-db', 'test'), os.path.join(os.getcwd(), 'datasets', 'raf-db', 'DATASET', 'test')]
        for cand in raf_candidates:
            p = maybe(cand)
            if p:
                setattr(args, 'rafdb_eval_root', p)
                eval_autodiscovered.append(f'rafdb_eval_root={p}')
                break
    if eval_autodiscovered:
        print('Autodiscovered eval sets:')
        for item in eval_autodiscovered:
            print('  ', item)
        if args.gen_csv_from_rafdb:
            root_dir, out_csv = args.gen_csv_from_rafdb
            img_subdir = args.raf_image_subdir
            gen_csv_from_rafdb(root_dir, out_csv, split=args.raf_split, image_subdir=img_subdir)
            print(f'Wrote RAF-DB CSV: {out_csv}')
            return
    # If requested, run AffectNet cleaner to generate cleaned CSVs and mapping
    if getattr(args, 'affectnet_labels_csv', None):
        labels_csv = args.affectnet_labels_csv
        images_root = getattr(args, 'affectnet_root', None) or os.path.join(os.getcwd(), 'datasets', 'affectNet')
        cache_dir = args.affectnet_cache_dir or os.path.join(os.getcwd(), 'outputs', 'affectnet_cache')
        seven = bool(getattr(args, 'affectnet_7class', False))
        cleaned_csv, mapping, uniq, counts = affectnet_clean(labels_csv, images_root, cache_dir=cache_dir, seven_class=seven)
        setattr(args, 'affectnet_csv', cleaned_csv)
        setattr(args, 'affectnet_root', images_root)
        # Also process test if provided
        if getattr(args, 'affectnet_labels_csv_test', None):
            labels_csv_t = args.affectnet_labels_csv_test
            if os.path.exists(labels_csv_t):
                try:
                    cleaned_csv_t, _, _, _ = affectnet_clean(labels_csv_t, images_root, cache_dir=cache_dir, seven_class=seven)
                    setattr(args, 'affectnet_eval_csv', cleaned_csv_t)
                    setattr(args, 'affectnet_eval_root', images_root)
                except Exception as e:
                    print(f'AffectNet test cleaning skipped: {e}')

    cfg = TrainConfig(
        affectnet_csv=args.affectnet_csv,
        affectnet_root=args.affectnet_root,
        rafdb_csv=args.rafdb_csv,
        rafdb_root=args.rafdb_root,
        dmd_csv=args.dmd_csv,
        dmd_root=args.dmd_root,
        class_map_json=args.class_map_json,
        num_classes=(7 if getattr(args, 'affectnet_7class', False) else 8),
        eff_variant='b0',
        img_size=112,
        batch_size=64,
        lr=3e-4,
        weight_decay=1e-4,
        epochs_pretrain=5,
        epochs_main=15,
        seed=42,
        out_dir='./outputs',
        amp=True,
        balance_sampling=True,
    )
    seed_everything(cfg.seed)
    device = torch_device()
    try:
        if isinstance(device, torch.device) and device.type == 'cuda':
            num = torch.cuda.device_count()
            names = [torch.cuda.get_device_name(i) for i in range(num)]
            print(f'Using device: cuda (GPUs={num}) -> {names}')
        elif isinstance(device, torch.device) and device.type == 'mps':
            print('Using device: mps (Apple Silicon)')
        elif isinstance(device, torch.device):
            print(f'Using device: {device}')
        else:
            print(f'Using device object: {device}')
    except Exception:
        print(f'Using device: {device}')
    # Propagate AffectNet flags via env for build_loaders
    if getattr(args, 'affectnet_7class', False):
        os.environ['AFFECTNET_7CLASS'] = '1'
    if getattr(args, 'affectnet_cache_dir', None):
        os.environ['AFFECTNET_CACHE_DIR'] = args.affectnet_cache_dir
    pretrain_train_loader, pretrain_val_loader, dmd_train_loader, dmd_val_loader = build_loaders(cfg, device)

    def peek(loader, name: str, max_batches: int):
        if loader is None:
            return
        print(f'\nInspecting loader: {name}')
        bcount = 0
        t0 = time.time()
        for images, labels in loader:
            bcount += 1
            print(f'  batch {bcount}: images={tuple(images.shape)} dtype={images.dtype} min={float(images.min()):.3f} max={float(images.max()):.3f}')
            try:
                labs = labels.detach().cpu().numpy().tolist()
                c = Counter(labs)
                print(f'  labels: {dict(c)}')
            except Exception:
                pass
            if bcount >= max_batches:
                break
        t1 = time.time()
        print(f'  iterated {bcount} batch(es) in {t1 - t0:.2f}s')
    peek(pretrain_train_loader, 'pretrain_train', 1)
    peek(pretrain_val_loader, 'pretrain_val', 1)
    if dmd_train_loader is not None:
        peek(dmd_train_loader, 'dmd_train', 1)
    if dmd_val_loader is not None:
        peek(dmd_val_loader, 'dmd_val', 1)
    print('\nData loading and preprocessing check complete.')
    if args.train:
        train_loader = pretrain_train_loader
        val_loader = pretrain_val_loader
        if train_loader is None or val_loader is None:
            raise RuntimeError('Training requested but train/val loaders are unavailable.')
        device = torch_device()
        backbone = EfficientNetFeatureExtractor('b0', pretrained=True).to(device)

        def make_head(in_dim: int, num_classes: int) -> nn.Module:
            return ArcFaceHead(in_dim, num_classes)
        head_name: str = 'arcface'

        def find_latest_ckpt(out_dir: str, preferred: str) -> Optional[str]:
            try:
                cand1 = os.path.join(out_dir, f'last_{preferred}.pt')
                if os.path.isfile(cand1):
                    return cand1
                cand2 = os.path.join(out_dir, f'best_{preferred}.pt')
                if os.path.isfile(cand2):
                    return cand2
                pts = [os.path.join(out_dir, f) for f in os.listdir(out_dir) if f.endswith('.pt')]
                if pts:
                    pts.sort(key=lambda p: os.path.getmtime(p), reverse=True)
                    return pts[0]
            except Exception:
                pass
            return None
        start_epoch = 0
        best_acc = 0.0
        optimizer = None
        auto_ckpt_path = find_latest_ckpt(cfg.out_dir, head_name)
        ckpt = None
        if auto_ckpt_path is not None:
            try:
                ckpt = torch.load(auto_ckpt_path, map_location=device)
                saved_head = ckpt.get('head_type')
                if isinstance(saved_head, str) and saved_head != 'arcface':
                    print("Auto-resume: checkpoint used non-arcface head; forcing 'arcface'.")
            except Exception:
                ckpt = None
        head = make_head(cast(int, cast(Any, backbone).feature_dim), cfg.num_classes).to(device)
        if isinstance(ckpt, dict):
            bstate = ckpt.get('backbone_state') or ckpt.get('backbone') or ckpt.get('state_dict')
            hstate = ckpt.get('head_state') or ckpt.get('head')
            if isinstance(bstate, dict):
                backbone.load_state_dict(bstate, strict=False)
            if isinstance(hstate, dict):
                head.load_state_dict(hstate, strict=False)
            start_epoch = int(ckpt.get('epoch', 0))
            best_acc = float(ckpt.get('best_acc', 0.0))
        params = list(backbone.parameters()) + list(head.parameters())
        optimizer = torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
        if isinstance(ckpt, dict):
            try:
                opt_state = ckpt.get('optimizer', None)
                if isinstance(opt_state, dict):
                    optimizer.load_state_dict(opt_state)
            except Exception:
                pass
        criterion = nn.CrossEntropyLoss()

        def run_epoch(loader, train_mode: bool) -> Tuple[float, float]:
            if train_mode:
                backbone.train()
                head.train()
            else:
                backbone.eval()
                head.eval()
            total_loss = 0.0
            total_correct = 0
            total = 0
            for images, labels in progress(loader, total=len(loader), unit='batch', desc='train' if train_mode else 'val', leave=False):
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                with torch.set_grad_enabled(train_mode):
                    feats = backbone(images)
                    # label-conditioned logits for loss
                    logits_train = head(feats, labels)
                    loss = criterion(logits_train, labels)
                if train_mode:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                total_loss += float(loss.detach().cpu().item()) * images.size(0)
                # label-free logits for prediction/metrics
                logits_pred = head(feats, None)
                preds = torch.argmax(logits_pred, dim=1)
                total_correct += int((preds == labels).sum().detach().cpu().item())
                total += int(images.size(0))
            avg_loss = total_loss / max(1, total)
            acc = total_correct / max(1, total)
            return (avg_loss, acc)
        epochs = int(cfg.epochs_main)
        os.makedirs(cfg.out_dir, exist_ok=True)
        best_path = os.path.join(cfg.out_dir, f'best_{head_name}.pt')
        last_path = os.path.join(cfg.out_dir, f'last_{head_name}.pt')
        for epoch in range(start_epoch, epochs):
            tr_loss, tr_acc = run_epoch(train_loader, True)
            va_loss, va_acc = run_epoch(val_loader, False)
            print(f'Epoch {epoch + 1}/{epochs}  train loss {tr_loss:.4f} acc {tr_acc:.4f} | val loss {va_loss:.4f} acc {va_acc:.4f}')
            torch.save({'head_type': 'arcface', 'epoch': epoch + 1, 'best_acc': max(best_acc, va_acc), 'backbone_state': backbone.state_dict(), 'head_state': head.state_dict(), 'eff_variant': 'b0', 'feature_dim': backbone.feature_dim, 'optimizer': optimizer.state_dict()}, last_path)
            if va_acc > best_acc:
                best_acc = va_acc
                torch.save({'head_type': 'arcface', 'epoch': epoch + 1, 'best_acc': best_acc, 'backbone_state': backbone.state_dict(), 'head_state': head.state_dict(), 'eff_variant': 'b0', 'feature_dim': backbone.feature_dim}, best_path)
        print(f'Training complete. Best val acc {best_acc:.4f}. Checkpoints saved to {cfg.out_dir}')
if __name__ == '__main__':
    main()
