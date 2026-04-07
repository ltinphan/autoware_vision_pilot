"""
AutoDrive ZOD dataset loader.

Directory layout expected:
    {zod_root}/labels/{seq}/*.json          — one JSON label per frame
    {zod_root}/images_blur_*/sequences/{seq}/camera_front_blur/
                                            — camera images (resolved via zod_utils)

Each JSON label must contain at minimum:
    "image"                        : image filename
    "cipo_detected"                : bool  (CIPO presence flag)
    "distance_to_in_path_object"   : float (metres, ≥0)
    "curvature"                    : float (1/m)

Sequential pairs (T-1, T) are built within each sequence only — no cross-sequence
pairing so temporal ordering is always valid.

Split is done at sequence level (no frame-level leakage):
    85 % → train   10 % → val   5 % → test
"""

import json
import os
import sys
from pathlib import Path

import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset

sys.path.append(str(Path(__file__).resolve().parents[2]))
from Models.data_parsing.AutoDrive.zod.zod_utils import get_images_blur_dir

# ImageNet normalisation (same as the rest of the codebase)
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]

# Network input resolution
_NET_W = 1024
_NET_H = 512

# Distance clamp used when normalising GT
_D_MAX = 150.0


def _default_transform():
    return T.Compose([
        T.Resize((_NET_H, _NET_W)),
        T.ToTensor(),
        T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])


def _norm_distance(d_metres: float) -> float:
    """GT normalisation: d_norm = (150 - min(d, 150)) / 150."""
    d = min(d_metres, _D_MAX)
    return (_D_MAX - d) / _D_MAX


class AutoDriveDataset(Dataset):
    """
    PyTorch Dataset for AutoDrive.

    Returns per item:
        img_prev   : (3, H, W) float tensor — frame T-1
        img_curr   : (3, H, W) float tensor — frame T
        d_norm     : float  — normalised distance GT ∈ [0, 1]
        curvature  : float  — curvature GT in 1/m (raw)
        flag       : float  — CIPO flag GT  {0.0, 1.0}
    """

    def __init__(self, zod_root: str, sequences: list[str], transform=None):
        self.zod_root  = Path(zod_root)
        self.transform = transform or _default_transform()
        self.pairs: list[tuple] = []

        for seq in sequences:
            label_dir = self.zod_root / "labels" / seq
            if not label_dir.exists():
                continue

            label_files = sorted(label_dir.glob("*.json"))
            if len(label_files) < 2:
                continue

            labels = []
            for lf in label_files:
                with open(lf) as fh:
                    data = json.load(fh)
                labels.append(data)

            img_dir = get_images_blur_dir(self.zod_root, seq)

            for i in range(1, len(labels)):
                lbl_prev = labels[i - 1]
                lbl_curr = labels[i]

                img_prev_path = img_dir / lbl_prev["image"]
                img_curr_path = img_dir / lbl_curr["image"]

                if not img_prev_path.exists() or not img_curr_path.exists():
                    continue

                d_norm    = _norm_distance(lbl_curr["distance_to_in_path_object"])
                curvature = float(lbl_curr["curvature"])
                flag      = 1.0 if lbl_curr["cipo_detected"] else 0.0

                self.pairs.append((
                    str(img_prev_path),
                    str(img_curr_path),
                    d_norm,
                    curvature,
                    flag,
                ))

        print(f"AutoDriveDataset: {len(self.pairs)} pairs from {len(sequences)} sequences.")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_prev_path, img_curr_path, d_norm, curvature, flag = self.pairs[idx]

        img_prev = Image.open(img_prev_path).convert("RGB")
        img_curr = Image.open(img_curr_path).convert("RGB")

        img_prev = self.transform(img_prev)
        img_curr = self.transform(img_curr)

        return img_prev, img_curr, d_norm, curvature, flag


class LoadDataAutoDrive:
    """
    Scans {zod_root}/labels/ for all sequences and produces train / val / test
    Dataset objects split at the sequence level (85 / 10 / 5).

    Usage:
        loader = LoadDataAutoDrive("/path/to/zod")
        loader.train  →  AutoDriveDataset
        loader.val    →  AutoDriveDataset
        loader.test   →  AutoDriveDataset
    """

    TRAIN_FRAC = 0.85
    VAL_FRAC   = 0.10
    # TEST_FRAC  = 0.05  (remainder)

    def __init__(self, zod_root: str, transform=None, seed: int = 42):
        zod_root   = Path(zod_root)
        labels_dir = zod_root / "labels"

        all_seqs = sorted([
            d.name for d in labels_dir.iterdir()
            if d.is_dir()
        ])

        if not all_seqs:
            raise FileNotFoundError(f"No sequence folders found under {labels_dir}")

        n          = len(all_seqs)
        n_train    = max(1, round(n * self.TRAIN_FRAC))
        n_val      = max(1, round(n * self.VAL_FRAC))

        train_seqs = all_seqs[:n_train]
        val_seqs   = all_seqs[n_train : n_train + n_val]
        test_seqs  = all_seqs[n_train + n_val :]

        print(f"Sequences — train: {len(train_seqs)}  val: {len(val_seqs)}  test: {len(test_seqs)}")

        self.train = AutoDriveDataset(zod_root, train_seqs, transform)
        self.val   = AutoDriveDataset(zod_root, val_seqs,   transform)
        self.test  = AutoDriveDataset(zod_root, test_seqs,  transform)
