"""Leaf/background masking utilities for shortcut diagnostics.

Masks are produced by rembg (U2Net) and cached on disk.
MaskedDataset wraps raw (path, label) samples with per-image masking.
"""

import io
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm


MASK_THRESHOLD = 128  # binarize rembg alpha at this level
FILL_MODES = ("black", "random_color", "noise")


# ── rembg helpers ──

def get_rembg_session(model_name: str = "u2netp"):
    """Lazily import rembg so modules can be parsed without the dep installed."""
    from rembg import new_session
    return new_session(model_name)


def compute_mask(img: Image.Image, session) -> Image.Image:
    """Return a binary ('L') mask the same size as img: 255=leaf, 0=background."""
    from rembg import remove
    out = remove(img, session=session, only_mask=True)
    if isinstance(out, (bytes, bytearray)):
        out = Image.open(io.BytesIO(out))
    return out.convert("L").point(lambda v: 255 if v >= MASK_THRESHOLD else 0)


def ensure_masks(samples: List[Tuple[str, int]], masks_dir: Path,
                 model_name: str = "u2netp"):
    """Compute masks for (path, label) samples, cached under masks_dir/<class>/<name>.png."""
    missing = []
    for path, _ in samples:
        p = Path(path)
        cache = masks_dir / p.parent.name / (p.stem + ".png")
        if not cache.exists():
            missing.append((p, cache))

    if not missing:
        print(f"All {len(samples)} masks already cached at {masks_dir}")
        return

    print(f"Computing {len(missing)} masks with rembg (cached ones skipped)...")
    session = get_rembg_session(model_name)
    for src, dst in tqdm(missing, desc="rembg"):
        dst.parent.mkdir(parents=True, exist_ok=True)
        img = Image.open(src).convert("RGB")
        mask = compute_mask(img, session)
        mask.save(dst)


# ── fill helpers ──

def _fill_region(arr: np.ndarray, region: np.ndarray, fill: str,
                 rng: np.random.Generator):
    """Fill `arr[region]` in-place using one of the FILL_MODES."""
    n = int(region.sum())
    if n == 0:
        return
    if fill == "black":
        arr[region] = 0
    elif fill == "random_color":
        color = rng.integers(0, 256, size=3, dtype=np.uint8)
        arr[region] = color
    elif fill == "noise":
        arr[region] = rng.integers(0, 256, size=(n, 3), dtype=np.uint8)
    else:
        raise ValueError(f"unknown fill mode: {fill}")


# ── dataset ──

class MaskedDataset(Dataset):
    """PlantVillage samples with per-image masking applied before transform.

    mode:
      'none'       — return original image
      'leaf'       — replace background (keep only leaf)
      'background' — replace leaf (keep only background)

    fill: 'black', 'random_color', or 'noise'.

    seed:
      int  — deterministic per-sample RNG seeded from (seed, idx). Use for eval.
      None — fresh RNG each access (different fill every epoch). Use for training.
    """

    def __init__(self, samples: List[Tuple[str, int]], masks_dir: Path,
                 transform, mode: str, fill: str = "black",
                 seed: Optional[int] = None):
        assert mode in {"none", "leaf", "background"}
        assert fill in FILL_MODES
        self.samples = samples
        self.masks_dir = masks_dir
        self.transform = transform
        self.mode = mode
        self.fill = fill
        self.seed = seed

    def __len__(self):
        return len(self.samples)

    @property
    def labels(self) -> List[int]:
        return [lbl for _, lbl in self.samples]

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")

        if self.mode != "none":
            p = Path(path)
            mask_path = self.masks_dir / p.parent.name / (p.stem + ".png")
            mask = Image.open(mask_path).convert("L")
            if mask.size != img.size:
                mask = mask.resize(img.size, Image.NEAREST)
            arr = np.array(img)
            m = (np.array(mask) >= MASK_THRESHOLD)  # True where leaf is
            region = ~m if self.mode == "leaf" else m
            if self.seed is not None:
                rng = np.random.default_rng((self.seed, idx))
            else:
                rng = np.random.default_rng()
            _fill_region(arr, region, self.fill, rng)
            img = Image.fromarray(arr)

        if self.transform:
            img = self.transform(img)
        return img, label
