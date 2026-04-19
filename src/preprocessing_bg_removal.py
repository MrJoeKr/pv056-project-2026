import cv2
import numpy as np
from PIL import Image
from rembg import remove, new_session

class PreprocessingBgRemoval:
    """Torchvision-compatible transform that applies the full preprocessing pipeline.

    Runs the entire pipeline in a single RGB→LAB→RGB round-trip (2 colour-space
    conversions) instead of one per step:
      1. Foreground mask from original RGB via rembg (no conversion needed)
      2. RGB → LAB  (one conversion)
      3. Shadow correction on L channel (Guided Filter)
      4. L-channel contrast stretch within the foreground mask
      5. Background zeroed in LAB (L=0, A=128, B=128 ≡ black)
      6. LAB → RGB  (one conversion)

    Accepts a PIL Image and returns a PIL Image.
    """

    def __call__(self, img: Image.Image) -> Image.Image:
        print(f"[Preprocessing] Processing image of size {img.size}") # TEMP
        rgb = np.array(img)  # PIL RGB → numpy RGB
        mask = _get_foreground_mask(rgb)
        mask_bool = mask > 0

        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)

        l = _shadow_correct_l(l)
        l = _stretch_l(l, mask_bool)

        # Zero background: black in LAB is (L=0, A=128, B=128)
        l[~mask_bool] = 0
        a[~mask_bool] = 128
        b[~mask_bool] = 128

        return Image.fromarray(cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB))


# ── Preprocessing functions ───────────────────────────────────────────────────
# Shared between 07_preprocessing.py and dataset transforms.

_session = None


def get_session():
    """Get or create a process-safe rembg (U2-Net) session."""
    global _session
    if _session is None:
        _session = new_session(
            "u2netp",
            providers=['CoreMLExecutionProvider', 'CPUExecutionProvider'],
        )
    return _session


def _get_foreground_mask(image: np.ndarray) -> np.ndarray:
    """Return a binary uint8 mask (255=foreground) via rembg U2-Net."""
    out = remove(image, session=get_session(), only_mask=True)
    return (out >= 128).astype(np.uint8)


def _shadow_correct_l(l: np.ndarray) -> np.ndarray:
    """Shadow-correct a uint8 LAB L channel using a Guided Filter."""
    l_float = l.astype(np.float32) / 255.0
    radius = int(max(l.shape) * 0.1)
    illum_est = cv2.ximgproc.guidedFilter(
        guide=l_float, src=l_float, radius=radius, eps=0.05
    )
    ambient = np.mean(illum_est)
    return np.clip(
        (l_float / (illum_est + 0.1)) * ambient * 255.0, 0, 255
    ).astype(np.uint8)


def _stretch_l(l: np.ndarray, mask_bool: np.ndarray) -> np.ndarray:
    """Linearly stretch a uint8 LAB L channel to [0, 255] within the foreground mask."""
    if not np.any(mask_bool):
        return l
    l_flat = l[mask_bool]
    l_min, l_max = l_flat.min(), l_flat.max()
    l_stretched = l.copy().astype(np.float32)
    l_stretched[mask_bool] = (
        (l_stretched[mask_bool] - l_min) / (l_max - l_min + 1e-6) * 255
    )
    return np.clip(l_stretched, 0, 255).astype(np.uint8)


def apply_shadow_correction(image: np.ndarray) -> np.ndarray:
    """Apply shadow correction using a Guided Filter on the LAB L channel.

    Estimates per-pixel illumination with an edge-preserving Guided Filter
    and normalises the L channel to remove shadow gradients.

    Args:
        image: RGB image as numpy array (H, W, 3)

    Returns:
        Shadow-corrected RGB image.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l = _shadow_correct_l(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)


def apply_background_removal_simple(
    image: np.ndarray, bg_color: tuple = (0, 0, 0)
) -> tuple:
    """Remove background using rembg (U2-Net) with simple binary thresholding.

    Args:
        image:    RGB image as numpy array (H, W, 3)
        bg_color: RGB tuple for the replacement background colour (default black)

    Returns:
        Tuple (masked_foreground, mask) where mask is a binary uint8 array
        with 255 for foreground pixels.
    """
    mask = _get_foreground_mask(image)
    result = image.copy()
    result[mask == 0] = bg_color
    return result, mask


def apply_l_channel_stretch(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Linearly stretch the LAB L channel to [0, 255] within the foreground mask.

    Args:
        image: RGB image as numpy array (H, W, 3)
        mask:  Binary mask where foreground pixels are 255

    Returns:
        RGB image with contrast-stretched L channel.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l = _stretch_l(l, mask > 0)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)