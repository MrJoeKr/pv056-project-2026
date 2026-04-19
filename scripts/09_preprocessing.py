"""Script to perform histogram equalization and background removal (MediaPipe Selfie Segmentation)
on 50 images from each class in the PlantVillage dataset.

Results are saved to:
- results/experiments/preprocessing/hist_equalization/
- results/experiments/preprocessing/bg_removal/
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import cv2
import numpy as np
from rembg import remove
import gc
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

from src.visualization import plot_preprocessing_comparison

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Config
from src.dataset import (
    get_session,
    apply_shadow_correction,
    apply_background_removal_simple,
    apply_l_channel_stretch,
)


def setup_output_dirs(config: Config):
    """Create output directories for preprocessing results."""
    # Create a unique timestamped folder for each run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = config.results_dir.parent / "experiments" / "preprocessing"
    sequential_dir = exp_dir / f"sequential-{timestamp}"
    sequential_dir.mkdir(parents=True, exist_ok=True)

    # New preprocessed data directory
    preprocessed_data_dir = config.project_root / "data" / "PlantVillage_preprocessed"
    preprocessed_data_dir.mkdir(parents=True, exist_ok=True)

    return sequential_dir, preprocessed_data_dir


def apply_background_removal(
    image: np.ndarray, bg_color: tuple = (0, 0, 0)
) -> tuple[np.ndarray, np.ndarray]:
    """Apply rembg (U2-Net) for background removal with post-processing.

    Args:
        image: RGB image as numpy array (H, W, 3)
        bg_color: RGB tuple for the background color (default black)

    Returns:
        tuple (masked_foreground, mask) where:
            masked_foreground: RGB image with background replaced
            mask: Binary mask where foreground is white (255)
    """
    # rembg handles the heavy lifting
    # alpha_matting=True can help with fine edges but is slower
    # bgcolor=(0,0,0,0) ensures black background
    output = remove(
        image,
        session=get_session(),
        alpha_matting=False,  # Keep it fast, we'll refine manually
    )

    # Extract mask from alpha channel
    mask = output[:, :, 3]

    # Refine mask: morphological operations to remove small artifacts
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.GaussianBlur(mask, (3, 3), 0)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # Create background image
    background = np.full(image.shape, bg_color, dtype=np.uint8)

    # Extract foreground and combine with background
    foreground = cv2.bitwise_and(image, image, mask=mask)
    background_masked = cv2.bitwise_and(
        background, background, mask=cv2.bitwise_not(mask)
    )
    masked_foreground = cv2.add(foreground, background_masked)

    return masked_foreground, mask


def process_single_image(img_info):
    """Worker function to process a single image.

    img_info: tuple (img_path, img_name, class_name, output_class_dir, preprocessed_class_dir, bg_color, save_composite)
    """
    (
        img_path,
        img_name,
        class_name,
        output_class_dir,
        preprocessed_class_dir,
        bg_color,
        save_composite,
    ) = img_info

    # Skip if already processed (resume support)
    processed_out_path = preprocessed_class_dir / img_name
    if processed_out_path.exists():
        return True

    try:
        image_bgr = cv2.imread(str(img_path))
        if image_bgr is None:
            return False
        original_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # VARIANT 1: Shadow Correction (Guided Filter)
        shadow_corrected = apply_shadow_correction(original_rgb)

        # VARIANT 6: Full: Shadow (Guided) + BG + Stretch
        final_v6_base, mask_v6 = apply_background_removal_simple(
            shadow_corrected, bg_color=bg_color
        )
        final_v6 = apply_l_channel_stretch(final_v6_base, mask_v6)

        # Save just the preprocessed image to preprocessed_data_dir
        cv2.imwrite(str(processed_out_path), cv2.cvtColor(final_v6, cv2.COLOR_RGB2BGR))

        if save_composite:
            plot_preprocessing_comparison(
                original_rgb,
                final_v6,
                output_class_dir / f"comparison_{img_name}",
            )

        # Explicitly delete large numpy arrays to free memory early
        del image_bgr, original_rgb, shadow_corrected, final_v6_base, mask_v6, final_v6
        if gc.isenabled():
            gc.collect()

        return True

    except Exception as e:
        print(f"Error {img_name}: {e}")
        return False


def process_images(
    config: Config,
    images_per_class: int = None,
    classes: list = None,
    bg_color: tuple = (0, 0, 0),
    save_composite: bool = True,
):
    """Process images with multiple sequential variants to find the best combination."""
    sequential_dir, preprocessed_data_dir = setup_output_dirs(config)

    print(f"Sequential output directory: {sequential_dir}")
    print(f"Preprocessed data directory: {preprocessed_data_dir}\n")
    print(f"Requested background color (RGB): {bg_color}")

    total_processed = 0
    target_classes = classes if classes else config.classes

    # Collect all image processing tasks
    tasks = []
    for class_name in target_classes:
        class_dir = config.data_dir / class_name
        if not class_dir.exists():
            continue

        output_class_dir = sequential_dir / class_name
        output_class_dir.mkdir(exist_ok=True)

        preprocessed_class_dir = preprocessed_data_dir / class_name
        preprocessed_class_dir.mkdir(exist_ok=True)

        image_files = [
            f
            for f in sorted(os.listdir(class_dir))
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        if images_per_class is not None:
            image_files = image_files[:images_per_class]

        for img_name in image_files:
            img_path = class_dir / img_name
            tasks.append(
                (
                    img_path,
                    img_name,
                    class_name,
                    output_class_dir,
                    preprocessed_class_dir,
                    bg_color,
                    save_composite,
                )
            )

    # Cap at 4 workers: each process loads its own rembg model into memory.
    # More workers = more model copies = memory explosion. 4 is a good balance
    # for Apple Silicon where CoreML (ANE) handles the heavy lifting.
    num_workers = min(4, max(1, multiprocessing.cpu_count() // 4))
    already_done = sum(
        1 for t in tasks if (t[4] / t[1]).exists()
    )  # t[4]=preprocessed_dir, t[1]=img_name
    print(f"Starting parallel processing with {num_workers} workers...")
    print(
        f"Total images to process: {len(tasks)} ({already_done} already done, skipping)"
    )
    print(f"Save comparison composites: {save_composite}")

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # chunksize > 1 reduces IPC overhead for many small tasks
        chunksize = min(32, max(1, len(tasks) // (num_workers * 4)))
        results = list(
            tqdm(
                executor.map(process_single_image, tasks, chunksize=chunksize),
                total=len(tasks),
            )
        )

        for success in results:
            if success:
                total_processed += 1

    print(f"\n✓ Preprocessing complete!")
    print(f"Total processed: {total_processed}")
    print(f"Results saved to: {sequential_dir}")


if __name__ == "__main__":
    config = Config()
    config.data_dir = config.project_root / "data" / "PlantVillage"
    config.results_dir = config.project_root / "results"

    # Verify data directory exists
    if not config.data_dir.exists():
        print(f"Error: Data directory not found: {config.data_dir}")
        exit(1)

    print("=" * 70)
    print("Plant Disease Image Preprocessing")
    print("=" * 70)
    print(f"Data directory: {config.data_dir}")
    print(f"Classes: {len(config.classes)}")
    print()

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--classes", nargs="+", help="Subset of classes to process")
    parser.add_argument(
        "--n", type=int, default=None, help="Images per class (default: all)"
    )
    parser.add_argument(
        "--bg",
        type=int,
        nargs=3,
        default=[0, 0, 0],
        help="Background color as RGB (e.g., 255 255 255)",
    )
    parser.add_argument(
        "--no-composite",
        action="store_true",
        help="Skip saving comparison composite images (faster, less I/O)",
    )
    args = parser.parse_args()

    process_images(
        config,
        images_per_class=args.n,
        classes=args.classes,
        bg_color=tuple(args.bg),
        save_composite=not args.no_composite,
    )
