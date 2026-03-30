import os
import csv
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image
import cv2


SOURCE_DIRS = {
    "word": "word_pdfs_png",
    "google": "google_docs_pdfs_png",
    "python": "python_pdfs_png",
}

OUTPUT_ROOT = Path("augmented_images")
METADATA_CSV = OUTPUT_ROOT / "augmentation_metadata.csv"

VALID_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_grayscale_image(path: Path) -> np.ndarray:
    img = Image.open(path).convert("L")
    return np.array(img, dtype=np.uint8)


def save_grayscale_image(img_array: np.ndarray, path: Path) -> None:
    Image.fromarray(img_array, mode="L").save(path)


def add_gaussian_noise(img: np.ndarray) -> tuple[np.ndarray, dict]:
    sigma = random.uniform(5, 20)
    noise = np.random.normal(0, sigma, img.shape)
    noisy = img.astype(np.float32) + noise
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy, {"sigma": round(sigma, 4)}


def apply_jpeg_compression(img: np.ndarray) -> tuple[np.ndarray, dict]:
    quality = random.randint(20, 80)
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    success, encoded = cv2.imencode(".jpg", img, encode_params)
    if not success:
        raise RuntimeError("JPEG encoding failed")
    decoded = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
    return decoded, {"jpeg_quality": quality}


def apply_dpi_downsampling(img: np.ndarray) -> tuple[np.ndarray, dict]:
    target_dpi = random.choice([150, 72])
    scale = target_dpi / 300.0

    h, w = img.shape
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    downsampled = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    restored = cv2.resize(downsampled, (w, h), interpolation=cv2.INTER_LINEAR)

    return restored, {
        "original_dpi": 300,
        "target_dpi": target_dpi,
        "scale_factor": round(scale, 4),
        "intermediate_size": [new_h, new_w],
    }


def apply_random_crop(img: np.ndarray) -> tuple[np.ndarray, dict]:
    h, w = img.shape

    top_pct = random.uniform(0.01, 0.03)
    bottom_pct = random.uniform(0.01, 0.03)
    left_pct = random.uniform(0.01, 0.03)
    right_pct = random.uniform(0.01, 0.03)

    top = max(1, int(round(h * top_pct)))
    bottom = max(1, int(round(h * bottom_pct)))
    left = max(1, int(round(w * left_pct)))
    right = max(1, int(round(w * right_pct)))

    # Prevent invalid crop
    if top + bottom >= h:
        bottom = max(1, h - top - 1)
    if left + right >= w:
        right = max(1, w - left - 1)

    cropped = img[top:h - bottom, left:w - right]
    restored = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)

    return restored, {
        "top_crop_pct": round(top_pct, 4),
        "bottom_crop_pct": round(bottom_pct, 4),
        "left_crop_pct": round(left_pct, 4),
        "right_crop_pct": round(right_pct, 4),
        "cropped_size": [int(cropped.shape[0]), int(cropped.shape[1])],
    }


def apply_bit_depth_reduction(img: np.ndarray) -> tuple[np.ndarray, dict]:
    levels = 16  # 4-bit grayscale
    step = 256 // levels  # 16
    reduced = (img // step) * step
    reduced = np.clip(reduced, 0, 255).astype(np.uint8)
    return reduced, {"bit_depth_original": 8, "bit_depth_reduced": 4, "levels": levels}


AUGMENTATIONS = {
    "gaussian_noise": add_gaussian_noise,
    "jpeg_compression": apply_jpeg_compression,
    "dpi_downsampling": apply_dpi_downsampling,
    "random_crop": apply_random_crop,
    "bit_depth_reduction": apply_bit_depth_reduction,
}


def get_image_paths(folder: str) -> list[Path]:
    path = Path(folder)
    if not path.exists():
        print(f"Warning: source folder not found: {folder}")
        return []
    return sorted([p for p in path.iterdir() if p.suffix.lower() in VALID_EXTENSIONS])


def process_class(class_name: str, source_folder: str, writer: csv.DictWriter) -> tuple[int, int]:
    image_paths = get_image_paths(source_folder)
    print(f"\nProcessing class '{class_name}' from {source_folder} ({len(image_paths)} images)...")

    created_count = 0
    original_count = len(image_paths)

    for idx, img_path in enumerate(image_paths, start=1):
        img = load_grayscale_image(img_path)
        stem = img_path.stem

        for aug_name, aug_fn in AUGMENTATIONS.items():
            aug_img, params = aug_fn(img.copy())

            output_dir = OUTPUT_ROOT / class_name / aug_name
            ensure_dir(output_dir)

            out_name = f"{stem}__{aug_name}.png"
            out_path = output_dir / out_name
            save_grayscale_image(aug_img, out_path)

            writer.writerow({
                "class_name": class_name,
                "source_folder": source_folder,
                "original_filename": img_path.name,
                "original_path": str(img_path),
                "augmentation_type": aug_name,
                "output_filename": out_name,
                "output_path": str(out_path),
                "parameters_json": json.dumps(params),
                "original_height": img.shape[0],
                "original_width": img.shape[1],
                "output_height": aug_img.shape[0],
                "output_width": aug_img.shape[1],
            })

            created_count += 1

        if idx % 50 == 0 or idx == len(image_paths):
            print(f"  Processed {idx}/{len(image_paths)} images")

    return original_count, created_count


def main() -> None:
    random.seed(42)
    np.random.seed(42)

    ensure_dir(OUTPUT_ROOT)

    with open(METADATA_CSV, "w", newline="") as f:
        fieldnames = [
            "class_name",
            "source_folder",
            "original_filename",
            "original_path",
            "augmentation_type",
            "output_filename",
            "output_path",
            "parameters_json",
            "original_height",
            "original_width",
            "output_height",
            "output_width",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        total_original = 0
        total_augmented = 0

        for class_name, source_folder in SOURCE_DIRS.items():
            original_count, created_count = process_class(class_name, source_folder, writer)
            total_original += original_count
            total_augmented += created_count

    print("\n=== AUGMENTATION COMPLETE ===")
    print(f"Original images found: {total_original}")
    print(f"Augmented images created: {total_augmented}")
    print(f"Expected augmented count: {total_original * 5}")
    print(f"Expected total dataset size including originals: {total_original * 6}")
    print(f"Metadata CSV: {METADATA_CSV}")

    if total_augmented == total_original * 5:
        print("SUCCESS: Exactly five augmentations per original image were created.")
    else:
        print("WARNING: Augmented count does not match expected 5x multiplier.")


if __name__ == "__main__":
    main()
