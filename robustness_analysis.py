import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# ----------------------------
# Configuration
# ----------------------------
IMAGE_SIZE = (200, 200)  # 200x200 -> 40000 features
RANDOM_STATE = 42
TEST_SIZE = 0.2

ORIGINAL_DIRS = {
    "word": Path("word_pdfs_png"),
    "google": Path("google_docs_pdfs_png"),
}

AUGMENTED_DIRS = {
    "gaussian_noise": {
        "word": Path("augmented_images/word/gaussian_noise"),
        "google": Path("augmented_images/google/gaussian_noise"),
    },
    "jpeg_compression": {
        "word": Path("augmented_images/word/jpeg_compression"),
        "google": Path("augmented_images/google/jpeg_compression"),
    },
    "dpi_downsampling": {
        "word": Path("augmented_images/word/dpi_downsampling"),
        "google": Path("augmented_images/google/dpi_downsampling"),
    },
    "random_crop": {
        "word": Path("augmented_images/word/random_crop"),
        "google": Path("augmented_images/google/random_crop"),
    },
    "bit_depth_reduction": {
        "word": Path("augmented_images/word/bit_depth_reduction"),
        "google": Path("augmented_images/google/bit_depth_reduction"),
    },
}

RESULTS_DIR = Path("results")
CONFUSION_DIR = RESULTS_DIR / "confusion_matrices"
PLOTS_DIR = RESULTS_DIR / "robustness_plots"
METRICS_CSV = RESULTS_DIR / "robustness_metrics.csv"


# ----------------------------
# Utilities
# ----------------------------
def ensure_dirs():
    RESULTS_DIR.mkdir(exist_ok=True)
    CONFUSION_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def load_image_as_vector(path: Path, image_size=IMAGE_SIZE) -> np.ndarray:
    img = Image.open(path).convert("L")
    img = img.resize(image_size)
    arr = np.array(img, dtype=np.float32)
    return arr.flatten()


def get_original_file_map(folder: Path) -> dict:
    """
    Maps stem -> file path for original images.
    Example: DNA -> word_pdfs_png/DNA.png
    """
    mapping = {}
    for file in sorted(folder.glob("*.png")):
        mapping[file.stem] = file
    return mapping


def get_augmented_file_map(folder: Path) -> dict:
    """
    Maps original stem -> augmented file path.
    Example: DNA -> augmented_images/word/gaussian_noise/DNA__gaussian_noise.png
    """
    mapping = {}
    for file in sorted(folder.glob("*.png")):
        stem = file.stem
        if "__" in stem:
            original_stem = stem.split("__")[0]
            mapping[original_stem] = file
    return mapping


def build_original_dataset():
    X = []
    y = []
    ids = []

    for class_name, folder in ORIGINAL_DIRS.items():
        label = 0 if class_name == "word" else 1
        for file in sorted(folder.glob("*.png")):
            X.append(load_image_as_vector(file))
            y.append(label)
            ids.append(file.stem)

    X = np.array(X)
    y = np.array(y)
    ids = np.array(ids)
    return X, y, ids


def build_augmented_dataset(augmentation_name: str, test_ids: np.ndarray, test_labels: np.ndarray):
    X_aug = []
    y_aug = []

    aug_maps = {
        "word": get_augmented_file_map(AUGMENTED_DIRS[augmentation_name]["word"]),
        "google": get_augmented_file_map(AUGMENTED_DIRS[augmentation_name]["google"]),
    }

    for original_id, label in zip(test_ids, test_labels):
        class_name = "word" if label == 0 else "google"
        aug_path = aug_maps[class_name].get(original_id)

        if aug_path is None:
            raise FileNotFoundError(
                f"Missing augmented file for original '{original_id}' "
                f"in augmentation '{augmentation_name}' for class '{class_name}'"
            )

        X_aug.append(load_image_as_vector(aug_path))
        y_aug.append(label)

    return np.array(X_aug), np.array(y_aug)


def plot_confusion_matrix(cm, class_names, title, save_path):
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title=title,
    )

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def evaluate_model(model_name, model, X_eval, y_eval, condition, metrics_rows):
    y_pred = model.predict(X_eval)
    acc = accuracy_score(y_eval, y_pred)
    cm = confusion_matrix(y_eval, y_pred)

    print(f"\n[{model_name}] Condition: {condition}")
    print(f"Accuracy: {acc:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print("Classification Report:")
    print(classification_report(y_eval, y_pred, target_names=["Word", "Google"]))

    metrics_rows.append({
        "model": model_name,
        "condition": condition,
        "accuracy": acc,
    })

    cm_path = CONFUSION_DIR / f"{model_name}_{condition}_confusion_matrix.png"
    plot_confusion_matrix(
        cm,
        ["Word", "Google"],
        f"{model_name} - {condition}",
        cm_path,
    )

    return acc


# ----------------------------
# Main pipeline
# ----------------------------
def main():
    ensure_dirs()

    print("Loading original dataset...")
    X, y, ids = build_original_dataset()
    print(f"Dataset shape: {X.shape}")
    print(f"Labels shape: {y.shape}")

    X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
        X, y, ids,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "SVM": SVC(kernel="rbf", random_state=RANDOM_STATE),
        "SGD": SGDClassifier(loss="hinge", random_state=RANDOM_STATE, max_iter=1000, tol=1e-3),
    }

    metrics_rows = []
    baseline_accuracies = {}

    print("\n=== TRAINING ON ORIGINAL DATA ONLY ===")
    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")
        model.fit(X_train_scaled, y_train)

        baseline_acc = evaluate_model(
            model_name=model_name,
            model=model,
            X_eval=X_test_scaled,
            y_eval=y_test,
            condition="original",
            metrics_rows=metrics_rows,
        )
        baseline_accuracies[model_name] = baseline_acc

        for aug_name in AUGMENTED_DIRS.keys():
            X_aug, y_aug = build_augmented_dataset(aug_name, ids_test, y_test)
            X_aug_scaled = scaler.transform(X_aug)

            evaluate_model(
                model_name=model_name,
                model=model,
                X_eval=X_aug_scaled,
                y_eval=y_aug,
                condition=aug_name,
                metrics_rows=metrics_rows,
            )

    df = pd.DataFrame(metrics_rows)
    df["baseline_accuracy"] = df["model"].map(baseline_accuracies)
    df["accuracy_drop"] = df["baseline_accuracy"] - df["accuracy"]
    df.to_csv(METRICS_CSV, index=False)

    print(f"\nSaved metrics to: {METRICS_CSV}")

    # Robustness plot per model
    for model_name in df["model"].unique():
        model_df = df[df["model"] == model_name].copy()

        ordered_conditions = [
            "original",
            "gaussian_noise",
            "jpeg_compression",
            "dpi_downsampling",
            "random_crop",
            "bit_depth_reduction",
        ]
        model_df["condition"] = pd.Categorical(model_df["condition"], categories=ordered_conditions, ordered=True)
        model_df = model_df.sort_values("condition")

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(model_df["condition"], model_df["accuracy"], marker="o")
        ax.set_title(f"{model_name} Robustness Across Augmentations")
        ax.set_xlabel("Condition")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1.05)
        plt.xticks(rotation=30)
        plt.tight_layout()

        plot_path = PLOTS_DIR / f"{model_name}_robustness_curve.png"
        plt.savefig(plot_path, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved plot: {plot_path}")

    print("\n=== SUMMARY ===")
    print(df)

    # Largest degradation
    aug_only = df[df["condition"] != "original"].copy()
    worst = aug_only.sort_values(["model", "accuracy_drop"], ascending=[True, False])
    print("\nLargest degradations by model:")
    print(worst.groupby("model").first()[["condition", "accuracy", "accuracy_drop"]])


if __name__ == "__main__":
    main()