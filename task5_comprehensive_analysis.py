from pathlib import Path
import time

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


# ----------------------------
# Configuration
# ----------------------------
IMAGE_SIZE = (200, 200)
RANDOM_STATE = 42
TEST_SIZE = 0.2
N_BOOTSTRAP = 1000

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
TASK5_DIR = RESULTS_DIR / "task5"
CONFUSION_DIR = TASK5_DIR / "confusion_matrices"
PLOTS_DIR = TASK5_DIR / "plots"
METRICS_CSV = TASK5_DIR / "task5_metrics.csv"
SIGNIFICANCE_CSV = TASK5_DIR / "task5_significance.csv"


# ----------------------------
# Utilities
# ----------------------------
def ensure_dirs():
    TASK5_DIR.mkdir(parents=True, exist_ok=True)
    CONFUSION_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def load_image_as_vector(path: Path, image_size=IMAGE_SIZE) -> np.ndarray:
    img = Image.open(path).convert("L")
    img = img.resize(image_size)
    arr = np.array(img, dtype=np.float32)
    return arr.flatten()


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

    return np.array(X), np.array(y), np.array(ids)


def get_augmented_file_map(folder: Path) -> dict:
    mapping = {}
    for file in sorted(folder.glob("*.png")):
        stem = file.stem
        if "__" in stem:
            original_stem = stem.split("__")[0]
            mapping[original_stem] = file
    return mapping


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
                f"Missing augmented file for {original_id} in {augmentation_name}/{class_name}"
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
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black"
            )

    fig.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def bootstrap_accuracy_ci(y_true, y_pred, n_bootstrap=N_BOOTSTRAP, seed=RANDOM_STATE):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    scores = []

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        score = accuracy_score(y_true[idx], y_pred[idx])
        scores.append(score)

    lower = np.percentile(scores, 2.5)
    upper = np.percentile(scores, 97.5)
    return lower, upper


def bootstrap_p_value(y_true, pred_a, pred_b, n_bootstrap=N_BOOTSTRAP, seed=RANDOM_STATE):
    """
    Simple paired bootstrap on accuracy differences.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    diffs = []

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        acc_a = accuracy_score(y_true[idx], pred_a[idx])
        acc_b = accuracy_score(y_true[idx], pred_b[idx])
        diffs.append(acc_a - acc_b)

    diffs = np.array(diffs)
    p_value = 2 * min(np.mean(diffs <= 0), np.mean(diffs >= 0))
    return float(min(p_value, 1.0))


def evaluate_condition(model_name, condition, y_true, y_pred, metrics_rows):
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=1, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred)
    ci_low, ci_high = bootstrap_accuracy_ci(y_true, y_pred)

    metrics_rows.append({
        "model": model_name,
        "condition": condition,
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy_ci_low": ci_low,
        "accuracy_ci_high": ci_high,
    })

    cm_path = CONFUSION_DIR / f"{model_name}_{condition}_confusion_matrix.png"
    plot_confusion_matrix(cm, ["Word", "Google"], f"{model_name} - {condition}", cm_path)


def main():
    ensure_dirs()

    print("Loading original dataset...")
    X, y, ids = build_original_dataset()
    print(f"Dataset shape: {X.shape}")

    X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
        X, y, ids,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    augmented_data = {}
    for aug_name in AUGMENTED_DIRS.keys():
        X_aug, y_aug = build_augmented_dataset(aug_name, ids_test, y_test)
        augmented_data[aug_name] = (X_aug, y_aug, scaler.transform(X_aug))

    models = {
        "SVM": {
            "model": SVC(kernel="rbf", random_state=RANDOM_STATE),
            "use_scaled": True,
        },
        "SGD": {
            "model": SGDClassifier(loss="hinge", random_state=RANDOM_STATE, max_iter=1000, tol=1e-3),
            "use_scaled": True,
        },
        "RandomForest": {
            "model": RandomForestClassifier(
                n_estimators=100,
                max_depth=None,
                min_samples_split=2,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            "use_scaled": False,
        },
        "KNN": {
            "model": KNeighborsClassifier(
                metric="euclidean",
                n_neighbors=3,
                weights="uniform",
            ),
            "use_scaled": True,
        },
    }

    metrics_rows = []
    prediction_store = {}

    print("\n=== TRAINING AND EVALUATING ALL MODELS ===")
    for model_name, config in models.items():
        model = config["model"]
        use_scaled = config["use_scaled"]

        print(f"\nTraining {model_name}...")
        start = time.time()
        if use_scaled:
            model.fit(X_train_scaled, y_train)
            train_time = time.time() - start
            print(f"{model_name} trained in {train_time:.2f} sec")

            # Original
            pred_orig = model.predict(X_test_scaled)
            evaluate_condition(model_name, "original", y_test, pred_orig, metrics_rows)
            prediction_store[(model_name, "original")] = pred_orig

            # Augmented
            for aug_name, (_, y_aug, X_aug_scaled) in augmented_data.items():
                pred_aug = model.predict(X_aug_scaled)
                evaluate_condition(model_name, aug_name, y_aug, pred_aug, metrics_rows)
                prediction_store[(model_name, aug_name)] = pred_aug
        else:
            model.fit(X_train, y_train)
            train_time = time.time() - start
            print(f"{model_name} trained in {train_time:.2f} sec")

            pred_orig = model.predict(X_test)
            evaluate_condition(model_name, "original", y_test, pred_orig, metrics_rows)
            prediction_store[(model_name, "original")] = pred_orig

            for aug_name, (X_aug, y_aug, _) in augmented_data.items():
                pred_aug = model.predict(X_aug)
                evaluate_condition(model_name, aug_name, y_aug, pred_aug, metrics_rows)
                prediction_store[(model_name, aug_name)] = pred_aug

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(METRICS_CSV, index=False)
    print(f"\nSaved metrics to: {METRICS_CSV}")

    # Add accuracy drop
    baseline_map = (
        metrics_df[metrics_df["condition"] == "original"]
        .set_index("model")["accuracy"]
        .to_dict()
    )
    metrics_df["baseline_accuracy"] = metrics_df["model"].map(baseline_map)
    metrics_df["accuracy_drop"] = metrics_df["baseline_accuracy"] - metrics_df["accuracy"]
    metrics_df.to_csv(METRICS_CSV, index=False)

    # ----------------------------
    # Significance testing
    # Compare models on original condition
    # ----------------------------
    significance_rows = []
    model_names = list(models.keys())

    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            m1 = model_names[i]
            m2 = model_names[j]

            pred1 = prediction_store[(m1, "original")]
            pred2 = prediction_store[(m2, "original")]

            p_val = bootstrap_p_value(y_test, pred1, pred2)

            acc1 = accuracy_score(y_test, pred1)
            acc2 = accuracy_score(y_test, pred2)

            significance_rows.append({
                "model_a": m1,
                "model_b": m2,
                "condition": "original",
                "accuracy_a": acc1,
                "accuracy_b": acc2,
                "accuracy_diff": acc1 - acc2,
                "bootstrap_p_value": p_val,
            })

    significance_df = pd.DataFrame(significance_rows)
    significance_df.to_csv(SIGNIFICANCE_CSV, index=False)
    print(f"Saved significance tests to: {SIGNIFICANCE_CSV}")

    # ----------------------------
    # Plot 1: robustness curves
    # ----------------------------
    ordered_conditions = [
        "original",
        "gaussian_noise",
        "jpeg_compression",
        "dpi_downsampling",
        "random_crop",
        "bit_depth_reduction",
    ]

    fig, ax = plt.subplots(figsize=(9, 5))
    for model_name in metrics_df["model"].unique():
        model_df = metrics_df[metrics_df["model"] == model_name].copy()
        model_df["condition"] = pd.Categorical(
            model_df["condition"],
            categories=ordered_conditions,
            ordered=True,
        )
        model_df = model_df.sort_values("condition")
        ax.plot(model_df["condition"], model_df["accuracy"], marker="o", label=model_name)

    ax.set_title("Robustness Curves for All Models")
    ax.set_xlabel("Condition")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.legend()
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "all_models_robustness_curves.png", bbox_inches="tight")
    plt.close(fig)

    # ----------------------------
    # Plot 2: bar chart of original accuracy
    # ----------------------------
    orig_df = metrics_df[metrics_df["condition"] == "original"].copy()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(orig_df["model"], orig_df["accuracy"])
    ax.set_title("Original Test Accuracy by Model")
    ax.set_xlabel("Model")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "original_accuracy_comparison.png", bbox_inches="tight")
    plt.close(fig)

    print("\n=== TASK 5 SUMMARY ===")
    print(metrics_df)
    print("\n=== SIGNIFICANCE TESTS ===")
    print(significance_df)


if __name__ == "__main__":
    main()
