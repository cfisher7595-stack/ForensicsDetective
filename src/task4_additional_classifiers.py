from pathlib import Path
import time

import numpy as np
import pandas as pd
from PIL import Image

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


IMAGE_SIZE = (200, 200)
RANDOM_STATE = 42
TEST_SIZE = 0.2

ORIGINAL_DIRS = {
    "word": Path("word_pdfs_png"),
    "google": Path("google_docs_pdfs_png"),
}


def load_image_as_vector(path: Path, image_size=IMAGE_SIZE) -> np.ndarray:
    img = Image.open(path).convert("L")
    img = img.resize(image_size)
    arr = np.array(img, dtype=np.float32)
    return arr.flatten()


def build_dataset():
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


def main():
    print("Loading original dataset...")
    X, y, ids = build_dataset()
    print(f"Dataset shape: {X.shape}")
    print(f"Class distribution: Word={(y == 0).sum()}, Google={(y == 1).sum()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = []

    # -------------------------
    # Random Forest
    # -------------------------
    print("\n=== Tuning Random Forest ===")
    rf = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)

    rf_param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [None, 20, 40],
        "min_samples_split": [2, 5],
    }

    rf_search = GridSearchCV(
        rf,
        rf_param_grid,
        cv=3,
        scoring="accuracy",
        n_jobs=-1,
        verbose=1,
    )

    start = time.time()
    # RF does not require scaling, but using original X_train is better
    rf_search.fit(X_train, y_train)
    rf_time = time.time() - start

    best_rf = rf_search.best_estimator_
    rf_pred = best_rf.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_pred)

    print(f"Best RF Params: {rf_search.best_params_}")
    print(f"RF Training Time: {rf_time:.2f} sec")
    print(f"RF Accuracy: {rf_acc:.4f}")
    print("RF Confusion Matrix:")
    print(confusion_matrix(y_test, rf_pred))
    print("RF Classification Report:")
    print(classification_report(y_test, rf_pred, target_names=["Word", "Google"]))

    results.append({
        "model": "RandomForest",
        "best_params": str(rf_search.best_params_),
        "accuracy": rf_acc,
        "training_time_sec": round(rf_time, 2),
    })

    # -------------------------
    # KNN
    # -------------------------
    print("\n=== Tuning KNN ===")
    knn = KNeighborsClassifier()

    knn_param_grid = {
        "n_neighbors": [3, 5, 7],
        "weights": ["uniform", "distance"],
        "metric": ["euclidean", "manhattan"],
    }

    knn_search = GridSearchCV(
        knn,
        knn_param_grid,
        cv=3,
        scoring="accuracy",
        n_jobs=-1,
        verbose=1,
    )

    start = time.time()
    # KNN should use scaled features
    knn_search.fit(X_train_scaled, y_train)
    knn_time = time.time() - start

    best_knn = knn_search.best_estimator_
    knn_pred = best_knn.predict(X_test_scaled)
    knn_acc = accuracy_score(y_test, knn_pred)

    print(f"Best KNN Params: {knn_search.best_params_}")
    print(f"KNN Training Time: {knn_time:.2f} sec")
    print(f"KNN Accuracy: {knn_acc:.4f}")
    print("KNN Confusion Matrix:")
    print(confusion_matrix(y_test, knn_pred))
    print("KNN Classification Report:")
    print(classification_report(y_test, knn_pred, target_names=["Word", "Google"]))

    results.append({
        "model": "KNN",
        "best_params": str(knn_search.best_params_),
        "accuracy": knn_acc,
        "training_time_sec": round(knn_time, 2),
    })

    results_df = pd.DataFrame(results)
    output_path = Path("results") / "task4_additional_classifiers_results.csv"
    output_path.parent.mkdir(exist_ok=True, parents=True)
    results_df.to_csv(output_path, index=False)

    print("\n=== TASK 4 SUMMARY ===")
    print(results_df)
    print(f"\nSaved results to: {output_path}")


if __name__ == "__main__":
    main()
