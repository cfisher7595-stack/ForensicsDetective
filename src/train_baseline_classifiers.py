#!/usr/bin/env python3
"""
Baseline PDF Provenance Classifiers

Trains simple machine learning models (SVM, SGD) on flattened binary image data
to establish baseline performance for PDF creator identification.
"""

import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import pickle
import time

def load_dataset(word_dir='word_pdfs_png', google_dir='google_docs_pdfs_png', 
                max_samples_per_class=None, target_size=(200, 200)):
    """
    Load binary image dataset and create feature vectors.
    
    Args:
        word_dir (str): Directory containing Word-generated PDF images
        google_dir (str): Directory containing Google Docs-generated PDF images
        max_samples_per_class (int): Limit samples per class for faster testing
        target_size (tuple): Resize all images to this size for consistent vectors
    
    Returns:
        tuple: (X, y) where X is flattened images, y is labels (0=Word, 1=Google)
    """
    print("Loading dataset...")
    
    X = []
    y = []
    
    # Load Word-generated images (label = 0)
    word_files = [f for f in os.listdir(word_dir) if f.endswith('.png')]
    if max_samples_per_class:
        word_files = word_files[:max_samples_per_class]
    
    print(f"Loading {len(word_files)} Word-generated images...")
    for i, filename in enumerate(word_files):
        try:
            img_path = os.path.join(word_dir, filename)
            img = Image.open(img_path).convert('L')  # Ensure grayscale
            img = img.resize(target_size, Image.LANCZOS)  # Resize to consistent dimensions
            img_array = np.array(img).flatten()  # Flatten to 1D vector
            X.append(img_array)
            y.append(0)  # Word = 0
            
            if (i + 1) % 50 == 0:
                print(f"  Loaded {i + 1}/{len(word_files)} Word images")
        except Exception as e:
            print(f"  Error loading {filename}: {e}")
    
    # Load Google Docs-generated images (label = 1)
    google_files = [f for f in os.listdir(google_dir) if f.endswith('.png')]
    if max_samples_per_class:
        google_files = google_files[:max_samples_per_class]
    
    print(f"Loading {len(google_files)} Google Docs-generated images...")
    for i, filename in enumerate(google_files):
        try:
            img_path = os.path.join(google_dir, filename)
            img = Image.open(img_path).convert('L')  # Ensure grayscale
            img = img.resize(target_size, Image.LANCZOS)  # Resize to consistent dimensions
            img_array = np.array(img).flatten()  # Flatten to 1D vector
            X.append(img_array)
            y.append(1)  # Google = 1
            
            if (i + 1) % 50 == 0:
                print(f"  Loaded {i + 1}/{len(google_files)} Google images")
        except Exception as e:
            print(f"  Error loading {filename}: {e}")
    
    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features per sample")
    print(f"Class distribution: Word={np.sum(y==0)}, Google={np.sum(y==1)}")
    
    return X, y

def train_svm_classifier(X_train, y_train, X_test, y_test):
    """Train and evaluate SVM classifier."""
    print("\n=== Training SVM Classifier ===")
    
    # Use RBF kernel for non-linear classification
    print("Training SVM with RBF kernel...")
    start_time = time.time()
    
    svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
    svm_model.fit(X_train, y_train)
    
    train_time = time.time() - start_time
    print(f"Training completed in {train_time:.2f} seconds")
    
    # Predictions
    y_pred = svm_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"SVM Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Word', 'Google']))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    return svm_model, accuracy

def train_sgd_classifier(X_train, y_train, X_test, y_test):
    """Train and evaluate SGD classifier."""
    print("\n=== Training SGD Classifier ===")
    
    print("Training SGD with hinge loss...")
    start_time = time.time()
    
    sgd_model = SGDClassifier(loss='hinge', alpha=0.01, max_iter=1000, 
                             tol=1e-3, random_state=42)
    sgd_model.fit(X_train, y_train)
    
    train_time = time.time() - start_time
    print(f"Training completed in {train_time:.2f} seconds")
    
    # Predictions
    y_pred = sgd_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"SGD Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Word', 'Google']))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    return sgd_model, accuracy

def main():
    """Main training pipeline."""
    
    # Load dataset (use subset for initial testing)
    print("PDF Provenance Detection - Baseline Classifiers")
    print("=" * 50)
    
    # Start with smaller subset for faster iteration
    X, y = load_dataset(max_samples_per_class=100)
    
    # Normalize features to same scale
    print("\nNormalizing features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split dataset
    print("Splitting dataset (80% train, 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Train classifiers
    svm_model, svm_accuracy = train_svm_classifier(X_train, y_train, X_test, y_test)
    sgd_model, sgd_accuracy = train_sgd_classifier(X_train, y_train, X_test, y_test)
    
    # Summary
    print("\n" + "=" * 50)
    print("BASELINE RESULTS SUMMARY")
    print("=" * 50)
    print(f"SVM Accuracy:  {svm_accuracy:.4f}")
    print(f"SGD Accuracy:  {sgd_accuracy:.4f}")
    
    best_model = "SVM" if svm_accuracy > sgd_accuracy else "SGD"
    print(f"Best performing model: {best_model}")
    
    # Save models
    print("\nSaving trained models...")
    with open('svm_model.pkl', 'wb') as f:
        pickle.dump(svm_model, f)
    with open('sgd_model.pkl', 'wb') as f:
        pickle.dump(sgd_model, f)
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    print("Models saved successfully!")

if __name__ == "__main__":
    main()