#!/usr/bin/env python3
"""
3-Class PDF Provenance Classifiers

Trains machine learning models (SVM, SGD) to classify PDFs by generation method:
- Microsoft Word (label 0)
- Google Docs (label 1) 
- Python/ReportLab (label 2)
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

def load_3class_dataset(word_dir='word_pdfs_png', google_dir='google_docs_pdfs_png', 
                       python_dir='python_pdfs_png', max_samples_per_class=None,
                       target_size=(200, 200)):
    """
    Load 3-class binary image dataset.
    
    Args:
        word_dir (str): Directory containing Word-generated PDF images
        google_dir (str): Directory containing Google Docs-generated PDF images
        python_dir (str): Directory containing Python-generated PDF images
        max_samples_per_class (int): Limit samples per class for faster testing
        target_size (tuple): Resize all images to this size
    
    Returns:
        tuple: (X, y) where X is flattened images, y is labels (0=Word, 1=Google, 2=Python)
    """
    print("Loading 3-class dataset...")
    
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
            img = Image.open(img_path).convert('L')
            img = img.resize(target_size, Image.LANCZOS)
            img_array = np.array(img).flatten()
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
            img = Image.open(img_path).convert('L')
            img = img.resize(target_size, Image.LANCZOS)
            img_array = np.array(img).flatten()
            X.append(img_array)
            y.append(1)  # Google = 1
            
            if (i + 1) % 50 == 0:
                print(f"  Loaded {i + 1}/{len(google_files)} Google images")
        except Exception as e:
            print(f"  Error loading {filename}: {e}")
    
    # Load Python-generated images (label = 2)
    python_files = [f for f in os.listdir(python_dir) if f.endswith('.png')]
    if max_samples_per_class:
        python_files = python_files[:max_samples_per_class]
    
    print(f"Loading {len(python_files)} Python-generated images...")
    for i, filename in enumerate(python_files):
        try:
            img_path = os.path.join(python_dir, filename)
            img = Image.open(img_path).convert('L')
            img = img.resize(target_size, Image.LANCZOS)
            img_array = np.array(img).flatten()
            X.append(img_array)
            y.append(2)  # Python = 2
            
            if (i + 1) % 50 == 0:
                print(f"  Loaded {i + 1}/{len(python_files)} Python images")
        except Exception as e:
            print(f"  Error loading {filename}: {e}")
    
    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features per sample")
    print(f"Class distribution: Word={np.sum(y==0)}, Google={np.sum(y==1)}, Python={np.sum(y==2)}")
    
    return X, y

def train_3class_svm(X_train, y_train, X_test, y_test):
    """Train and evaluate 3-class SVM classifier."""
    print("\n=== Training 3-Class SVM Classifier ===")
    
    print("Training SVM with RBF kernel...")
    start_time = time.time()
    
    svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
    svm_model.fit(X_train, y_train)
    
    train_time = time.time() - start_time
    print(f"Training completed in {train_time:.2f} seconds")
    
    # Predictions
    y_pred = svm_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"3-Class SVM Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Word', 'Google', 'Python']))
    print("\nConfusion Matrix:")
    print("         Word  Google  Python")
    cm = confusion_matrix(y_test, y_pred)
    for i, row_name in enumerate(['Word', 'Google', 'Python']):
        print(f"{row_name:8s} {cm[i][0]:4d}   {cm[i][1]:4d}    {cm[i][2]:4d}")
    
    return svm_model, accuracy

def train_3class_sgd(X_train, y_train, X_test, y_test):
    """Train and evaluate 3-class SGD classifier."""
    print("\n=== Training 3-Class SGD Classifier ===")
    
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
    
    print(f"3-Class SGD Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Word', 'Google', 'Python']))
    print("\nConfusion Matrix:")
    print("         Word  Google  Python")
    cm = confusion_matrix(y_test, y_pred)
    for i, row_name in enumerate(['Word', 'Google', 'Python']):
        print(f"{row_name:8s} {cm[i][0]:4d}   {cm[i][1]:4d}    {cm[i][2]:4d}")
    
    return sgd_model, accuracy

def analyze_class_separability(X, y):
    """Analyze how separable the three classes are."""
    print("\n=== Class Separability Analysis ===")
    
    # Calculate mean image dimensions for each class
    class_stats = {}
    for class_id, class_name in enumerate(['Word', 'Google', 'Python']):
        class_mask = (y == class_id)
        class_samples = X[class_mask]
        
        # Calculate statistics
        mean_intensity = np.mean(class_samples)
        std_intensity = np.std(class_samples)
        
        class_stats[class_name] = {
            'count': np.sum(class_mask),
            'mean_intensity': mean_intensity,
            'std_intensity': std_intensity
        }
        
        print(f"{class_name:8s}: {class_stats[class_name]['count']:3d} samples, "
              f"mean intensity: {mean_intensity:.2f}, std: {std_intensity:.2f}")
    
    return class_stats

def main():
    """Main training pipeline for 3-class classification."""
    
    print("PDF Provenance Detection - 3-Class Classification")
    print("=" * 60)
    print("Classes: Word (0), Google Docs (1), Python/ReportLab (2)")
    print("=" * 60)
    
    # Load dataset (use larger subset for better testing)
    X, y = load_3class_dataset(max_samples_per_class=100)  # Larger subset
    
    # Analyze class separability
    class_stats = analyze_class_separability(X, y)
    
    # Normalize features
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
    svm_model, svm_accuracy = train_3class_svm(X_train, y_train, X_test, y_test)
    sgd_model, sgd_accuracy = train_3class_sgd(X_train, y_train, X_test, y_test)
    
    # Summary
    print("\n" + "=" * 60)
    print("3-CLASS CLASSIFICATION RESULTS SUMMARY")
    print("=" * 60)
    print(f"SVM Accuracy:  {svm_accuracy:.4f}")
    print(f"SGD Accuracy:  {sgd_accuracy:.4f}")
    
    best_model = "SVM" if svm_accuracy > sgd_accuracy else "SGD"
    print(f"Best performing model: {best_model}")
    
    # Save models
    print("\nSaving 3-class trained models...")
    with open('svm_3class_model.pkl', 'wb') as f:
        pickle.dump(svm_model, f)
    with open('sgd_3class_model.pkl', 'wb') as f:
        pickle.dump(sgd_model, f)
    with open('scaler_3class.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    print("3-class models saved successfully!")
    print("\nNext steps:")
    print("- Scale up to full dataset")
    print("- Try CNN for even better performance")
    print("- Test on real-world forensic samples")

if __name__ == "__main__":
    main()