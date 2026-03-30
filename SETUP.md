# SETUP.md

## Repository Setup
Forked from:
https://github.com/delveccj/ForensicsDetective.git

Cloned my fork locally:
git clone https://github.com/cfisher7595-stack/ForensicsDetective.git
cd ForensicsDetective

## Environment Setup
Created and activated a Python virtual environment:
python3 -m venv venv
source venv/bin/activate

## Installed Dependencies
pip install numpy pandas matplotlib scikit-learn pillow opencv-python

## Verification

### PDF to Image Conversion
Ran:
python pdf_to_binary_image.py

Result:
- Successfully converted Word-generated PDFs
- Successfully converted Google Docs-generated PDFs
- Successfully converted Python-generated PDFs
- PNG outputs were generated successfully

### Baseline Classifier Training
Ran:
python train_baseline_classifiers.py

Result:
- Dataset loaded successfully
- 200 samples, 40000 features per sample
- SVM Accuracy: 0.9750
- SGD Accuracy: 0.9750
- Models saved successfully

Confusion Matrix:
[[20  0]
 [ 1 19]]

## System Information
- macOS
- Python virtual environment (venv)

## GitHub Setup
Required collaborators added:
- delveccj
- AnushkaTi
