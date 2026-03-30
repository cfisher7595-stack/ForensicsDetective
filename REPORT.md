# Project Report

## Task 3: Robustness Analysis

The classifiers were trained exclusively on the original dataset and evaluated on both clean and augmented datasets to assess robustness.

### Baseline Performance

* SVM Accuracy (original): 0.9686
* SGD Accuracy (original): 0.9937

### Augmentation Impact

#### 1. Gaussian Noise

* Minimal impact on both models
* Accuracy remained nearly identical to baseline
* Explanation: Gaussian noise introduces pixel-level variation but preserves global structural layout patterns, which are the main features used by the classifiers.

#### 2. JPEG Compression

* No significant degradation observed
* Explanation: Compression artifacts mainly affect fine details, but document layout structure and text positioning remain intact.

#### 3. DPI Downsampling

* Severe degradation for SVM (~50% accuracy)
* Moderate impact on SGD
* Explanation: Downsampling removes high-frequency spatial details and structural cues that distinguish document generation pipelines, causing SVM to fail.

#### 4. Random Cropping

* Largest degradation overall (especially for SGD)
* Accuracy dropped to ~50%
* Explanation: Cropping removes borders and layout regions that contain important provenance signals such as margins, alignment, and formatting consistency.

#### 5. Bit-Depth Reduction

* No significant impact
* Explanation: Reducing grayscale levels preserves structural patterns, which are more important than fine intensity variations for classification.

### Key Findings

* Most harmful augmentations:

  * SVM: DPI downsampling
  * SGD: Random cropping

* Most robust augmentations:

  * Gaussian noise
  * JPEG compression
  * Bit-depth reduction

### Conclusion

The classifiers rely heavily on spatial structure and layout features rather than fine pixel intensity variations. Augmentations that distort structure (cropping and downsampling) significantly degrade performance, while those that preserve layout (noise, compression) have minimal impact.
