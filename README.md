# -CT-Bone-Classifier-
CNN-based classifier for human vs. non-human bone identification using micro-CT histomorphological images. Developed for MA thesis research in forensic anthropology. Includes sample-level k-fold cross-validation, early stopping, and Grad-CAM visualization.
# Bone Cell Classification System

Code repository for the MA thesis:

**"Non-Destructive Automated Classification of Human and Large Mammal Long Bone Fragments: A Deep Learning Approach Using Micro-CT Histomorphology and Grad-CAM Interpretation"**

---

## Overview

This project uses a convolutional neural network (CNN) to automatically classify micro-CT images of cortical bone as either human or non-human. The model is built on a pretrained ResNet-18 architecture and trained using sample-level k-fold cross-validation to prevent data leakage. Grad-CAM visualizations are generated to show which regions of each image informed the model's classification decision.

---

## Requirements

- Python 3.10+
- PyTorch
- torchvision
- scikit-learn
- matplotlib
- seaborn
- Pillow
- numpy

Install all dependencies with:

```bash
pip install torch torchvision scikit-learn matplotlib seaborn Pillow numpy
```

---

## Setup

Before running, update the file paths in the `Config` class at the top of `bone_classifier.py` to match your local directory structure:

```python
DATA_DIR = "/path/to/data"
OUTPUT_DIR = "/path/to/results"
```

Images should be organized as follows:

```
data/
├── Human Images Train VOIs/
│   ├── SampleA_VOI1/
│   │   ├── image001.bmp
│   │   └── ...
│   └── ...
└── Non-Human Images Train VOIs/
    ├── SampleB_VOI1/
    │   ├── image001.bmp
    │   └── ...
    └── ...
```

---

## Usage

Run the classifier from the command line:

```bash
python bone_classifier.py
```

The script will:
1. Load and group images by sample
2. Run 5-fold cross-validation with early stopping
3. Evaluate each fold and print accuracy and confusion matrix
4. Generate Grad-CAM visualizations for 10 human and 10 non-human samples per fold
5. Save all results, figures, and a summary text file to the output directory

---

## Output

Results are saved to the directory specified in `OUTPUT_DIR`:

| File | Description |
|------|-------------|
| `results.txt` | Per-fold and overall accuracy, confidence intervals, per-image predictions |
| `training_curves_all_folds.png` | Training and validation loss curves |
| `confusion_matrix_mean.png` | Mean confusion matrix across all folds |
| `accuracy_per_fold.png` | Bar chart of per-fold test accuracy |
| `gradcam_fold*/` | Grad-CAM visualizations per fold |

---

## About

This code accompanies the following MA thesis:

Kelley, K. (2025). Non-Destructive Automated Classification of Human and Large Mammal Long Bone Fragments: A Deep Learning Approach Using Micro-CT Histomorphology and Grad-CAM Interpretation. MA Thesis.
