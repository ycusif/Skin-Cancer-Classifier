# Melanoma Skin Cancer Classification with CNNs

A deep learning project that classifies dermoscopic skin lesion images as **benign** or **malignant** (melanoma), comparing a custom CNN built from scratch against a transfer-learning approach using MobileNetV2.

## Overview

Melanoma is one of the most dangerous forms of skin cancer, and early detection significantly improves patient outcomes. This project explores whether convolutional neural networks can reliably distinguish malignant lesions from benign ones using the [Melanoma Skin Cancer Dataset of 10,000 Images](https://www.kaggle.com/datasets/hasnainjaved/melanoma-skin-cancer-dataset-of-10000-images) from Kaggle.

Three modeling experiments were run and compared on the same train/validation split:

1. A baseline **vanilla CNN** built from scratch
2. A **deeper CNN** with more filters and a smaller input resolution
3. **Transfer learning** with a frozen MobileNetV2 backbone

## Dataset

- **Source:** Kaggle — Melanoma Skin Cancer Dataset of 10,000 Images
- **Classes:** `benign` (5,000 images), `malignant` (4,605 images)
- **Split used:** 80% train / 20% validation (7,684 / 1,921 images), plus a separate held-out test set
- **Preprocessing:**
  - Resized to 224×224 (150×150 for Experiment 2)
  - Rescaled pixel values to [0, 1]
  - Augmentation on the training set: rotation, width/height shift, zoom, horizontal flip
  - CLAHE (Contrast Limited Adaptive Histogram Equalization) explored as an additional contrast-enhancement step

## Repository Structure

```
.
├── melanoma-skin-cancer-cnn.ipynb   # Full training & evaluation pipeline
├── report.pdf                       # Written project report
└── LICENSE
```

## Methodology

### Experiment 1 — Vanilla CNN
A 4-block convolutional network (32 → 64 → 128 → 256 filters) with max-pooling and progressively increasing dropout (0.25 → 0.3), trained for up to 50 epochs with early stopping, learning-rate reduction on plateau, and checkpointing on best validation accuracy.

### Experiment 2 — Deeper CNN
A wider/deeper variant (64 → 128 → 256 → 512 filters) at a reduced 150×150 input resolution, intended to test whether additional capacity and a smaller input size would improve results.

### Experiment 3 — MobileNetV2 Transfer Learning
A MobileNetV2 backbone (ImageNet weights, frozen) with a custom classification head (batch normalization, dense layer, dropout, sigmoid output), trained for up to 20 epochs with early stopping.

All experiments use binary cross-entropy loss, the Adam optimizer, and are evaluated with accuracy, precision, recall, F1-score, and confusion matrices.

## Results

| Experiment | Architecture | Validation Accuracy | Notes |
|---|---|---|---|
| 1 | Vanilla CNN (4-block) | **~91–92%** | Best-performing model; balanced precision/recall across both classes (F1 ≈ 0.91–0.92) |
| 2 | Deeper CNN (150×150) | Not conclusively evaluated | Training loop reused the Experiment 1 model rather than the newly defined deeper model — see Known Issues |
| 3 | MobileNetV2 (frozen) | Not conclusively evaluated | Same training-loop issue as Experiment 2 — see Known Issues |

The vanilla CNN from Experiment 1 was the only model verified to have trained and evaluated correctly, reaching approximately **92% test accuracy** with balanced performance across both benign and malignant classes.

## Known Issues

While documenting this notebook, I found that the training calls for Experiments 2 and 3 (`model.fit(...)`) reference the original Experiment 1 model object instead of `model_2` and `model_3`. As a result, the deeper CNN and MobileNetV2 models were never actually trained, which explains their poor, near-random evaluation scores later in the notebook. This is a good next fix: updating those calls to `model_2.fit(...)` and `model_3.fit(...)` respectively should allow a fair comparison between all three architectures.

## Tech Stack

- **Language:** Python
- **Deep Learning:** TensorFlow / Keras
- **Computer Vision:** OpenCV
- **Data Handling:** NumPy, Pandas
- **Visualization:** Matplotlib, Seaborn
- **Evaluation:** Scikit-learn (confusion matrix, classification report, accuracy score)

## Running the Project

1. Download the dataset from Kaggle and update `DATASET_PATH` in the notebook to point to your local copy.
2. Install dependencies:
   ```
   pip install tensorflow opencv-python numpy pandas matplotlib seaborn scikit-learn
   ```
3. Run the notebook cell by cell. Each experiment section is clearly labeled and can be run independently once the data-loading and preprocessing cells have been executed.

## Report

See [`report.pdf`](./report.pdf) for the full written analysis and discussion of results.

## License

See [`LICENSE`](./LICENSE) for details.
