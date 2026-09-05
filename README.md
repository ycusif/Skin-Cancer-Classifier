# Melanoma Skin Cancer Classification with CNNs

A deep learning project that classifies dermoscopic skin lesion images as **benign** or **malignant** (melanoma), comparing a custom CNN built from scratch against a transfer-learning approach using MobileNetV2.


## Overview

Melanoma is one of the most aggressive forms of skin cancer, and early detection significantly improves survival outcomes. This project explores whether convolutional neural networks can reliably distinguish malignant lesions from benign ones using the [Melanoma Skin Cancer Dataset of 10,000 Images](https://www.kaggle.com/datasets/hasnainjaved/melanoma-skin-cancer-dataset-of-10000-images) from Kaggle.

Three modeling experiments were run and compared:

1. A baseline **vanilla CNN** built from scratch
2. A **deeper "Improved" CNN** with more filters and a smaller input resolution
3. **Transfer learning** with a frozen MobileNetV2 backbone

## Dataset

- **Source:** Kaggle — Melanoma Skin Cancer Dataset of 10,000 Images
- **Classes:** `benign` (~5,000 images), `malignant` (~4,600 images)
- **Structure:** Folder-based, with separate training and testing directories
- **Split used:** 80% train / 20% validation (1,921 validation images)
- **Preprocessing:**
  - Normalized to [0, 1] by dividing pixel values by 255
  - Augmentation on the training set only: rotation (±20°), width/height shift (10%), zoom (10%), horizontal flip
  - CLAHE (Contrast Limited Adaptive Histogram Equalization) applied to the luminance (Y) channel to improve local contrast and edge definition on lesions

## Repository Structure

```
.
├── melanoma-skin-cancer-cnn.ipynb   # Full training & evaluation pipeline
├── report.pdf                       # Full written project report
└── LICENSE
```

## Methodology

### Experiment 1 — Vanilla CNN
A 4-block convolutional network (32 → 64 → 128 → 256 filters), each followed by max-pooling and progressively increasing dropout (0.25 → 0.3), then a 256-unit dense layer (dropout 0.5) and a sigmoid output. Trained for up to 50 epochs with:
- Adam optimizer (learning rate 0.0001), binary cross-entropy loss
- `EarlyStopping` (patience 5, restores best weights)
- `ModelCheckpoint` (saves best validation accuracy)
- `ReduceLROnPlateau` (factor 0.2, patience 5, min LR 1e-5)

### Experiment 2 — Improved CNN
A deeper variant (64 → up to 512 filters) at a reduced 150×150 input resolution with lighter dropout (0.15–0.20) and a 128-unit dense layer, aiming to test whether more capacity at lower resolution would improve results.

### Experiment 3 — MobileNetV2 Transfer Learning
A MobileNetV2 backbone (ImageNet weights, frozen) with a custom classification head — batch normalization, a 512-unit dense layer, dropout (0.5), and a sigmoid output.

## Results

**Experiment 1 (Vanilla CNN)** — final validation metrics:

| Metric | Training | Validation |
|---|---|---|
| Accuracy | 92% | 91.25% |
| Loss | 0.2326 | 0.2478 |

Classification report (validation set):

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| Benign | 0.92 | 0.92 | 0.92 | 1,000 |
| Malignant | 0.91 | 0.91 | 0.91 | 921 |
| **Overall accuracy** | | | **0.92** | 1,921 |

Per the project report, the best recorded **test** accuracy across all experiments was **89%**, achieved by the Vanilla CNN.

**Experiments 2 and 3** were affected by a coding bug (see below): `history_ex2` and `history_ex3` were assigned from `model.fit(...)` instead of `model_2.fit(...)` / `model_3.fit(...)`, so the recorded training/validation curves for those two experiments don't actually reflect their own models. The report notes the final architectures were built correctly and prediction-based evaluation was still carried out, but flags the Improved CNN's result as "less trustworthy" specifically because of the lost training history. The MobileNetV2 experiment is discussed as benefiting from pretrained ImageNet features, though given the shared bug, its reported per-epoch metrics should be read with the same caution.

## Known Issues

The training calls for Experiments 2 and 3 reference the wrong model object (`model` instead of `model_2` / `model_3`), corrupting the recorded training history for those two runs. This is flagged directly in the report as a limitation. A good next step would be correcting these references and re-running Experiments 2 and 3 to get a clean, trustworthy three-way comparison.

## Deployment

A **Streamlit** web app was built on top of the trained model, letting a user upload a skin lesion image and get a real-time benign/malignant prediction with a confidence score — intended to demonstrate how this kind of model could support telemedicine or early screening in areas without easy access to dermatologists.

## Tech Stack

- **Language:** Python
- **Deep Learning:** TensorFlow / Keras
- **Computer Vision:** OpenCV
- **Data Handling:** NumPy, Pandas
- **Visualization:** Matplotlib, Seaborn
- **Evaluation:** Scikit-learn (confusion matrix, classification report, accuracy score)
- **Deployment:** Streamlit

## Running the Project

1. Download the dataset from Kaggle and update `DATASET_PATH` in the notebook to point to your local copy.
2. Install dependencies:
   ```
   pip install tensorflow opencv-python numpy pandas matplotlib seaborn scikit-learn
   ```
3. Run the notebook cell by cell. Each experiment section is clearly labeled and can be run independently once the data-loading and preprocessing cells have been executed.

## Report

See [`report.pdf`](./report.pdf) for the full written analysis, architecture rationale, and discussion of results.

## References

- [Kaggle – Melanoma Skin Cancer Dataset](https://www.kaggle.com/datasets/hasnainjaved/melanoma-skin-cancer-dataset-of-10000-images)
- [MobileNetV2](https://keras.io/api/applications/mobilenet/)

## License

See [`LICENSE`](./LICENSE) for details.
