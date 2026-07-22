# Project 4: UCI Student Performance Classification

## Project Overview

This project uses machine learning and a PyTorch neural network to classify students as **Pass** or **Fail** based on their demographic, social, and academic features.

## Dataset

**Source:** UCI Machine Learning Repository

- Samples: **649**
- Features: **56** (after one-hot encoding)
- Target: **Pass (G3 ≥ 10)** or **Fail (G3 < 10)**

## Models

- PyTorch Feedforward Neural Network
- Random Forest Classifier
- K-Nearest Neighbors (KNN)
- Decision Tree Classifier

## Results

| Model | Accuracy |
|--------|----------|
| Decision Tree | 77.0% |
| K-Nearest Neighbors | 83.0% |
| Random Forest | 82.0% |
| PyTorch Neural Network | **84.6%** |

## Key Findings

### Limitations

The models did not achieve higher accuracy because:

- The dataset is relatively small (649 samples).
- The dataset is imbalanced (Pass: 549, Fail: 100).
- The available features cannot fully explain student performance.
- Students near the pass/fail boundary (e.g., grades 9 and 10) have very similar feature values, making classification difficult.

## Technologies Used

- Python
- PyTorch
- scikit-learn
- pandas
- NumPy

## How to Run

```bash
python exam_project_StudentPerformancePrediction.py
```