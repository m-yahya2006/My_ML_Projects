# Project 3: Breast Cancer Classification

## What this project does

This project predicts whether a breast tumor is **malignant** or **benign** using medical features such as **mean radius, mean texture, mean perimeter, mean area**, and other measurements from the dataset.

## Dataset

* **Source:** `sklearn.datasets.load_breast_cancer`
* **Samples:** 569
* **Features:** 30 input features + 1 target column

## Data preprocessing

* Split the dataset into training and testing sets
* Normalized the features using `StandardScaler`
* Converted the data into PyTorch tensors for the neural network model

## Models tested

Four models were trained and compared:

* Decision Tree Classifier
* K-Nearest Neighbors (KNN)
* Random Forest Classifier
* PyTorch Neural Network

## Key metric and why recall matters

Recall is an important metric because it measures the percentage of actual malignant cases that the model correctly identifies. In medical applications, missing a cancer case (false negative) can have serious consequences.

## Results

* **Decision Tree:** 95.61%
* **K-Nearest Neighbors (KNN):** 97.37%
* **Random Forest Classifier:** 96.49%
* **PyTorch Neural Network:** 95.61%

## Key findings

* KNN achieved the highest accuracy among all tested models.
* Feature normalization improved the performance of the PyTorch model.
* The PyTorch neural network achieved competitive results with a simple architecture.

## Limitations

* The dataset is relatively small (569 samples).
* The model may still miss a small number of malignant cases.
* This project is for educational purposes and is not intended for medical diagnosis.

## Real-world use

This model can be used as a **decision support tool** to assist healthcare professionals, but it should never replace medical diagnosis by a qualified doctor.

## How to run

```bash
python end_to_end_project.py
```
