# Project 4: UCI Banknote Authentication (PyTorch)

## Project Overview
This project uses a PyTorch neural network to classify banknotes as **genuine** or **fake** based on four input features:
- Variance
- Skewness
- Curtosis
- Entropy

## Dataset
**Source:** UCI Machine Learning Repository

**URL:**  
https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt

## Model
- PyTorch Feedforward Neural Network

## Results
- **Accuracy:** 100% (on the test dataset)

## Key Findings
The neural network successfully learned the patterns in the dataset and achieved high classification accuracy. The four statistical features provide enough information to distinguish genuine banknotes from fake ones.

## How to Run

```bash
python bankNote.py
```