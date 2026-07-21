# Project 5: MNIST_CNN

## Project Overview
This project contain 70,000 images that are divided into 10 classes. Because the dataset is quite big, I have divided the training data into batch size of 32. 

## Dataset
**Source:** torchvision dataset

- 70,000 images (60,000 train, 10,000 test)
- 10 classes (digits 0-9)
- 28×28 grayscale images

## Model
- Architecture: CNN
- 3 Conv layers (16, 32, 16 filters)
- MaxPool after each conv layer
- Dropout (0.5) for regularization
- Fully connected output layer (10 classes)

## Results
- **Accuracy:** 98.65% (on the test dataset)

## Key Findings
The neural network successfully learned the patterns in the dataset and achieved high accuracy.
- 3 conv layers achieved 98.65% in 5 epochs
- Dropout prevented overfitting
- Loss still decreasing at epoch 5 — more epochs would improve accuracy further

## How to Run

```bash
python mnist_cnn.py
```