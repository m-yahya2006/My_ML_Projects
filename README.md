# Project 1: College Majors Salary Prediction

## What this project does
This project predict salary using features like Major_category, full_time, part_time etc.

## Dataset
Source: FiveThirtyEight
URL: https://raw.githubusercontent.com/fivethirtyeight/data/master/college-majors/recent-grads.csv
172 majors, 21 columns

## Models tested
Two models were tested : Linear Regression , Decision Tree & Random Tree Regression

## Results
Linear Regression  → R² = 0.68  ← best
Random Forest      → R² = 0.63
Decision Tree      → R² = 0.31  ← worst

## Key finding
Small dataset  → simpler models often win
Large dataset  → complex models like Random Forest shine

## How to run
python major.py



# Project 2: UCI BankNote Authentication


## What this project does
This project predict Fake/real notes using Features like 'variance', 'skewness', 'curtosis', 'entropy'.

## Dataset
Source: UCI Machine Learning
URL: "https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt"


## Models tested
Decision Tree Classification

## Results
Accuracy: 97%
Recall (fake): 100%

## Key finding
Recall matters more than accuracy for fraud detection

## How to run
python bankNote.py



# Project 3: Breast Cancer

## What this project does
This project predict Breast Cancer using features like 'mean radius' 'mean texture' 'mean perimeter' 'mean area' and many more

## Dataset
Source: sklearn dataset
569 rows, 31 columns

## Models tested
3 models were tested : DecisionTree , KNN, RandomForestTree

## Key metric and why recall matters here
Recall was chosen over accuracy because recall measures the percentage of actual cancer patients that the model correctly identifies.

## Results
Acurracy:  0.956140350877193 ->DecisionTree
Acurracy:  0.9736842105263158 ->KNN
Acurracy:  0.9649122807017544 -> RandomForestClassifer

## Key findings
Limitation: model misses ~3% of malignant cases
Real world use: decision support tool, not a replacement for medical diagnosis


## How to run
python end_to_end_project.py