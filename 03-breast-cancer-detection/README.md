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