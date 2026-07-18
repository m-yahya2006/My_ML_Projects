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


