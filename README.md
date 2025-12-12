# CS549 Final Procject: Detection of Malicious Links via Lexical Analysis 

## Overview
This project first processes a dataset of URLs and prepares it for machine-learning classification of malicious links. From there it implements three supervised machine learning models to classify sample based on the features in proccessed_data.csv:
  - K-Nearest Neighbors(KNN)
  - Random Forest
  - Logistic Regression

Each model performs data loading, train/test splitting, GirdSearchCV tuning, training, evaluation, and metrics reporting. KNN and Logistic regression Generate a multiclass ROC curve. While Random Forest generates a confusion matrix.

## Project Structure 
  -urldata.csv
  -preprocess.py
  -processed_data.csv
  -KNN.py
  -RandomForest.py
  -train_logistic_regression.py
  -README.pdf

## Environment & Requirements
### Python Version
  Python 3.8+
### Required libraries
   - pandas
   - numpy
   - scikit-learn
   - matplotlib
   - 


