# CS549 Final Procject: Detection of Malicious Links via Lexical Analysis 

## Overview
This project first processes a dataset of URLs and prepares it for machine-learning classification of malicious links. From there it implements three supervised machine learning models to classify sample based on the features in proccessed_data.csv:\
  - K-Nearest Neighbors(KNN)\
  - Random Forest\
  - Logistic Regression\

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
   - urllib
   - math
   - collections
   - imblearn
### Installation:
  pip install pandas, numpy, scikit-learn, matplotlib, urllib, math, collections, imblearn

### Dataset Requirements
urldata.csv must be in the same directory
**Required columms:**
  -url: url string
  -type: catogorical label
  **Lable Columns:**
    -Primary label: type_encoded
        -Encoded values:
          benign → 0
          phishing → 1
          malware → 2
          defacement → 3
        -Ignored label: type
          -Kept for reference only and is not used for training
  all remaining colums are numeric features
  
processed_data.csv must be in the same directory
  - Label: type_encoded
  - Ignored label: type
  -  all remaining colums are numeric features
### How to run
**Preprocessing**
  python preprocess_urls.py
  Outputs: processed_data.csv
  
**Logistic Regression:**
  python train_logistic_regression.py
  Outputs: metrics + prediction runtime

**Random Forest:**
  python RandomForest.py
  Outputs: best params, metrics, confusion matrix, feature importance
**KNN:**
  python KNN.py
  Outputs: best params, metrics, ROC curve
  
**Hyperparameter Tuning Summary**
KNN: n_neighbors, weights, metric

Random Forest: n_estimators, max_depth, min_samples_split,           min_samples_leaf, max_features

Logistic Regression: C values


   

