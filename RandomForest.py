import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# load processed dataset
df = pd.read_csv('processed_data.csv')

# features = all columns except the label columns
feature_columns = [c for c in df.columns if c not in ['type', 'type_encoded']]
X = df[feature_columns].values
y = df['type_encoded'].values

# train / test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y  # preserve class proportions
)

# Random Forest model + hyperparameter grid
rf = RandomForestClassifier(
    random_state=42,
    n_jobs=-1   # use all CPU cores
)

param_grid = {
    'n_estimators': [100, 300], # num of trees
    'max_depth': [None, 10, 20],    # depth of each tree
    'min_samples_split': [2, 5],    # min samples to split internal node
    'min_samples_leaf': [1, 2], # min samples at leaf node
    'max_features': ['sqrt', 'log2']    # num of features considered at each split
}

grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=3,   # 3-fold cross validation
    scoring='f1_weighted',  # for class imbalance
    n_jobs=-1,
    verbose=2
)

# fit model
print("Initiating Random Forest training with cross-validation...")
grid_search.fit(X_train, y_train)

print("\nBest hyperparameters found:")
print(grid_search.best_params_)

best_rf = grid_search.best_estimator_

# test set evaluate 
y_pred = best_rf.predict(X_test)

print("\nAccuracy on test set: {:.4f}".format(accuracy_score(y_test, y_pred)))
print("\nClassification report:")
print(classification_report(y_test, y_pred))

print("\nConfusion matrix:")
print(confusion_matrix(y_test, y_pred))

# feature importance
importances = best_rf.feature_importances_
indices = np.argsort(importances)[::-1] # sort descending

print("\nTop 20 most important features:")
for rank, idx in enumerate(indices[:20], start=1):
    print(f"{rank:2d}. {feature_columns[idx]:30s}  importance = {importances[idx]:.4f}")
