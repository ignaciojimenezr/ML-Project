import pandas as pd
import numpy as np
import time

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# load processed dataset
df = pd.read_csv('processed_data.csv')

# features = all columns except the label columns
feature_columns = [c for c in df.columns if c not in ['type', 'type_encoded']]
X = df[feature_columns].values
y = df['type_encoded'].values

# train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# hyperparameter grid for Random Forest
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'bootstrap': [True]
}

rf = RandomForestClassifier(random_state=42, n_jobs=-1)

grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    scoring='f1_weighted',   # same idea as KNN tuning
    cv=5,
    n_jobs=-1,
    verbose=1
)

# training runtime 
start_train = time.time()
grid_search.fit(X_train, y_train)
end_train = time.time()
runtime_train = end_train - start_train

best_rf = grid_search.best_estimator_

# prediction runtime
start_pred = time.time()
y_pred = best_rf.predict(X_test)
end_pred = time.time()
runtime_pred = end_pred - start_pred

# metrics on test set
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# print metrics 
print("Best parameters:", grid_search.best_params_)
print("Best cross-validation score:", grid_search.best_score_)

print(f"Model training runtime: {runtime_train:.4f} seconds")
print(f"Model prediction runtime: {runtime_pred:.4f} seconds")
print("------------------------------------------------------------------------------")
print(f"Accuracy: {accuracy}")
print(f"Recall: {recall:.4f}")
print(f"Precision: {precision:.4f}")
print(f"F1 Score: {f1}")

# confusion matrix viusalization
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Random Forest - Confusion Matrix")
plt.tight_layout()
plt.show()
