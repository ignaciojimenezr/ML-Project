import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression

# Load the processed dataset
df = pd.read_csv('processed_data.csv')

# Prepare features and target
feature_columns = [c for c in df.columns if c not in ['type', 'type_encoded']]
X = df[feature_columns].values
y = df['type_encoded'].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hyperparameter tuning using cross-validation
# Test different C values to find the best one (C controls how strict the model is)
param_grid = {
    'C': [0.1, 1, 10, 100]
}

# GridSearchCV to find the best hyperparameters
grid_search = GridSearchCV(
    LogisticRegression(max_iter=1000, random_state=42),
    param_grid,
    cv=5, # 5-fold cross-validation
    scoring='f1_weighted'
)
grid_search.fit(X_train, y_train)

