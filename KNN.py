import pandas as pd
import numpy as np
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve,auc
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import label_binarize
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

# load the processed data set
df= pd.read_csv('processed_data.csv')

# prepare features and target
feature_columns =[c for c in df.columns if c not in ['type','type_encoded']]
X =df[feature_columns].values
y=df['type_encoded'].values

# split the data in to training and testing sets
X_train,X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# hyperparameter tuning using cross-validation

# define the parameter grid
parm_grid = {
    'n_neighbors' : [9,15,19,25,29],
    'weights' : ['uniform','distance'],
    'metric' : ['euclidean','manhattan']
}

# creating the GridSearchCV object
grid_search = GridSearchCV(KNeighborsClassifier(),parm_grid,cv=5)# 5-fold cross-validation

# Training and fitting the grid search
start_time = time.time()
grid_search.fit(X_train,y_train)
end_time = time.time()
runtime = end_time - start_time

# testing and making predictions
start_time_predict = time.time()
predictions = grid_search.predict(X_test)
end_time_predict = time.time()
runtime_predict = end_time_predict - start_time_predict



#accuracy
accuracy = accuracy_score(y_test, predictions)

#recall
recall = recall_score(y_test, predictions, average='weighted')

#precision
precision= precision_score(y_test,predictions, average='weighted')

#F1 score
f1= f1_score(y_test,predictions, average='weighted')

# Print the best parameters and score
print("Best parameters:", grid_search.best_params_)
print("Best cross-validation score:", grid_search.best_score_)

print(f"Model training runtime: {runtime:.4f} seconds")
print(f"Model prediction runtime: {runtime_predict:.4f} seconds")

# Displaying metrics
print(f"Accuracy: {accuracy}")
print(f"Recall: {recall:.4f}")
print(f"Precision: {precision:.4f}")
print(f"F1 Score: {f1}")

#ROC curve

# Get the best model
best_knn = grid_search.best_estimator_

# Binarize the labels for multiclass ROC
classes = np.unique(y)
y_test_bin = label_binarize(y_test, classes=classes)

# Predict probabilities
y_score = best_knn.predict_proba(X_test)

# Plot ROC curve for each class
plt.figure(figsize=(8, 6))

for i, cls in enumerate(classes):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"Class {cls} (AUC = {roc_auc:.2f})")

# Random chance line
plt.plot([0, 1], [0, 1], 'k--')

plt.title("Multiclass ROC Curve (One-vs-Rest)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.grid(True)
plt.legend()
plt.show()


