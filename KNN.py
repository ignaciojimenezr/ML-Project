import pandas as pd
import numpy as np
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

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

# fit the grid search
start_time = time.time()
grid_search.fit(X_train,y_train)
end_time = time.time()
runtime = end_time - start_time

# Make predictions
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

print(f"Accuracy: {accuracy}")
print(f"Recall: {recall:.4f}")
print(f"Precision: {precision:.4f}")
print(f"F1 Score: {f1}")

