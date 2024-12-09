import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
# Load training dataset
train_file_path = 'train_split.csv'
train_data = pd.read_csv(train_file_path)

valid_file_path = 'valid_split.csv'
valid_data = pd.read_csv(valid_file_path)

# Feature selection: Drop non-numerical and target columns for training data
str_columns = train_data.select_dtypes(include=['object','string']).columns.tolist()
str_columns.append('home_team_win')

X_train = train_data.drop(columns=str_columns)

y_train = train_data['home_team_win']
X_val = valid_data.drop(columns=str_columns)
y_val = valid_data['home_team_win']


# Handle missing values
X_train = X_train.fillna(X_train.mean())  # Fill missing values with column means
X_val = X_val.fillna(X_val.mean())  # Fill missing values with column means


# Standardize features
scaler = StandardScaler()

selector = SelectKBest(score_func=f_classif, k=20)
X_train = selector.fit_transform(X_train, y_train)
X_val = selector.transform(X_val)

# Logistic Regression as the base estimator for AdaBoost
log_reg = LogisticRegression(max_iter=500, solver='lbfgs', random_state=42)

# Define a decision stump
decision_stump = DecisionTreeClassifier(max_depth=1, random_state=42)

# Configure AdaBoostClassifier
adaboost_model = AdaBoostClassifier(
    estimator=decision_stump,
    n_estimators=500,  # Number of weak learners
    learning_rate=0.5,  # Learning rate for updating weights
    algorithm='SAMME',
    random_state=42
)

# Create a pipeline
pipeline = Pipeline([
    ('scaler', scaler),  # Standardize the data
    ('adaboost', adaboost_model)
])

# Fit the AdaBoost model
pipeline.fit(X_train, y_train)

# Evaluate the model on the validation set
y_val_pred = pipeline.predict(X_val)

# Metrics to evaluate the performance
print("Accuracy:", accuracy_score(y_val, y_val_pred))
print("Confusion Matrix:\n", confusion_matrix(y_val, y_val_pred))
print("Classification Report:\n", classification_report(y_val, y_val_pred))


# Define parameter grid
param_grid = {
    'adaboost__n_estimators': [1000, 5000, 10000],
    'adaboost__learning_rate': [0.01, 0.1, 1.0]
}

# GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='accuracy', verbose=1, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best parameters and accuracy
print("Best Parameters:", grid_search.best_params_)
print("Best Cross-Validation Accuracy:", grid_search.best_score_)

# Predict with the best model
best_model = grid_search.best_estimator_
y_val_pred_best = best_model.predict(X_val)

# Evaluate the tuned model
print("Tuned Model Accuracy:", accuracy_score(y_val, y_val_pred_best))
print("Confusion Matrix:\n", confusion_matrix(y_val, y_val_pred_best))
print("Classification Report:\n", classification_report(y_val, y_val_pred_best))


# Generate learning curve data
train_sizes, train_scores, val_scores = learning_curve(
    best_model, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1
)

# Compute mean and standard deviation of scores
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)
val_std = np.std(val_scores, axis=1)

# Plot learning curves
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean, 'o-', label='Training Accuracy', color='blue')
plt.plot(train_sizes, val_mean, 'o-', label='Validation Accuracy', color='green')
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='green')

plt.xlabel('Training Set Size')
plt.ylabel('Accuracy')
plt.title('Learning Curves')
plt.legend()
plt.grid()
plt.show()
