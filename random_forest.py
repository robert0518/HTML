import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.impute import KNNImputer
import numpy as np

# Load training, validation, and test data
train_file_path = 'train_split.csv'
valid_file_path = 'valid_split.csv'
test_file_path = 'same_season_test_data.csv'

data = pd.read_csv(train_file_path)
valid_data = pd.read_csv(valid_file_path)
test_data = pd.read_csv(test_file_path)

# Target column for training and validation
data['home_team_win'] = data['home_team_win'].astype(int)
valid_data['home_team_win'] = valid_data['home_team_win'].astype(int)
y_train = data['home_team_win']
y_valid = valid_data['home_team_win']

# Convert date to datetime for training and validation
data['date'] = pd.to_datetime(data['date'])
valid_data['date'] = pd.to_datetime(valid_data['date'])

# Define linear weighting scheme
def linear_weighting(data):
    max_days = (data['date'] - data['date'].min()).dt.days.max()
    return (data['date'] - data['date'].min()).dt.days / max_days

# Remarked other weighting schemes
# def exponential_weighting(data, alpha=0.05):
#     days_since_start = (data['date'] - data['date'].min()).dt.days
#     return np.exp(alpha * days_since_start)

# def inverse_weighting(data):
#     days_since_start = (data['date'] - data['date'].min()).dt.days
#     return 1 / (days_since_start + 1)

# def bucket_weighting(data, buckets=(10, 30, 60), weights=(3, 2, 1)):
#     days_since_start = (data['date'] - data['date'].min()).dt.days
#     weight_series = pd.Series(1, index=data.index)
#     for bucket, weight in zip(buckets, weights):
#         weight_series[days_since_start <= bucket] = weight
#     return weight_series

# Use only linear weighting
sample_weights = linear_weighting(data)

# Prepare features (one-hot encoding)
categorical_columns = ['is_night_game', 'home_team_abbr', 'away_team_abbr', 'home_team_season', 'away_team_season']
combined_data = pd.concat([
    data.drop(columns=['Unnamed: 0', 'id', 'home_pitcher', 'away_pitcher', 'home_team_win', 'date']),
    valid_data.drop(columns=['Unnamed: 0', 'id', 'home_pitcher', 'away_pitcher', 'home_team_win', 'date']),
    test_data.drop(columns=['id', 'home_pitcher', 'away_pitcher'])
], ignore_index=True)
combined_data = pd.get_dummies(combined_data, columns=categorical_columns, drop_first=True)

# Split into training, validation, and test sets
X_train = combined_data.iloc[:len(data), :].copy()
X_valid = combined_data.iloc[len(data):len(data) + len(valid_data), :].copy()
X_test = combined_data.iloc[len(data) + len(valid_data):, :].copy()

# Apply KNN imputation
knn_imputer = KNNImputer(n_neighbors=5, weights="uniform")
X_train_imputed = knn_imputer.fit_transform(X_train)
X_valid_imputed = knn_imputer.transform(X_valid)
X_test_imputed = knn_imputer.transform(X_test)

# Convert back to DataFrame
X_train = pd.DataFrame(X_train_imputed, columns=X_train.columns)
X_valid = pd.DataFrame(X_valid_imputed, columns=X_valid.columns)
X_test = pd.DataFrame(X_test_imputed, columns=X_test.columns)

# Train a baseline Random Forest to determine feature importances
baseline_model = RandomForestClassifier(random_state=42, n_estimators=100, max_depth=10, class_weight='balanced')
baseline_model.fit(X_train, y_train, sample_weight=sample_weights)

# Extract feature importances
feature_importances = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': baseline_model.feature_importances_
}).sort_values(by='Importance', ascending=False).reset_index(drop=True)

print("Top 10 Features by Importance:")
print(feature_importances.head(10))

# Select top features (e.g., top 500)
selected_features = feature_importances.head(500)['Feature']
X_train_selected = X_train[selected_features]
X_valid_selected = X_valid[selected_features]
X_test_selected = X_test[selected_features]

# Hyperparameter tuning for Random Forest
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42, class_weight='balanced'),
    param_grid,
    cv=3,
    scoring='accuracy',
    verbose=2,
    n_jobs=-1
)

grid_search.fit(X_train_selected, y_train, sample_weight=sample_weights)

# Best model
best_rf_model = grid_search.best_estimator_
print("Best Parameters:", grid_search.best_params_)

# Validate the model
y_valid_pred = best_rf_model.predict(X_valid_selected)
valid_accuracy = accuracy_score(y_valid, y_valid_pred)
valid_classification_metrics = classification_report(y_valid, y_valid_pred)

print(f"Validation Accuracy with Random Forest: {valid_accuracy}")
print("Classification Report:")
print(valid_classification_metrics)

# Predict on the test set
test_predictions = best_rf_model.predict(X_test_selected)
output = pd.DataFrame({
    'id': test_data['id'],
    'home_team_win': test_predictions.astype(bool)
})

# Save predictions
output.to_csv('Answer.csv', index=False)
print("\nTest predictions saved to 'Answer.csv'.")
