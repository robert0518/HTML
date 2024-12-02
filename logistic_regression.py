import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.feature_selection import SelectKBest, f_classif
import numpy as np

# Load training dataset
train_file_path = 'train_data.csv'  # Replace with your dataset path
train_data = pd.read_csv(train_file_path)
print('Training data loaded successfully')

# Load testing dataset
test_file_path = 'same_season_test_data.csv'  # Replace with your testing dataset path
test_data = pd.read_csv(test_file_path)
print('Testing data loaded successfully')

# Feature selection: Drop non-numerical and target columns for training data
X = train_data.drop(columns=['home_team_win', 'home_team_abbr', 'away_team_abbr', 'home_pitcher', 'away_pitcher', 'date'])
y = train_data['home_team_win']  # Target variable

# Extract and process the date column for training weights
train_data['date'] = pd.to_datetime(train_data['date'])
train_data['date_numeric'] = train_data['date'].map(pd.Timestamp.toordinal)  # Convert dates to ordinal values
max_date = train_data['date_numeric'].max()
weights = train_data['date_numeric'].apply(lambda x: 1 + (x - max_date) / 365.0)  # Higher weight for recent games

# Handle categorical columns
X = pd.get_dummies(X, drop_first=True)
print('Finished processing dummy variables for training data')

# Handle missing values
if X.isnull().sum().sum() > 0:
    print("Missing values detected. Filling missing values with column means.")
    X = X.fillna(X.mean())  # Fill missing values with column means

# Feature selection: Use SelectKBest to choose top features
print("Selecting top features...")
selector = SelectKBest(score_func=f_classif, k=20)  # Adjust `k` as needed
X_selected = selector.fit_transform(X, y)
selected_features = X.columns[selector.get_support()]
print(f"Selected features: {list(selected_features)}")

# Split training data into training and validation sets
X_train, X_val, y_train, y_val, train_weights, val_weights = train_test_split(
    X_selected, y, weights, test_size=0.2, random_state=42
)
print(f"Split data: {len(X_train)} training samples and {len(X_val)} validation samples.")

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Initialize and train Logistic Regression model with L2 regularization
lr_model = LogisticRegression(penalty='l2', solver='liblinear', max_iter=1000, random_state=42)
lr_model.fit(X_train_scaled, y_train, sample_weight=train_weights)

# Evaluate the model on the validation set
y_val_pred = lr_model.predict(X_val_scaled)
accuracy = accuracy_score(y_val, y_val_pred)
conf_matrix = confusion_matrix(y_val, y_val_pred)
report = classification_report(y_val, y_val_pred)

print(f"Validation Accuracy: {accuracy}")
print("Validation Confusion Matrix:")
print(conf_matrix)
print("Validation Classification Report:")
print(report)

# Process test data (future games without date column)
X_test = test_data.drop(columns=['home_team_abbr', 'away_team_abbr', 'home_pitcher', 'away_pitcher'])
X_test = pd.get_dummies(X_test, drop_first=True)
X_test = X_test.reindex(columns=selected_features, fill_value=0)  # Align with selected features
X_test_scaled = scaler.transform(X_test)

# Predict on test data
y_test_pred = lr_model.predict(X_test)
test_data['home_game_win'] = y_test_pred.astype(bool)  # Convert to True/False

# Save predictions to CSV
output_file = 'test_predictions.csv'
test_data[['id', 'home_game_win']].to_csv(output_file, index=False)
print(f"Predictions saved to {output_file}")
