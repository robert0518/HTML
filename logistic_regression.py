import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.feature_selection import SelectKBest, f_classif
import numpy as np

# Load training dataset
train_file_path = 'train_split.csv'  # Replace with your dataset path
train_data = pd.read_csv(train_file_path)

valid_file_path = 'valid_split.csv'
valid_data = pd.read_csv(valid_file_path)

# Load testing dataset
test_file_path = 'same_season_test_data.csv'  # Replace with your testing dataset path
test_data = pd.read_csv(test_file_path)
print('Testing data loaded successfully')
train_data['is_night_game'] = train_data['is_night_game'].map({'True':1, 'False':0})
# Feature selection: Drop non-numerical and target columns for training data
str_columns = train_data.select_dtypes(include=['object','string']).columns.tolist()
for col in str_columns:
    print(col)


X_train = train_data.drop(columns=['home_team_win', 'home_team_abbr', 'away_team_abbr', 'home_pitcher', 'away_pitcher', 'date', 'home_team_season', 'away_team_season'])

y_train = train_data['home_team_win']  # Target variable
X_val = valid_data.drop(columns=['home_team_win', 'home_team_abbr', 'away_team_abbr', 'home_pitcher', 'away_pitcher', 'date', 'home_team_season', 'away_team_season'])
y_val = valid_data['home_team_win']

# Extract and process the date column for training weights
# train_data['date'] = pd.to_datetime(train_data['date'])
# train_data['date_numeric'] = train_data['date'].map(pd.Timestamp.toordinal)  # Convert dates to ordinal values
# max_date = train_data['date_numeric'].max()
# weights = train_data['date_numeric'].apply(lambda x: 1 + (x - max_date) / 365.0)  # Higher weight for recent games


# Handle missing values
if X_train.isnull().sum().sum() > 0:
    print("Missing values detected. Filling missing values with column means.")
    X_train = X_train.fillna(X_train.mean())  # Fill missing values with column means
if X_val.isnull().sum().sum() > 0:
    print("Missing values detected. Filling missing values with column means.")
    X_train = X_val.fillna(X_val.mean())  # Fill missing values with column means

# Feature selection: Use SelectKBest to choose top features
print("Selecting top features...")
selector = SelectKBest(score_func=f_classif, k=20)  # Adjust `k` as needed
X_selected = selector.fit_transform(X_train, y_train)
selected_features = X_train.columns[selector.get_support()]
print(f"Selected features: {list(selected_features)}")

# Split training data into training and validation sets


# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Initialize and train Logistic Regression model with L2 regularization
lr_model = LogisticRegression(penalty='l2', solver='liblinear', max_iter=1000, random_state=42)
lr_model.fit(X_train_scaled, y_train)

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
