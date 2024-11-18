import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np

# Load dataset
train_file_path = 'train_data.csv'  # Replace with the path to your dataset
train_data = pd.read_csv(train_file_path)
print('Training data loaded successfully')

# REMARK: Test data loading and processing is not used now
# test_file_path = 'same_season_test_data.csv'
# test_data = pd.read_csv(test_file_path)
# print('Testing data loaded successfully')

# Feature selection
# Drop non-numerical and target columns for training data
X = train_data.drop(columns=['home_team_win', 'home_team_abbr', 'away_team_abbr', 'date', 'home_pitcher', 'away_pitcher'])
y = train_data['home_team_win']  # Target variable

# Handle categorical columns
X = pd.get_dummies(X, drop_first=True)
print('Finished processing dummy variables for training data')

# Handle missing values
if X.isnull().sum().sum() > 0:
    print("Missing values detected. Filling missing values with column means.")
    X = X.fillna(X.mean())  # Fill missing values with column means

# Split training data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Split data: {len(X_train)} training samples and {len(X_test)} testing samples.")

# Generate sample weights for training
# Use an index-based approach to prioritize recent games in training
decay_factor = 0.95  # Adjust this decay factor to change the weight importance
weights = np.array([decay_factor ** i for i in range(len(X_train))][::-1])  # Higher weight for more recent games

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train logistic regression model with sample weights
model = LogisticRegression()
model.fit(X_train_scaled, y_train, sample_weight=weights)

# Make predictions for testing data
y_pred = model.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(report)
