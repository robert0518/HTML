import pandas as pd
import numpy as np
import category_encoders as ce
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# 1. Load Data
train_file_path = 'stage2_train_split.csv'
valid_file_path = 'stage2_valid_split.csv'
test_file_path = 'stage2_data/2024_test_data.csv'

data = pd.read_csv(train_file_path)
valid_data = pd.read_csv(valid_file_path)
test_data = pd.read_csv(test_file_path)

# 2. Preprocess Data

# Convert target to integer
data['home_team_win'] = data['home_team_win'].astype(int)
valid_data['home_team_win'] = valid_data['home_team_win'].astype(int)

y_train = data['home_team_win']
y_valid = valid_data['home_team_win']

print("Data loaded successfully.")

# Convert 'date' to datetime
data['date'] = pd.to_datetime(data['date'], errors='coerce')
valid_data['date'] = pd.to_datetime(valid_data['date'], errors='coerce')

# Define categorical columns
categorical_columns = ['is_night_game', 'home_team_abbr', 'away_team_abbr'] 
                       # 'home_pitcher', 'away_pitcher']

# Drop unnecessary columns (ensure commas are correctly placed)
drop_columns_train = ['Unnamed: 0', 'id', 'home_team_season', 'away_team_season', 'home_pitcher', 'away_pitcher',
                      'home_team_win', 'date']
drop_columns_valid = ['Unnamed: 0', 'id', 'home_team_season', 'away_team_season', 'home_pitcher', 'away_pitcher',
                      'home_team_win', 'date']
drop_columns_test = ['id', 'home_team_season', 'away_team_season', 'home_pitcher', 'away_pitcher']

# Split raw features
X_train_raw = data.drop(columns=drop_columns_train)
X_valid_raw = valid_data.drop(columns=drop_columns_valid)
X_test_raw = test_data.drop(columns=drop_columns_test)

# 3. Perform Leave-One-Out Encoding

# Initialize the Leave-One-Out Encoder
encoder = ce.LeaveOneOutEncoder(cols=categorical_columns)

# Fit the encoder only on the training data
encoder.fit(X_train_raw, y_train)

# Transform the training, validation, and test data
X_train_encoded = encoder.transform(X_train_raw)
X_valid_encoded = encoder.transform(X_valid_raw)
X_test_encoded = encoder.transform(X_test_raw)

print("Categorical variables encoded using Leave-One-Out Encoding.")

# 4. Handle Missing Values using KNN Imputation

knn_imputer = KNNImputer(n_neighbors=5)

# Fit on training data and transform
X_train_imputed = knn_imputer.fit_transform(X_train_encoded)
X_valid_imputed = knn_imputer.transform(X_valid_encoded)
X_test_imputed = knn_imputer.transform(X_test_encoded)

# Convert imputed arrays back to DataFrames
X_train = pd.DataFrame(X_train_imputed, columns=X_train_encoded.columns, index=X_train_encoded.index)
X_valid = pd.DataFrame(X_valid_imputed, columns=X_valid_encoded.columns, index=X_valid_encoded.index)
X_test = pd.DataFrame(X_test_imputed, columns=X_test_encoded.columns, index=X_test_encoded.index)

print("Missing values imputed using KNNImputer.")

# 5. Feature Scaling
scaler = StandardScaler()

# Fit on training data and transform
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)
X_test_scaled = scaler.transform(X_test)

# Convert scaled arrays back to DataFrames
X_train = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
X_valid = pd.DataFrame(X_valid_scaled, columns=X_valid.columns, index=X_valid.index)
X_test = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

print("Features scaled using StandardScaler.")

def drop_skew_std_columns(df, dataset_name):
    # Identify columns containing 'skew' or 'std' (case-insensitive)
    columns_to_drop = [col for col in df.columns if 'skew' in col.lower() or 'std' in col.lower()]
    df_dropped = df.drop(columns=columns_to_drop, errors='ignore')
    print(f"Dropped {len(columns_to_drop)} 'skew/std' columns from {dataset_name}: {columns_to_drop}")
    return df_dropped

# Apply the function to training, validation, and test sets
X_train = drop_skew_std_columns(X_train, "X_train")
X_valid = drop_skew_std_columns(X_valid, "X_valid")
X_test = drop_skew_std_columns(X_test, "X_test")

# 6. Feature Selection Based on Random Forest Feature Importances

# Initialize the RandomForestClassifier
rf_model = RandomForestClassifier(
    random_state=42, 
    class_weight='balanced', 
    n_jobs=-1,
    n_estimators=100, 
    max_depth=10
)

# Fit the model on the entire training set
rf_model.fit(X_train, y_train)

print('Model Finish Training.')

# Extract feature importances
feature_importances = pd.Series(rf_model.feature_importances_, index=X_train.columns)
feature_importances = feature_importances.sort_values(ascending=False)

# Plot feature importances (optional)
plt.figure(figsize=(10, 8))
sns.barplot(x=feature_importances.values, y=feature_importances.index)
plt.title('Feature Importances from Random Forest')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.tight_layout()
plt.show()

# Select top N features (e.g., top 20)
top_n = 20
top_features = feature_importances.head(top_n).index.tolist()

print(f"Top {top_n} features selected based on feature importances.")

# Alternatively, set a threshold and select features above that threshold
# threshold = 0.01
# top_features = feature_importances[feature_importances > threshold].index.tolist()

# Update datasets with selected features
X_train_selected = X_train[top_features]
X_valid_selected = X_valid[top_features]
X_test_selected = X_test[top_features]

print(f"Selected features: {top_features}")

# 7. Train with 5-Fold Cross-Validation and Evaluate

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Initialize the RandomForestClassifier (if you prefer to reset it)
rf_model_cv = RandomForestClassifier(
    random_state=42, 
    class_weight='balanced', 
    n_jobs=-1,
    n_estimators=100, 
    max_depth=10
)

# To store out-of-fold predictions (optional)
fold_predictions = np.zeros(len(X_train_selected))

# Cross-validation training
for fold, (train_idx, val_idx) in enumerate(cv.split(X_train_selected, y_train)):
    print(f"Training fold {fold + 1}...")
    
    X_train_fold, X_val_fold = X_train_selected.iloc[train_idx], X_train_selected.iloc[val_idx]
    y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]
    
    rf_model_cv.fit(X_train_fold, y_train_fold)
    
    # Predict on this fold's validation portion
    fold_pred = rf_model_cv.predict(X_val_fold)
    fold_predictions[val_idx] = fold_pred

# Optional: Evaluate cross-validated training performance
train_accuracy = accuracy_score(y_train, fold_predictions)
print(f"Cross-Validated Training Accuracy: {train_accuracy:.4f}")

# Retrain the model on the entire training set with selected features
rf_model.fit(X_train_selected, y_train)

# 8. Evaluate on the Validation Set
valid_pred = rf_model.predict(X_valid_selected)
valid_proba = rf_model.predict_proba(X_valid_selected)[:, 1]

valid_accuracy = accuracy_score(y_valid, valid_pred)
valid_roc_auc = roc_auc_score(y_valid, valid_proba)
valid_classification_metrics = classification_report(y_valid, valid_pred)

print(f"Validation Accuracy: {valid_accuracy:.4f}")
print(f"Validation ROC-AUC: {valid_roc_auc:.4f}")
print("Classification Report:")
print(valid_classification_metrics)

# 9. Predict on Test Set (using the model retrained on the entire training set)
test_predictions = rf_model.predict(X_test_selected)

# Output and save predictions
output = pd.DataFrame({
    'id': test_data['id'],
    'home_team_win': test_predictions.astype(bool)
})

output_file = 'final_test_predictions_stage2.csv'
output.to_csv(output_file, index=False)
print(f"Test predictions saved to '{output_file}'.")

print("\nAll steps completed.")
