import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import torch
import joblib

# Load data
df = pd.read_csv('data/diabetes_dataset.csv')

# Identify categorical and numeric columns
categorical_cols = ['gender', 'ethnicity', 'education_level', 'income_level',
                    'employment_status', 'smoking_status', 'diabetes_stage']
numeric_cols = [col for col in df.columns if col not in categorical_cols + ['diagnosed_diabetes', 'diabetes_risk_score']]

# Encode categorical columns
encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

    #save encoders for future use
    encoders[col] = le
    # save each encoder
    joblib.dump(le, f'models/encoders_scaler/{col}_encoder.pkl')




# Define features (X) and multi-output labels (y)
X = df[categorical_cols + numeric_cols].drop(
        columns=['diagnosed_diabetes', 'diabetes_risk_score', 'diabetes_stage'], 
        errors='ignore'
    ).values
y = df['diabetes_stage'].values

# print(f'Feature matrix shape: {X.shape}, Labels shape: {y.shape}')
# print(f'X : {X}, and Y : {y}')

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale numeric features 
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
joblib.dump(scaler, 'models/encoders_scaler/scaler.pkl')

# print(type(X_train), type(y_train))

# Convert to PyTorch tensors
X_train_tensor = torch.from_numpy(X_train).float()
X_test_tensor  = torch.from_numpy(X_test).float()
y_train_tensor = torch.from_numpy(y_train).float()
y_test_tensor  = torch.from_numpy(y_test).float()

# Print shapes to verify
print(f'X_train shape: {X_train_tensor.shape}, y_train shape: {y_train_tensor.shape}')
print(f'X_test shape: {X_test_tensor.shape}, y_test shape: {y_test_tensor.shape}')
print("Preprocessing complete. Encoders and scaler saved in 'models/encoders_scaler/'")