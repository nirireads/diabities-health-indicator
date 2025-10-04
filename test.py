import torch
from models.models import DiabetesModel
from utils.preprocessing import X_test_tensor, y_test_tensor
from torch.utils.data import TensorDataset, DataLoader
from config import BATCH_SIZE
from sklearn.metrics import accuracy_score, mean_squared_error

# --- Load model ---
num_features = X_test_tensor.shape[1]
model = DiabetesModel(num_features=num_features)
model.load_state_dict(torch.load('models/diabetes_model.pth'))
model.eval()

# --- Prepare test DataLoader ---
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# --- Collect predictions ---
all_predictions = []
all_labels = []

with torch.no_grad():
    for features, labels in test_loader:
        outputs = model(features)
        all_predictions.append(outputs)
        all_labels.append(labels)
    
all_predictions = torch.cat(all_predictions, dim=0)
all_labels = torch.cat(all_labels, dim=0)

print("Predictions shape:", all_predictions.shape)
print("Actual Labels shape:", all_labels.shape)

# --- Compute metrics ---

# 1️⃣ Binary diabetes prediction
pred_diabetes = (all_predictions[:,0] > 0.5).int()
true_diabetes = all_labels[:,0].int()
accuracy_diabetes = accuracy_score(true_diabetes, pred_diabetes)

# 2️⃣ Diabetes risk score (continuous)
pred_risk = all_predictions[:,1]
true_risk = all_labels[:,1]
mse_risk = mean_squared_error(true_risk, pred_risk)

# 3️⃣ Diabetes stage (categorical)
pred_stage = torch.round(all_predictions[:,2]).int()
true_stage = all_labels[:,2].int()
accuracy_stage = accuracy_score(true_stage, pred_stage)

# --- Print results ---
print(f"Diabetes Prediction Accuracy: {accuracy_diabetes:.4f}")
print(f"Diabetes Risk Score MSE: {mse_risk:.4f}")
print(f"Diabetes Stage Accuracy: {accuracy_stage:.4f}")
