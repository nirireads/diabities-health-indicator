import torch
from models.models import DiabetesModel  
from utils.preprocessing import X_test_tensor, y_test_tensor
from torch.utils.data import TensorDataset, DataLoader
from config import BATCH_SIZE
from sklearn.metrics import accuracy_score, classification_report

# --- Load model ---
num_features = X_test_tensor.shape[1]
model = DiabetesModel(num_features=num_features)
model.load_state_dict(torch.load('models/diabetes_model.pth'))
model.eval()

# --- Prepare test DataLoader ---
test_dataset = TensorDataset(X_test_tensor, y_test_tensor.long())
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

all_predictions = []
all_labels = []

# --- Collect predictions ---
with torch.no_grad():
    for features, labels in test_loader:
        outputs = model(features)                # shape [batch_size, num_classes]
        preds = torch.argmax(outputs, dim=1)    # predicted class
        all_predictions.append(preds)
        all_labels.append(labels)

all_predictions = torch.cat(all_predictions)
all_labels = torch.cat(all_labels)

# --- Metrics ---
accuracy = accuracy_score(all_labels, all_predictions)
report = classification_report(all_labels, all_predictions)

print(f"Diabetes Stage Accuracy: {accuracy:.4f}")
print("\nClassification Report:\n", report)
