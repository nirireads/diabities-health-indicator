from models.models import DiabetesModel
import torch
from utils.preprocessing import X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor
from torch.utils.data import Dataset, DataLoader
from config import LEARNING_RATE, EPOCHS, BATCH_SIZE

class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels.long()

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index], self.labels[index]
    
train_dataset = CustomDataset(X_train_tensor, y_train_tensor)
test_dataset = CustomDataset(X_test_tensor,y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = DiabetesModel(num_features=X_train_tensor.shape[1])
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for features, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    epoch_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {epoch_loss:.4f}")


#save trained model
torch.save(model.state_dict(), 'models/diabetes_model.pth')
print("Model saved as diabetes_model.pth")


