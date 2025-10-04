import torch
import torch.nn as nn

class DiabetesModel(nn.Module):
    def __init__(self,num_features):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(num_features, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.stage_head = nn.Linear(32, 5)  # Regression

    def forward(self, features):
        output = self.shared(features)
        logits = self.stage_head(output)
        return logits