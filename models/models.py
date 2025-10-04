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

        self.diabites_head = nn.Linear(32, 1)  # Binary classification
        self.risk_head = nn.Linear(32, 1)  # Regression
        self.stage_head = nn.Linear(32, 1)  # Regression

    def forward(self, features):
        output = self.shared(features)
        diabetes_out = torch.sigmoid(self.diabites_head(output))  # Sigmoid for binary classification
        risk_out = self.risk_head(output)  # Linear for regression
        stage_out = self.stage_head(output)  # Linear for regression
        return torch.cat([diabetes_out, risk_out, stage_out], dim=1)    