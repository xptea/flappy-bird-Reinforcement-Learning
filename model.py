import torch
import torch.nn as nn
import torch.nn.functional as F

class FlappyNet(nn.Module):
    def __init__(self):
        super(FlappyNet, self).__init__()
        self.fc1 = nn.Linear(180, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)