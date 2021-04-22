
import torch.nn as nn
import torch.nn.functional as F

# Neural network model used consisted on 4 fully connected layers

class LinNet(nn.Module):
    def __init__(self):
        super(LinNet, self).__init__()
        self.fc1 = nn.Linear(30, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return F.log_softmax(x, dim=1)  
