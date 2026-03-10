import torch
import torch.nn as nn
import torch.nn.functional as F


class PTBXL_CNN_Medium(nn.Module):

    def __init__(self):

        super().__init__()

        self.conv1 = nn.Conv1d(12,16,kernel_size=5)
        self.conv2 = nn.Conv1d(16,32,kernel_size=5)

        self.pool = nn.MaxPool1d(2)

        self.fc1 = nn.Linear(32*247,64)
        self.fc2 = nn.Linear(64,5)

    def forward(self,x):

        x = x.permute(0,2,1)

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        x = x.flatten(1)

        x = F.relu(self.fc1(x))

        return self.fc2(x)