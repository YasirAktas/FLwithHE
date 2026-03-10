import torch
import torch.nn as nn
import torch.nn.functional as F


class PTBXL_CNN_Large(nn.Module):

    def __init__(self):

        super().__init__()

        self.conv1 = nn.Conv1d(12,32,7)
        self.conv2 = nn.Conv1d(32,64,5)
        self.conv3 = nn.Conv1d(64,128,5)

        self.pool = nn.MaxPool1d(2)

        self.fc1 = nn.Linear(128*121,256)
        self.fc2 = nn.Linear(256,64)
        self.fc3 = nn.Linear(64,5)

    def forward(self,x):

        x = x.permute(0,2,1)

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        x = x.flatten(1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return self.fc3(x)