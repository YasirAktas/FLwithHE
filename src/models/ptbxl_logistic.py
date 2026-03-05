import torch.nn as nn

class PTBXL_Logistic(nn.Module):

    def __init__(self, input_dim=100):
        super().__init__()

        self.linear = nn.Linear(input_dim, 1)

    def forward(self,x):
        return self.linear(x)