import torch
from torch import nn


class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()
        
        self.mse = nn.MSELoss()
    
    def forward(self, pred, y):
        return torch.sqrt(self.mse(pred, y))
