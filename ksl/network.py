"""Optional PyTorch dependency; data preparation only needs NumPy."""
import torch
from torch import nn


class SequenceGRU(nn.Module):
    def __init__(self,classes,hidden=64):
        super().__init__()
        self.gru=nn.GRU(90,hidden,num_layers=2,batch_first=True,dropout=.2)
        self.head=nn.Linear(hidden,classes)
    def forward(self,x):
        _,state=self.gru(x)
        return self.head(state[-1])
