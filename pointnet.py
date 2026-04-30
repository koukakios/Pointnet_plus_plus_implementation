import  torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from Tnet.py import TNet

class Pointnet(torch.nn.Module):
    def __init__(self):
        super(Pointnet, self).__init__()
        #Tnet1
        self.Tnet1 = TNet(k = 3)

        #mlp 3 -> 64 -> 64
        self.lin1 = torch.nn.Linear(3, 64)
        self.activation1 = torch.nn.ReLU()
        self.lin2 = torch.nn.Linear(64, 64)
        self.activation2 = torch.nn.ReLU()

        #Tnet2
        self.Tnet2 = TNet(k = 64)

        #mlp 64-> 128 -> 1024
        self.lin3 = torch.nn.Linear(64, 128)
        self.activation3 = torch.nn.ReLU()
        self.lin4 = torch.nn.Linear(128, 1024)
        self.activation4 = torch.nn.ReLU()

        #maxpool

    def forward(self, x):
        #x (B, M, K, D)

        x = self.Tnet1.forward(x)

        x = self.lin1(x)
        x = self.activation1(x)
        x = self.lin2(x)
        x = self.activation2(x)

        x = self.Tnet2.forward(x)

        x = self.lin3(x)
        x = self.activation3(x)
        x = self.lin4(x)
        x = self.activation4(x)

        x = x.max(dim = 2, keepdim = True)

        return x
