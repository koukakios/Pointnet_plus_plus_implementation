import  torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from SetAbsctraction.py import SA

class Pointnet_plus(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.sa1 = SA()
        self.sa2 = SA()
        self.sa3 = SA()
        self.linear1 = torch.nn.Linear()
        self.activation1 = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear()
        self.activation2 = torch.nn.ReLU()
        self.linear3 = torch.nn.Linear()
        self.softmax = torch.nn.Softmax()

    def forward(self, x):
        x = self.sa1(x)
        x = self.sa2(x)
        x = self.sa3(x)
        x = self.linear1(x)
        x = self.activation1(x)
        x = self.linear2(x)
        x = self.activation2(x)
        x = self.linear3(x)
        x = self.softmax(x)
        return x