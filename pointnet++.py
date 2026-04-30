import  torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from SetAbstraction import SA

class Pointnet_plus(torch.nn.Module):
    def __init__(self):
        super(Pointnet_plus, self).__init__()
        #blocks of Set Abstractions
        self.sa1 = SA()
        self.sa2 = SA()
        self.sa3 = SA()

        #classification part
        #maybe change the numbers, i put them random
        C4 = 10
        C5 = 50
        C6 = 128
        k = 4 #number of classes
        self.linear1 = torch.nn.Linear(C4, C5)
        self.activation1 = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(C5, C6)
        self.activation2 = torch.nn.ReLU()
        self.linear3 = torch.nn.Linear(C6, k)
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