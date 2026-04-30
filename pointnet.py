import  torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from Tnet import TNet

class Pointnet(torch.nn.Module):
    def __init__(self, dim_in, dim_out):
        super(Pointnet, self).__init__()

        #in - out dimensions
        self.dim_in = dim_in
        self.dim_out = dim_out

        #Tnet1
        self.Tnet1 = TNet(k = dim_in)

        """
        So in pointnet basic u always begin with fixed dimensions and go to 1024 (in the paper it is 
        3 -> (blah, blah) -> 1024.
        But in our case this 1024 dimensional vector becomes an input to the next pointnet in the next SetAbstraction
        layer. SO we cant fix the input and output dimension. So my guess is that these are hyperparameters that
        we take as input and we can ,fingers crossed, tune later.
        
        Then the question is what to do with the in between dimensions. I say they are multiples or somehow have to do
        with the dim_in and should end up always to dim_out. I guess extra hyperparam tuning...
        """
        #mlp dim_in -> idk -> idk
        self.lin1 = torch.nn.Linear(dim_in, 64)
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
