import  torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch_geometric.nn import fps
from pointnet import Pointnet

class SA(torch.nn.Module):
    def __init__(self):
        super(SA, self).__init__()
        self.pointnet = Pointnet()

    def forward(self, x):
        #x (B, N, D)
        B, N, D = x.shape

        #do FPS sampling
        sampled_points = self.sample_fps(x, ratio = 0.25)
        M = sampled_points.shape[1]

        #do kNN
        dist = torch.cdist(sampled_points, x)
        idx = dist.topk(k=32, largest=False)[1]

        # make groups
        x_expanded = x.unsqueeze(1).expand(-1, M, -1, -1)  # (B, M, N, D)
        idx_expanded = idx.unsqueeze(-1).expand(-1, -1, -1, D)  # (B, M, 32, D)
        groups = torch.gather(x_expanded, 2, idx_expanded)

        #groups (B, M, K, D) K = 32 (neighbours in our case)
        #B is batchsize
        #M is number of groups
        #Each group has K points and D dimensions
        #apply pointnet in all groups
        result = self.pointnet.forward(groups)

        


    def sample_fps(self, x, ratio = 0.25):
        B, N, D = x.shape

        x_flat = x.reshape(B*N, D)

        batch = torch.arange(B, device = x.device).repeat_interleave(N)

        idx = fps(x_flat, batch = batch, ratio = ratio)

        sampled_flat = x_flat[idx]

        M = sampled_flat.shape[0]
        sampled_points = sampled_flat.reshape(B, M, D)
        return sampled_points