from typing import Callable

import torch
from torch import nn
import torch.nn.functional as F


class TNet(nn.Module):
    """
    Transformation network module of PointNet
    """

    def __init__(self, k: int = 64, activation: Callable = F.relu):
        """
        Args:
            k: The number of input dimensions to expect.
            activation: The activation function to use.
        """
        super(TNet, self).__init__()
        self.k = k
        self.act = activation

        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pass a point cloud tensor through the module.
        Args:
            x: The input tensor.
        Returns:
            x: The predicted affine transformation matrix.
        """
        batch_size = x.size()[0]
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        x = self.act(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = self.act(self.bn4(self.fc1(x)))
        x = self.act(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        # Add identity to make x an affine transformation matrix
        I = torch.eye(self.k, dtype=torch.float32, requires_grad=True)
        I = I.view(1, self.k ** 2).repeat(batch_size, 1)
        I = I.to(x.device)
        x = x + I
        x = x.view(-1, self.k, self.k)
        return x