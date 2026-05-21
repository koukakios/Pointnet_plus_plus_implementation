import torch
from SetAbstraction import SA
from pointnet import Pointnet

class Pointnet_plus(torch.nn.Module):
    def __init__(self, dim_in=3, num_classes=3):
        super(Pointnet_plus, self).__init__()

        #first set abstraction
        dims_sa1_1st_mlp = [dim_in, 64, 64]
        dims_sa1_2nd_mlp = [dims_sa1_1st_mlp[-1], 64, 128, 1024]
        self.sa1 = SA(dims_sa1_1st_mlp, dims_sa1_2nd_mlp, ratio=0.25, k=32)

        #second set abstaction
        dims_sa2_1st_mlp = [dims_sa1_2nd_mlp[-1], 64, 64]
        dims_sa2_2nd_mlp = [dims_sa2_1st_mlp[-1], 64, 128, 1024]
        self.sa2 = SA(dims_sa2_1st_mlp, dims_sa2_2nd_mlp, ratio=0.25, k=32)

        #final pointnet in the end on all points practically so no sampling
        dims_pointnet_1st_mlp = [dims_sa2_2nd_mlp[-1], 64, 64]
        dims_pointnet_2nd_mlp = [dims_pointnet_1st_mlp[-1], 64, 128, 1024]
        self.pointnet = Pointnet(dims_pointnet_1st_mlp, dims_pointnet_2nd_mlp)

        dims_to_classifier = dims_pointnet_2nd_mlp[-1]
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(dims_to_classifier, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x: (B, N, dim_in)

        x = self.sa1(x)   # (B, M1, dims_sa1_2nd_mlp[-1])
        x = self.sa2(x)   # (B, M2, dims_sa2_2nd_mlp[-1])

        x = self.pointnet(x)   # (B, 1, dims_pointnet_2nd_mlp[-1])

        x = self.classifier(x)  # (B, num_classes)

        return x