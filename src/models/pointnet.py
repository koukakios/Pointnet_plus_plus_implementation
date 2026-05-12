import torch
#from .Tnet import TNet


class Pointnet(torch.nn.Module):
    def __init__(self, dims_1, dims_2):
        super(Pointnet, self).__init__()

        # Optional T-Net for input alignment.
        # Not needed yet for minimal PointNet++.
        # self.tnet1 = TNet(k=dims1[0])

        #define first mlp
        layers = []
        for v, w in zip(dims_1, dims_1[1:]):
            layers.append(torch.nn.Linear(v, w))
            layers.append(torch.nn.ReLU())

        self.mlp_1 = torch.nn.Sequential(*layers)

        # Optional feature T-Net.
        # In full PointNet this aligns 64D features.
        # Skip for now until the basic model runs.
        # self.tnet2 = TNet(k=64)

        #define second mlp
        layers = []
        for v, w in zip(dims_2, dims_2[1:]):
            layers.append(torch.nn.Linear(v, w))
            layers.append(torch.nn.ReLU())

        self.mlp_2 = torch.nn.Sequential(*layers)


def forward(self, x):
        # x: (B, M, K, dim_in)
        # B = batch size
        # M = sampled center points
        # K = neighbours per center, 32 by default
        # dim_in = point feature dimension

        # Optional input transform:
        # x = self.tnet1(x)

        x = self.mlp_1(x)

        # Optional feature transform:
        # x = self.tnet2(x)

        x = self.mlp_2(x)

        # Max-pool over neighbours K
        x = x.max(dim=2)[0]   # (B, M, dim_out)

        return x