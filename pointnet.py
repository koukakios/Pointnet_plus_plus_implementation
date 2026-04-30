import torch
#from Tnet import TNet


class Pointnet(torch.nn.Module):
    def __init__(self, dim_in, dim_out):
        super(Pointnet, self).__init__()

        self.dim_in = dim_in
        self.dim_out = dim_out

        # Optional T-Net for input alignment.
        # Not needed yet for minimal PointNet++.
        # self.tnet1 = TNet(k=dim_in)

        self.mlp1 = torch.nn.Sequential(
            torch.nn.Linear(dim_in, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
        )

        # Optional feature T-Net.
        # In full PointNet this aligns 64D features.
        # Skip for now until the basic model runs.
        # self.tnet2 = TNet(k=64)

        self.mlp2 = torch.nn.Sequential(
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, dim_out),
            torch.nn.ReLU(),
        )

    def forward(self, x):
        # x: (B, M, K, dim_in)
        # B = batch size
        # M = sampled center points
        # K = neighbours per center
        # dim_in = point feature dimension

        # Optional input transform:
        # x = self.tnet1(x)

        x = self.mlp1(x)

        # Optional feature transform:
        # x = self.tnet2(x)

        x = self.mlp2(x)

        # Max-pool over neighbours K
        x = x.max(dim=2)[0]   # (B, M, dim_out)

        return x