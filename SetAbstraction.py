import torch
from pointnet import Pointnet
from torch_geometric.nn import fps

class SA(torch.nn.Module):
    def __init__(self, dims_2, dims_1, ratio=0.25, k=32):
        super(SA, self).__init__()
        self.pointnet = Pointnet(dims_1, dims_2)
        self.ratio = ratio
        self.k = k

    def forward(self, x):
        # x: (B, N, D)
        B, N, D = x.shape

        sampled_points = self.sample_fps(x)
        M = sampled_points.shape[1]
        #sample points is B, M, D

        dist = torch.cdist(sampled_points, x)
        idx = dist.topk(k=self.k, largest=False)[1]

        x_expanded = x.unsqueeze(1).expand(-1, M, -1, -1)
        idx_expanded = idx.unsqueeze(-1).expand(-1, -1, -1, D)
        groups = torch.gather(x_expanded, 2, idx_expanded)
        #groups: (B, M, K, D) K is 32 by default

        result = self.pointnet(groups)
        # result: (B, M, dim_out)

        return result


    def sample_fps(self, x):
        B, N, D = x.shape

        x_flat = x.reshape(B * N, D)
        batch = torch.arange(B, device=x.device).repeat_interleave(N)

        idx = fps(x_flat, batch=batch, ratio=self.ratio)

        sampled_flat = x_flat[idx]

        M = sampled_flat.shape[0] // B
        sampled_points = sampled_flat.reshape(B, M, D)

        return sampled_points
    """

    def sample_fps(self, x):
        B, N, D = x.shape
        M = int(N * self.ratio)

        idx = torch.randperm(N, device=x.device)[:M]
        sampled_points = x[:, idx, :]

        return sampled_points
    """