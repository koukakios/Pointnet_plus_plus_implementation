import torch
from pointnet import Pointnet

try:
    from torch_geometric.nn import fps
except ImportError:
    def fps(*_args, **_kwargs):
        raise RuntimeError(
            "torch_geometric is not installed, and no FPS fallback has been installed. "
            "Run train_file.py with --fps-mode fallback or install torch-geometric."
        )


class SA(torch.nn.Module):
    def __init__(self, dims_1st_mlp, dims_2nd_mlp, ratio=0.25, k=32):
        super(SA, self).__init__()
        self.pointnet = Pointnet(dims_1st_mlp, dims_2nd_mlp)
        self.ratio = ratio
        self.knn_neighbours = k

    def forward(self, x_3d, x_features):
        # x_3d: (B, N, 3) 3 dimensions
        # x_features (B, N, D) D features
        """
        I want to ensure that sampling happens only in 3 dimensions, not the ones added by
        the next linear layers. So per point matrix, we keep one that has all the features
        and one with only the 3d coordinates. We sample on the second one and use the indices
        to update the first one.
        """


        B, N, D = x_features.shape

        # centroids_x_3d is (B, M, 3) M is number of centroids
        # centroids_x_features is (B, M, D) D is number of features
        centroids_x_3d, centroids_x_all = self.sample_fps(x_3d, x_features)
        M = centroids_x_3d.shape[1]


        dist = torch.cdist(centroids_x_3d, x_3d)
        idx = dist.topk(k=self.knn_neighbours, largest=False)[1]

        x_3d_expanded = x_3d.unsqueeze(1).expand(-1, M, -1, -1)
        idx_expanded = idx.unsqueeze(-1).expand(-1, -1, -1, 3)
        groups_3d = torch.gather(x_3d_expanded, 2, idx_expanded)
        #groups_3d: (B, M, K, 3) K is 32 by default

        x_features_expanded = x_features.unsqueeze(1).expand(-1, M, -1, -1)
        idx_expanded = idx.unsqueeze(-1).expand(-1, -1, -1, D)
        groups_features = torch.gather(x_features_expanded, 2, idx_expanded)
        # groups_features: (B, M, K, D) K is 32 by default


        #so right now we have per group (M groups in total) K points which have 3 dimensions and D features
        result_3d = centroids_x_3d
        result_features = self.pointnet(groups_features)
        #maybe use also 3d points each time
        # result_3d: (B, M, 3)
        # result_features: (B, M, dimout_pointntet)

        return result_3d, result_features


    def sample_fps(self, x_3d, x_all):
        B, N, D = x_all.shape

        #x_all (B, N, D)
        #x_3d (B, N, 3) always
        x_flat_3d = x_3d.reshape(B * N, x_3d.shape[2]) #x_3d.shape[2] is 3 always
        x_flat_all = x_all.reshape(B * N, x_all.shape[2]) #x_all.shape[2] is N always

        batch = torch.arange(B, device=x_3d.device).repeat_interleave(N)
        idx = fps(x_flat_3d, batch=batch, ratio=self.ratio)

        #extract sampled points from indices
        sampled_flat_x_3d = x_flat_3d[idx]
        sampled_flat_x_all = x_flat_all[idx]

        #reconstruct sampled points
        M = sampled_flat_x_3d.shape[0] // B
        sampled_points_3d = sampled_flat_x_3d.reshape(B, M, sampled_flat_x_3d.shape[1])
        sampled_points_all = sampled_flat_x_all.reshape(B, M, sampled_flat_x_all.shape[1])

        return sampled_points_3d, sampled_points_all
    """

    def sample_fps(self, x):
        B, N, D = x.shape
        M = int(N * self.ratio)

        idx = torch.randperm(N, device=x.device)[:M]
        sampled_points = x[:, idx, :]

        return sampled_points
    """
