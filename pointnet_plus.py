import torch
from pointnet import Pointnet
from SetAbstraction import SA
from unitPointnet import UnitPointnet
from unitPointnetSegm import UnitPointnetSegm

class Pointnet_plus(torch.nn.Module):
    def __init__(self, dim_in=3, num_classes=3):
        super(Pointnet_plus, self).__init__()
        self.num_classes = num_classes

        #first set abstraction
        dims_sa1_1st_mlp = [dim_in, 64, 64]
        dims_sa1_2nd_mlp = [dims_sa1_1st_mlp[-1], 64, 128, 1024]
        self.sa1 = SA(dims_sa1_1st_mlp, dims_sa1_2nd_mlp, ratio=0.25, k=32)

        #in the implementation i found they have up to 128 pnts

        #second set abstaction
        dims_sa2_1st_mlp = [dims_sa1_2nd_mlp[-1], 64, 64]
        dims_sa2_2nd_mlp = [dims_sa2_1st_mlp[-1], 64, 128, 1024]
        self.sa2 = SA(dims_sa2_1st_mlp, dims_sa2_2nd_mlp, ratio=0.25, k=32)

        """
        The basis of the pointnet++ stops here. After this we split into segmentation and classification
        """

        """
        Classification
        
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
        """

        """
        Segmentation
        """
        dims_unitPointnet1_1st_mlp = [dims_sa2_2nd_mlp[-1] + dims_sa1_2nd_mlp[-1], 64, 64]
        dims_unitPointnet1_2nd_mlp = [dims_unitPointnet1_1st_mlp[-1], 64, 128, 1024]
        self.unitPointnet_1 = UnitPointnet(dims_unitPointnet1_1st_mlp, dims_unitPointnet1_2nd_mlp)

        dims_unitPointnet2_1st_mlp = [dims_unitPointnet1_2nd_mlp[-1] + dim_in, 64, 64]
        dims_unitPointnet2_2nd_mlp = [dims_unitPointnet2_1st_mlp[-1], 64, 128, 1024]
        dims_unitPointnet2_3rd_mlp = [dims_unitPointnet2_2nd_mlp[0] + dims_unitPointnet2_2nd_mlp[-1], 512, 256, 128]
        dims_unitPointnet2_4th_mlp = [dims_unitPointnet2_3rd_mlp[-1], 128, self.num_classes]
        self.unitPointnetSegm = UnitPointnetSegm(dims_unitPointnet2_1st_mlp, dims_unitPointnet2_2nd_mlp, dims_unitPointnet2_3rd_mlp, dims_unitPointnet2_4th_mlp)

        """
        dims_unitPointnet2_1st_mlp = [dims_unitPointnet1_2nd_mlp[-1] + dims_sa1_2nd_mlp[-1], 64, 64]
        dims_unitPointnet2_2nd_mlp = [dims_unitPointnet1_1st_mlp[-1], 64, 128, 1024]
        self.unitPointnet_1 = UnitPointnet(dims_unitPointnet1_1st_mlp, dims_unitPointnet1_2nd_mlp)
        """

    def forward(self, x):
        # x: (B, N, dim_in) in this case dim_in is x, y, z, etc extra feats

        x_3d = x[:, :, :3]
        x_features = x

        #used later for skip connections to segmentation
        x_3d_1st_layer = torch.clone(x_3d)
        x_features_1st_layer = torch.clone(x_features)

        x_3d, x_features = self.sa1(x_3d, x_features)   # (B, M1, dims_sa1_2nd_mlp[-1])

        # used later for skip connections to segmentation
        x_3d_2nd_layer = torch.clone(x_3d)
        x_features_2nd_layer = torch.clone(x_features)

        x_3d, x_features = self.sa2(x_3d, x_features)   # (B, M2, dims_sa2_2nd_mlp[-1])

        """
        The basis of the pointnet++ stops here. After this we split into segmentation and classification
        """

        """
        classification
        
        #from here onwards we dont need the x_3d as we aint doin any sampling.
        #we take all the features given by the last SetAbstraction and put them in pointnet

        #small tricky to feed all the points to the pointnet and
        #match the dimensions needed from the fwd function of pointnet
        x_expanded = x_features.unsqueeze(1).expand(-1, 1, -1, -1)

        x = self.pointnet(x_expanded)   # (B, 1, dims_pointnet_2nd_mlp[-1])

        x = self.classifier(x[:, 0, :])  # (B, num_classes)
        """

        """
        segmentation
        """
        #interpolate
        #x_3d (B, M, 3)
        #x_features (B, M, dims_sa2_2nd_mlp[-1] + dims of x_features_2nd_layer)
        x_3d, x_features = self.interpolate(x_3d, x_features, x_3d_2nd_layer, x_features_2nd_layer)

        #unit pointnet
        x_features = self.unitPointnet_1(x_features)

        #interpolate
        x_3d, x_features = self.interpolate(x_3d, x_features, x_3d_1st_layer, x_features_1st_layer)

        #unit pointnet for segmentation
        per_class_pnt_logits = self.unitPointnetSegm(x_features)

        return per_class_pnt_logits


    def interpolate(self, x_3d_downsampled, x_features_downsampled,
                    x_3d_original, x_features_original):
        """
        x_3d_downsampled:       (B, N, 3)
        x_features_downsampled: (B, N, D)
        x_3d_original:          (B, M, 3)
        x_features_original:    (B, M, L)

        returns:                (B, M, D + L)
        """

        # Distance from each original point to each downsampled point
        dist = torch.cdist(x_3d_original, x_3d_downsampled)  # (B, M, N)

        # For each original point, get 3 nearest downsampled points
        dist, idx = dist.topk(k=3, dim=-1, largest=False)  # (B, M, 3)

        # Compute inverse-distance weights
        weights = 1.0 / (dist + 1e-8)  # (B, M, 3)
        weights = weights / weights.sum(dim=-1, keepdim=True)  # (B, M, 3)

        # Take the features of those 3 nearest points
        neighbor_feats = x_features_downsampled.gather(
            dim=1,
            index=idx.reshape(idx.shape[0], -1).unsqueeze(-1).expand(-1, -1, x_features_downsampled.shape[-1])
        )

        neighbor_feats = neighbor_feats.reshape(
            idx.shape[0], idx.shape[1], idx.shape[2], x_features_downsampled.shape[-1]
        )  # (B, M, 3, D)

        # Weighted average of the 3 features
        interpolated = (neighbor_feats * weights.unsqueeze(-1)).sum(dim=2)  # (B, M, D)

        # Add skip connection from original features and return
        return x_3d_original, torch.cat([interpolated, x_features_original], dim=-1)  # (B, M, D + L)
