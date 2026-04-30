import torch
from SetAbstraction import SA


class Pointnet_plus(torch.nn.Module):
    def __init__(self, dim_in=3, num_classes=4):
        super(Pointnet_plus, self).__init__()

        self.sa1 = SA(dim_in=dim_in, dim_out=64, ratio=0.25, k=32)
        self.sa2 = SA(dim_in=64, dim_out=128, ratio=0.25, k=32)
        self.sa3 = SA(dim_in=128, dim_out=256, ratio=0.25, k=32)

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x: (B, N, dim_in)

        x = self.sa1(x)   # (B, M1, 64)
        x = self.sa2(x)   # (B, M2, 128)
        x = self.sa3(x)   # (B, M3, 256)

        x = x.max(dim=1)[0]   # (B, 256)

        x = self.classifier(x)  # (B, num_classes)

        return x