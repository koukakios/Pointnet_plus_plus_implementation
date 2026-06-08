import torch

from models.pointnet_plus import Pointnet_plus


BATCH_SIZE = 32
NUM_POINTS = 1024
DIM_IN = 5
NUM_CLASSES = 3
USE_FPS_FALLBACK = True


def install_fps_fallback():
    from models import SetAbstraction as sa_module

    def fallback_fps(x, batch, ratio):
        sampled_indices = []

        for batch_id in batch.unique(sorted=True):
            point_indices = (batch == batch_id).nonzero(as_tuple=False).view(-1)
            num_samples = max(1, int(point_indices.numel() * ratio))
            sampled_indices.append(point_indices[:num_samples])

        return torch.cat(sampled_indices, dim=0)

    sa_module.fps = fallback_fps


def main():
    if USE_FPS_FALLBACK:
        install_fps_fallback()

    x = torch.randn(BATCH_SIZE, NUM_POINTS, DIM_IN)
    model = Pointnet_plus(dim_in=DIM_IN, num_classes=NUM_CLASSES)
    model.eval()

    with torch.no_grad():
        output = model(x)

    return output


if __name__ == "__main__":
    main()
