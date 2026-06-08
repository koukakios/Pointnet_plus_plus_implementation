import torch

from models.pointnet_plus import Pointnet_plus
from models.unitPointnetSegm import UnitPointnetSegm


BATCH_SIZE = 32
NUM_POINTS = 1024
DIM_IN = 5
NUM_CLASSES = 3
USE_FPS_FALLBACK = True


def test_segmentation_head_has_linear_logits_output():
    model = UnitPointnetSegm(
        dims_1=[12, 8, 4],
        dims_2=[4, 8, 16],
        dims_3=[20, 8, 4],
        dims_4=[4, 4, NUM_CLASSES],
    )

    assert isinstance(model.mlp_4[-1], torch.nn.Linear)
    assert model.mlp_4[-1].out_features == NUM_CLASSES
    assert not isinstance(model.mlp_4[-1], torch.nn.ReLU)

    with torch.no_grad():
        model.mlp_4[-1].weight.zero_()
        model.mlp_4[-1].bias.copy_(torch.tensor([-1.0, 0.0, 1.0]))
        output = model(torch.randn(2, 7, 12))

    assert output.shape == (2, 7, NUM_CLASSES)
    assert (output[..., 0] < 0).all()


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
    test_segmentation_head_has_linear_logits_output()

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
