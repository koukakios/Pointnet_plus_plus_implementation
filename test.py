from pathlib import Path
import sys
import traceback

import torch


# Change this to False if you want to test the real torch_geometric FPS backend.
# Keeping it True lets this script test your network shapes even without torch-cluster.
USE_FPS_FALLBACK = True

BATCH_SIZE = 32
NUM_POINTS = 1024
DIM_IN = 5
NUM_CLASSES = 3


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from pointnet_plus import Pointnet_plus  # noqa: E402


def tensor_shapes(value):
    if torch.is_tensor(value):
        return tuple(value.shape)
    if isinstance(value, tuple):
        return tuple(tensor_shapes(item) for item in value)
    if isinstance(value, list):
        return [tensor_shapes(item) for item in value]
    return type(value).__name__


def install_fps_fallback():
    import SetAbstraction as sa_module

    def fallback_fps(x, batch, ratio):
        sampled_indices = []

        for batch_id in batch.unique(sorted=True):
            point_indices = (batch == batch_id).nonzero(as_tuple=False).view(-1)
            num_samples = max(1, int(point_indices.numel() * ratio))
            sampled_indices.append(point_indices[:num_samples])

        return torch.cat(sampled_indices, dim=0)

    sa_module.fps = fallback_fps


def register_shape_hooks(model):
    module_names = [
        "sa1.pointnet.mlp_1",
        "sa1.pointnet.mlp_2",
        "sa1.pointnet",
        "sa1",
        "sa2.pointnet.mlp_1",
        "sa2.pointnet.mlp_2",
        "sa2.pointnet",
        "sa2",
        "pointnet.mlp_1",
        "pointnet.mlp_2",
        "pointnet",
        "classifier",
    ]

    hooks = []

    for name in module_names:
        module = model.get_submodule(name)

        def hook(module, inputs, output, module_name=name):
            print(f"{module_name}")
            print(f"  input : {tensor_shapes(inputs)}")
            print(f"  output: {tensor_shapes(output)}")

        hooks.append(module.register_forward_hook(hook))

    return hooks


def main():
    print("Dummy PointNet++ forward test")
    print(f"torch version: {torch.__version__}")
    print(f"USE_FPS_FALLBACK: {USE_FPS_FALLBACK}")

    if USE_FPS_FALLBACK:
        install_fps_fallback()
        print("Using simple deterministic FPS fallback for this test file.")

    x = torch.randn(BATCH_SIZE, NUM_POINTS, DIM_IN)
    model = Pointnet_plus(dim_in=DIM_IN, num_classes=NUM_CLASSES)
    model.eval()

    print(f"dummy input x: {tuple(x.shape)}")
    print(f"x_3d would be: {tuple(x[:, :, :3].shape)}")
    print(f"x_features would be: {tuple(x.shape)}")

    hooks = register_shape_hooks(model)

    try:
        with torch.no_grad():
            logits = model(x)

        print("Final result")
        print(f"  logits shape: {tuple(logits.shape)}")
        print(f"  logits dtype : {logits.dtype}")
        print(f"  all finite   : {torch.isfinite(logits).all().item()}")
        print(f"  predictions  : {logits.argmax(dim=1).tolist()}")
    except Exception:
        print("Forward pass crashed.")
        traceback.print_exc()
        raise
    finally:
        for hook in hooks:
            hook.remove()


if __name__ == "__main__":
    print("piu")
    main()
