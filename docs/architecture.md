# Architecture Overview

## PointNet and PointNet++

PointNet is a pioneering deep learning architecture for processing unordered point clouds directly, without requiring conversion to regular grids like voxels or images. It achieves permutation invariance through symmetric functions (max pooling) that aggregate features across points.

PointNet++ extends this with hierarchical feature learning. It uses set abstraction layers that sample points, group local neighborhoods, and apply PointNet-style processing to extract local features at multiple scales.

## Why Permutation-Invariant Processing?

Point clouds are unordered sets of 3D points, unlike images with fixed pixel arrangements. Traditional CNNs assume spatial regularity, so PointNet uses shared MLPs and global max pooling to ensure the output doesn't depend on point ordering.

## Role of Set Abstraction

Set abstraction is the core of PointNet++. It:
1. Samples a subset of points (farthest point sampling)
2. Groups neighboring points around each sampled point
3. Applies local feature learning via shared MLPs
4. Aggregates features with max pooling

This creates a hierarchical representation from fine to coarse scales.

## Mapping to This Repository

- `pointnet.py`: Implements the baseline PointNet architecture with shared MLPs and max pooling.
- `pointnet_plus.py`: Combines multiple set abstraction layers with a final PointNet classifier.
- `SetAbstraction.py`: Defines the SA module for sampling, grouping, and local feature extraction.
- `MLP.py`: Utility for building multi-layer perceptrons (currently minimal).
- `Tnet.py`: Optional transformation network for input/feature alignment (inspired by original PointNet).

The implementation focuses on radar point-cloud clutter detection, adapting PointNet++ concepts for sparse, noisy radar data in autonomous driving scenarios.