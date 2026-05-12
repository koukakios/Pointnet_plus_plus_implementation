# PointNet++ for Radar Point-Cloud Clutter Detection

A PyTorch implementation attempt of PointNet and PointNet++-style architectures for clutter detection on RadarScenes radar point-cloud data.

## Project Status

This is a personal implementation and learning project exploring PointNet and PointNet++ ideas for radar point-cloud clutter detection. The code is not presented as a complete production-ready perception stack. Some scripts are experimental and may depend on local dataset paths.

## Motivation

Radar point clouds are sparse, noisy, and irregular compared with camera images or LiDAR scans. This makes clutter detection a useful problem for autonomous driving and perception research.

PointNet and PointNet++ are relevant because they process unordered point sets directly and can learn spatial features without converting the point cloud into a dense grid.

## Technical Overview

This project provides a PyTorch-based implementation featuring:
- PointNet-style feature extraction
- PointNet++-style set abstraction
- MLP blocks
- Optional T-Net style transformation module
- Dataset creation and preprocessing for RadarScenes
- Training and testing scripts

## Architecture

```
RadarScenes Data
      |
      v
Dataset Creation / Sampling
      |
      v
Point Cloud Tensor
      |
      v
PointNet / PointNet++ Model
      |
      v
Per-point or scene-level feature extraction
      |
      v
Clutter Detection Output
```

PointNet++-style logic:

```
Input Points
      |
      v
Sampling / Grouping
      |
      v
Set Abstraction
      |
      v
Shared MLP
      |
      v
Feature Aggregation
      |
      v
Classifier / Prediction Head
```

## Repository Structure

```
src/
  models/
    MLP.py
    Tnet.py
    pointnet.py
    pointnet_plus.py
    SetAbstraction.py
  data/
    dataset_creation.py
scripts/
  train_file.py
  test.py
docs/
  architecture.md
README.md
requirements.txt
.gitignore
```

## Main Components

| Component | File | Purpose |
|---|---|---|
| MLP block | `src/models/MLP.py` | Shared multi-layer perceptron utilities used in feature extraction |
| T-Net | `src/models/Tnet.py` | Optional learned transformation module inspired by PointNet |
| PointNet | `src/models/pointnet.py` | Baseline PointNet-style model |
| PointNet++ | `src/models/pointnet_plus.py` | Hierarchical point-cloud model using set abstraction |
| Set Abstraction | `src/models/SetAbstraction.py` | Sampling/grouping and local feature aggregation logic |
| Dataset creation | `src/data/dataset_creation.py` | Dataset preparation utilities for RadarScenes |
| Training script | `scripts/train_file.py` | Training entry point |
| Test script | `scripts/test.py` | Testing / debugging entry point |

## Dataset

This project is intended for RadarScenes-style radar point-cloud data. Dataset files are not included in the repository. Local paths may need to be configured manually. Raw data should not be committed, and large generated arrays/checkpoints should be ignored.

## Setup

```bash
git clone https://github.com/koukakios/Pointnet_plus_plus_implementation.git
cd Pointnet_plus_plus_implementation

python -m venv .venv
source .venv/bin/activate   # macOS/Linux
# or
.venv\Scripts\activate      # Windows

pip install -r requirements.txt
```

## Usage

Basic usage examples:

```bash
python scripts/train_file.py
python scripts/test.py
```

Depending on the local dataset layout, paths inside the scripts may need to be updated before running.

## Roadmap

- [x] Implement PointNet-style baseline
- [x] Implement PointNet++-style model components
- [x] Add set abstraction module
- [x] Add training/testing scripts
- [ ] Clean dataset path configuration
- [ ] Add command-line arguments for training settings
- [ ] Add reproducible experiment configuration
- [ ] Add evaluation metrics and saved result summaries
- [ ] Add unit tests for sampling/grouping operations
- [ ] Add documentation for RadarScenes preprocessing
- [ ] Add model checkpointing and experiment logging

## Skills Demonstrated

- PyTorch
- Deep learning for point clouds
- PointNet / PointNet++ architecture
- Radar perception
- Autonomous-driving data processing
- Sampling and grouping for unordered point sets
- Model training pipeline design
- Python project organization

## Limitations

- Experimental implementation
- Dataset not included
- No guaranteed reproducibility yet
- Some paths may be local-machine specific
- Not production-ready
- Not a complete autonomous driving perception stack

## Portfolio Note

This repository is maintained as a portfolio and learning project showing practical work on point-cloud deep learning, radar perception, and PointNet++-style model implementation.