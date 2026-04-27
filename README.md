# Pb4U-GNet: Resolution-Adaptive Garment Simulation via Propagation-before-Update Graph Network

### <img align=center src=./static/icons/paper.png width='24'/> [Paper](https://arxiv.org/abs/2601.15110) &ensp; <img align=center src=./static/icons/project.png width='32'/> [Code](https://github.com/adam-lau709/PB4U-GNet)

This is the official training and inference code for the paper [**"Pb4U-GNet: Resolution-Adaptive Garment Simulation via Propagation-before-Update Graph Network"**](https://github.com/adam-lau709/PB4U-GNet) (AAAI 2026).

Pb4U-GNet is a resolution-adaptive garment simulation framework built on graph neural networks. It decouples message propagation from feature updates, enabling generalisation to mesh resolutions unseen during training.

**Key capabilities:**

- Trained on low-resolution garment meshes, generalises to meshes with higher resolutions without retraining
- Dynamically adjusts message-passing depth based on mesh density (resolution-aware propagation control)
- Scales predicted vertex accelerations according to geometric scale (resolution-aware update scaling)
- Self-supervised training using physics-based loss terms — no ground-truth simulation data required

---

## Installation

### Install conda environment

A conda environment file `pb4u.yml` is provided to install all dependencies. Create and activate the environment with:

```bash
conda env create -f environment.yml
conda activate pb4u
```

To build the environment from scratch:

<details>
  <summary>Build environment from scratch</summary>

```bash
# Create and activate a new environment
conda create -n pb4u python=3.9 -y
conda activate pb4u

# Install PyTorch (see https://pytorch.org/)
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia -y

# Install PyTorch Geometric (see https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)
conda install pyg -c pyg -y

# Install PyTorch3D (see https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md)
conda install -c fvcore -c iopath -c conda-forge fvcore iopath -y
conda install -c bottler nvidiacub -y
conda install pytorch3d -c pytorch3d -y

# Install auxiliary packages with conda
conda install -c conda-forge munch pandas tqdm omegaconf matplotlib einops ffmpeg -y

# Install more auxiliary packages with pip
pip install smplx aitviewer chumpy huepy

# Create a new kernel for Jupyter notebook
conda install ipykernel -y; python -m ipykernel install --user --name pb4u --display-name "pb4u"
```

</details>

---

## Download Data

### Auxiliary Data

Download the auxiliary data using this [link](https://drive.google.com/file/d/1rnFQMJ70EeSNSzIhpIXmGmm0vSvL8KA8/view?usp=sharing). Create a `data` folder in the project root then unpack the downloaded zip file into this folder.

The environment variable `$PB4U_DATA` should be set to the path to this `data` folder.

### SMPL Models

Download SMPL models from [https://smpl.is.tue.mpg.de/](https://smpl.is.tue.mpg.de/) and unpack them into `$PB4U_DATA/aux_data/smpl`.

### File structure

Your `$PB4U_DATA` folder should have the following structure:

```
$PB4U_DATA
    |-- aux_data
        |-- datasplits/          # CSV data splits for training
        |-- smpl/                # SMPL body models
            |-- SMPL_NEUTRAL.pkl
            |-- SMPL_FEMALE.pkl
            |-- SMPL_MALE.pkl
        |-- garment_meshes/      # .obj meshes for supported garments
        |-- garments_dict.pkl    # garment meshes and auxiliary data for training/inference
        |-- garments_dict_multi_res  # equivalent to garments_dict, but includes multiple mesh resolutions per garment
        |-- garments_dict_multi_res_hier.pkl  # hierarchical version of garments_dict_multi_res, suitable for use with HOOD
        |-- garments_dict_high_res  # highest-resolution garment meshes used in the paper
        |-- smpl_aux.pkl         # SMPL vertex indices for hands (used to avoid self-intersections)
    |-- trained_models/
        |-- pb4u.pth             # Pb4U-GNet model
        |-- hood.pth             # HOOD model (baseline)
        |-- fine15.pth           # Fine15 baseline (15 message-passing steps, no long-range edges)
        |-- fine48.pth           # Fine48 baseline (48 message-passing steps, no long-range edges)
```

---

## Inference, Validation and Rendering

[Inference.ipynb](Inference.ipynb) walks through inference and validation on a garment mesh and pose sequence, and how to export frames and a video from the rollout.

## Training

To train a new Pb4U-GNet model from scratch, first download the [VTO dataset](https://github.com/isantesteban/vto-dataset) and convert it to the required format.

You can prepare the dataset with:

```bash
python prepare_dataset.py --vto-dataset-path /path/to/vto-dataset
```

This converts SMPL pose sequences from `tshirt/simulations` into `$PB4U_DATA/vto_dataset/smpl_parameters`.

The main training script is `train.py`. To train using the Pb4U-GNet configuration:

```bash
python train.py config=pb4u
```

Key training hyperparameters (set in `configs/pb4u.yaml`):

| Parameter                    | Value                             |
| ---------------------------- | --------------------------------- |
| Learning rate                | 5e-5                              |
| LR decay steps               | 50,000                            |
| LR decay rate                | 0.5                               |
| Message-passing steps (base) | 15                                |
| Batch size                   | 1                                 |

---

## Repository Structure

This repository is an extension of the code repository of [HOOD](https://github.com/dolorousrtur/hood) and has the same structure. Feel free to refer to the original repository for more information as well.

This repository has a modular structure where you can swap implementations by changing the configuration file. See [RepoIntro.md](RepoIntro.md) for full details.

```
pb4u/
    configs/          # OmegaConf YAML configuration files
        pb4u.yaml     # Main Pb4U-GNet config
        hood.yaml     # HOOD baseline config
        ...
    datasets/         # Data loading and preprocessing modules
        pb4u.py       # Pb4U-GNet dataset module
        ...
    models/           # GNN model wrappers (build input graph, run GNN, return accelerations)
        pb4u.py       # Pb4U-GNet model
        core/         # Core GNN architectures (EncodeProcessDecode)
            pb4u.py   # Pb4U core model
            hood.py   # HOOD core model
            ...
    runners/          # Training and validation loop modules
        pb4u.py       # Pb4U-GNet runner (forward, valid_rollout, optimizer)
        ...
    utils/            # Shared utility functions
    criterions/       # Physics-based loss terms
    train.py          # Main training script
    inference.ipynb   # Standard inference notebook
```

[OmegaConf](https://omegaconf.readthedocs.io) is used to manage configuration files and [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/) is used for graph data handling and message-passing.

---

## Citation

If you use this repository in your research, please cite:

```bibtex
@inproceedings{liu2026pb4u,
  author    = {Liu, Aoran and Hu, Kun and Mo, Clinton Ansun and Wu, Qiuxia and Kang, Wenxiong and Wang, Zhiyong},
  title     = {{Pb4U-GNet}: Resolution-Adaptive Garment Simulation via Propagation-before-Update Graph Network},
  booktitle = {AAAI Conference on Artificial Intelligence},
  year      = {2026},
}
```

This codebase is built on [HOOD](https://github.com/dolorousrtur/hood) (Grigorev et al., CVPR 2023) and [Learning-Based Animation of Clothing for Virtual Try-On](https://github.com/isantesteban/vto-learning-based-animation) (Santesteban et al., Eurographics 2019). Please also cite:

```bibtex
@inproceedings{grigorev2022hood,
  author = {Grigorev, Artur and Thomaszewski, Bernhard and Black, Michael J. and Hilliges, Otmar},
  title  = {{HOOD}: Hierarchical Graphs for Generalized Modelling of Clothing Dynamics},
  booktitle = {Computer Vision and Pattern Recognition (CVPR)},
  year   = {2023},
}

@article {santesteban2019virtualtryon,
    journal = {Computer Graphics Forum (Proc. Eurographics)},
    title = {{Learning-Based Animation of Clothing for Virtual Try-On}},
    author = {Santesteban, Igor and Otaduy, Miguel A. and Casas, Dan},
    year = {2019},
    ISSN = {1467-8659},
    DOI = {10.1111/cgf.13643}
}
```