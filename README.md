# Official code for SINDER: Repairing the Singular Defects of DINOv2 (ECCV 2024 Oral)

[![🦢 - Paper](https://img.shields.io/badge/🦢-Paper-red)](https://arxiv.org/abs/2407.16826)

![SINDER](./resources/high_norm.jpg)

## Citation

```bibtex
@InProceedings{Haoqi_2024_ECCV,
author = {Wang, Haoqi and Zhang, Tong and Salzmann, Mathieu},
title = {{SINDER}: Repairing the Singular Defects of DINOv2},
booktitle = {Proceedings of the European Conference on Computer Vision (ECCV)},
month = {September},
year = {2024}
}
```

## Install

```bash
conda env create -f environment.yml
conda activate sinder
pip install -e .
```

## Train

Put ImageNet-1K dataset at `data/imagenet` folder.

```bash
./main.py
```

## Visualize

```bash
# visualize original dinov2
./visualize.py resources/example.jpg

# visualize sinder, ckpt download link below
./visualize.py resources/example.jpg --checkpoint path/to/sinder.pth
```

## Checkpoints

[Google Drive](https://drive.google.com/file/d/1g0Aq5qXYuMmVrN9-gGwC9ybwlCDFAw-l/view?usp=sharing)
