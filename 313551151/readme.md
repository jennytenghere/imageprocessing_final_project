# Photomosaic Image Processing

## Video Note

**Important:** Some commentary was intentionally cut from the accompanying video around the 5:54 mark. The missing statement was:

*"Compressing to 1/4 of the side length greatly decreases the processing time while not increasing too much loss."*
## Overview

This project provides a powerful toolkit for creating photomosaics and performing efficient image clustering using Python, PyTorch, and GPU-accelerated processing.

### Key Components

- **A.photomosaic.py**: Generate stunning photomosaics by intelligently replacing image grid cells with tile images
- **B.KMeans.py**: Perform advanced K-Means clustering on image datasets using GPU acceleration

## Features

### Photomosaic Generation
- Flexible grid-based image composition
- Multiple color space and transform options
- Multiprocessing for efficient tile matching
- Configurable compression and fitness-based querying

### K-Means Clustering
- GPU-accelerated clustering using cuML
- Customizable tile selection and color space transformations
- Efficient representative tile extraction

## Requirements

- Python 3.x
- PyTorch
- cuML
- NumPy
- PIL (Pillow)
- OpenCV
- tqdm
- matplotlib
- CUDA-compatible GPU (for K-Means)

## Installation

1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd <repository_directory>
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### K-Means Clustering

```bash
python B.KMeans.py --used_pt cifar10_data_50000_32_32.pt \
                   --extract 1000 \
                   --select_Cspace rgb \
                   --select_transform none \
                   --select_compress_ratio 1 \
                   --output_name cluster.pt
```

### Photomosaic Generation

```bash
python A.photomosaic.py --tgt_pt data_9_1024_1024.pt \
                        --used_pt cifar10_data_50000_32_32.pt \
                        --n_row 50 --n_col 50 \
                        --exhibit \
                        --query_Cspace rgb \
                        --query_transform none \
                        --query_compress_ratio 1 \
                        --query_is_fitness \
                        --evaluate_Cspace rgb \
                        --evaluate_is_fitness \
                        --save_plots \
                        --output_dir ./output
```

## Key Options

### B.KMeans.py Options
- `--used_pt`: Input tile images tensor
- `--extract`: Number of tiles to cluster
- `--select_Cspace`: Color space (rgb, yiq, xyz)
- `--select_transform`: Image transform method

### A.photomosaic.py Options
- `--tgt_pt`: Target image tensor
- `--used_pt`: Tile images tensor
- `--n_row`, `--n_col`: Mosaic grid dimensions
- `--query_Cspace`, `--evaluate_Cspace`: Color spaces
- `--query_transform`: Tile querying transform
- `--query_compress_ratio`: Image compression level


