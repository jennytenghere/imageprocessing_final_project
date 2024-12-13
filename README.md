## Note on Video Content:  
# Some intended commentary was cut around the 5:54 mark in the accompanying video. The missing statement was:
# "Compressing to 1/4 of the side length greatly decreases the processing time while not increasing too much loss."
Overview

This project consists of two primary Python scripts designed to create photomosaics and perform efficient image clustering:

    A.photomosaic.py: Generates a photomosaic by matching and placing tile images onto a target image.
    B.KMeans.py: Performs K-Means clustering on image data using the cuML library for accelerated processing.



This highlights the trade-off between faster processing and image quality when applying compression ratios to tile images.
Table of Contents

    A.photomosaic.py
        Description
        Usage
        Command-Line Arguments
        Example
    B.KMeans.py
        Description
        Usage
        Command-Line Arguments
        Example
    Additional Notes
    Getting Started
    Requirements
    License
    Acknowledgments

A.photomosaic.py
Description

A.photomosaic.py creates a photomosaic by dividing a target image into a grid and replacing each cell with a tile image selected from a large dataset. The tile images are provided in a .pt file rather than a directory of images. Multiprocessing is used to speed up the tile-matching process, making it suitable for large datasets.
Usage

python A.photomosaic.py --tgt_pt <TARGET_IMAGE_PT> --used_pt <TILE_IMAGES_PT> --n_row <ROWS> --n_col <COLS> [OPTIONS]

Command-Line Arguments

    --tgt_pt (required): Path to the .pt file containing the target image. The tensor should have shape (N, H, W, C) with N=1.
    --used_pt (required): Path to the .pt file containing tile images (N, H, W, C), for example cifar10_data_50000_32_32.pt.
    --n_row (required): Number of rows in the resulting photomosaic.
    --n_col (required): Number of columns in the resulting photomosaic.

Optional Arguments:

    --exhibit: Display intermediate or final mosaic output.
    --query_Cspace {rgb, yiq, xyz}: Color space for querying tiles. Default rgb.
    --query_transform {none, fourier, wavelet}: Transform to apply for querying. Default none.
    --query_compress_ratio {1,2,4,8,16,32}: Compression ratio for tile querying. Default 1.
    --query_is_fitness: Use fitness-based querying.
    --evaluate_Cspace {rgb, yiq, xyz}: Color space for evaluation. Default rgb.
    --evaluate_is_fitness: Use fitness-based evaluation.
    --save_plots: Save plots generated during mosaic creation.
    --output_dir: Directory to save results (default: ./output).

Example

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

B.KMeans.py
Description

B.KMeans.py applies K-Means clustering to a set of tile images provided in a .pt file. Utilizing the cuML library for GPU acceleration, it can efficiently cluster large numbers of images. This clustering step can be used to pre-select representative tiles or enhance the speed and quality of the photomosaic generation process.
Usage

python B.KMeans.py --used_pt <TILE_IMAGES_PT> --extract <NUM_TILES> [OPTIONS]

Command-Line Arguments

    --used_pt (required): Path to the .pt file with tile images (N, H, W, C).
    --extract (required): Number of tiles to extract and cluster.

Optional Arguments:

    --select_Cspace {rgb, yiq, xyz}: Color space for selection. Default rgb.
    --select_transform {none, fourier, wavelet}: Transform applied for selection. Default none.
    --select_compress_ratio {1,2,4,8,16,32}: Compression ratio for selection. Default 1.
    --output_name: Name of the output .pt file containing cluster results.

Example

python B.KMeans.py --used_pt cifar10_data_50000_32_32.pt \
                   --extract 1000 \
                   --select_Cspace rgb \
                   --select_transform none \
                   --select_compress_ratio 1 \
                   --output_name cluster.pt

Additional Notes

Remember that compression can significantly speed up processing. The segment of the video cut at 5:54 intended to highlight that reducing each tile image to 1/4 of its side length can greatly decrease computation time without notably increasing the overall image loss.
Getting Started

    Clone the Repository:

git clone <repository_url>
cd <repository_directory>

Install Dependencies:

pip install -r requirements.txt

Ensure cuML is properly installed for GPU-accelerated K-Means (see cuML documentation).

Prepare Tile Data:

Place your .pt files (e.g., cifar10_data_50000_32_32.pt) in an accessible directory.

Run K-Means:

python B.KMeans.py --used_pt cifar10_data_50000_32_32.pt --extract 1000 --select_Cspace rgb --select_transform none --select_compress_ratio 1 --output_name cluster.pt

Generate Photomosaic:

    python A.photomosaic.py --tgt_pt data_9_1024_1024.pt --used_pt cifar10_data_50000_32_32.pt --n_row 50 --n_col 50 --exhibit --query_Cspace rgb --query_transform none --query_compress_ratio 1 --query_is_fitness --evaluate_Cspace rgb --evaluate_is_fitness --save_plots --output_dir ./output

Requirements

    Python 3.x
    PyTorch
    cuML (for K-Means acceleration)
    NumPy
    PIL (Pillow)
    OpenCV
    tqdm
    matplotlib

A CUDA-compatible GPU is required for B.KMeans.py to leverage cuML’s GPU acceleration.

This project is licensed under the MIT License. See the LICENSE file for details.
Acknowledgments

    cuML: GPU-accelerated ML algorithms.
    OpenCV: Image processing tools.
    Pillow: Image handling in Python.
    NumPy & SciPy: Numerical and scientific computing.
    tqdm: Progress bars for command-line interfaces.


