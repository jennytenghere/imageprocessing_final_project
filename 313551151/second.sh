#!/bin/bash

# Script: run_photomosaic_variations.sh
# Description: Executes KMeans.py with various parameter combinations, followed by photomosiac2.py
# Usage: ./run_photomosaic_variations.sh
# Make sure to give execute permissions: chmod +x run_photomosaic_variations.sh

# -------------------------------
# Base Command Components
# -------------------------------

# Python interpreter path
PYTHON_PATH="/home/ccwu/miniforge3/bin/python"

# Script paths
KMEANS_SCRIPT="/home/ccwu/PycharmProjects/DLP_pythonProject/DIP_final_project/KMeans.py"
PHOTOMOSAIC_SCRIPT="/home/ccwu/PycharmProjects/DLP_pythonProject/DIP_final_project/photomosiac2.py"

# Input files
USED_PT_BASE="cifar10_data_50000_32_32.pt"       # Base used_pt for KMeans.py
TGT_PT="data_9_1024_1024.pt"                    # Target images for photomosiac2.py

# Fixed parameters for KMeans.py
EXTRACT=1000

# Fixed parameters for photomosiac2.py
N_ROW=50
N_COL=50
EXHIBIT="--exhibit"
QUERY_CSPACE_VANILLA="rgb"
QUERY_TRANSFORM_VANILLA="none"
QUERY_COMPRESS_RATIO_VANILLA=1
QUERY_IS_FITNESS="--query_is_fitness"
EVALUATE_CSPACE="rgb"
EVALUATE_IS_FITNESS="--evaluate_is_fitness"
SAVE_PLOTS="--save_plots"

# -------------------------------
# Vanilla Command
# -------------------------------

echo "Running Vanilla KMeans.py and photomosiac2.py..."
mkdir -p "./output_vanilla"

# Run KMeans.py (Vanilla)
$PYTHON_PATH $KMEANS_SCRIPT \
    --used_pt "$USED_PT_BASE" \
    --extract "$EXTRACT" \
    --select_Cspace "rgb" \
    --select_transform "None" \
    --select_compress_ratio 1 \
    --output_name "cluster_vanilla.pt"

# Run photomosiac2.py (Vanilla)
$PYTHON_PATH $PHOTOMOSAIC_SCRIPT \
    --tgt_pt "$TGT_PT" \
    --used_pt "cluster_vanilla.pt" \
    --n_row "$N_ROW" \
    --n_col "$N_COL" \
    $EXHIBIT \
    --query_Cspace "$QUERY_CSPACE_VANILLA" \
    --query_transform "$QUERY_TRANSFORM_VANILLA" \
    --query_compress_ratio "$QUERY_COMPRESS_RATIO_VANILLA" \
    $QUERY_IS_FITNESS \
    --evaluate_Cspace "$EVALUATE_CSPACE" \
    $EVALUATE_IS_FITNESS \
    $SAVE_PLOTS \
    --output_dir "./output_vanilla"

echo "Vanilla run completed. Results saved in ./output_vanilla"
echo ""

# -------------------------------
# Variation 1: --select_Cspace yiq
# -------------------------------

echo "Running Variation 1: --select_Cspace yiq"
mkdir -p "./output_cluster_cspace_yiq"

# Run KMeans.py
$PYTHON_PATH $KMEANS_SCRIPT \
    --used_pt "$USED_PT_BASE" \
    --extract "$EXTRACT" \
    --select_Cspace "yiq" \
    --select_transform "None" \
    --select_compress_ratio 1 \
    --output_name "cluster_cspace_yiq.pt"

# Run photomosiac2.py
$PYTHON_PATH $PHOTOMOSAIC_SCRIPT \
    --tgt_pt "$TGT_PT" \
    --used_pt "cluster_cspace_yiq.pt" \
    --n_row "$N_ROW" \
    --n_col "$N_COL" \
    $EXHIBIT \
    --query_Cspace "$QUERY_CSPACE_VANILLA" \
    --query_transform "$QUERY_TRANSFORM_VANILLA" \
    --query_compress_ratio "$QUERY_COMPRESS_RATIO_VANILLA" \
    $QUERY_IS_FITNESS \
    --evaluate_Cspace "$EVALUATE_CSPACE" \
    $EVALUATE_IS_FITNESS \
    $SAVE_PLOTS \
    --output_dir "./output_cluster_cspace_yiq"

echo "Variation 1 completed. Results saved in ./output_cluster_cspace_yiq"
echo ""

# -------------------------------
# Variation 2: --select_Cspace xyz
# -------------------------------

echo "Running Variation 2: --select_Cspace xyz"
mkdir -p "./output_cluster_cspace_xyz"

# Run KMeans.py
$PYTHON_PATH $KMEANS_SCRIPT \
    --used_pt "$USED_PT_BASE" \
    --extract "$EXTRACT" \
    --select_Cspace "xyz" \
    --select_transform "None" \
    --select_compress_ratio 1 \
    --output_name "cluster_cspace_xyz.pt"

# Run photomosiac2.py
$PYTHON_PATH $PHOTOMOSAIC_SCRIPT \
    --tgt_pt "$TGT_PT" \
    --used_pt "cluster_cspace_xyz.pt" \
    --n_row "$N_ROW" \
    --n_col "$N_COL" \
    $EXHIBIT \
    --query_Cspace "$QUERY_CSPACE_VANILLA" \
    --query_transform "$QUERY_TRANSFORM_VANILLA" \
    --query_compress_ratio "$QUERY_COMPRESS_RATIO_VANILLA" \
    $QUERY_IS_FITNESS \
    --evaluate_Cspace "$EVALUATE_CSPACE" \
    $EVALUATE_IS_FITNESS \
    $SAVE_PLOTS \
    --output_dir "./output_cluster_cspace_xyz"

echo "Variation 2 completed. Results saved in ./output_cluster_cspace_xyz"
echo ""

# -------------------------------
# Variation 3: --select_transform fourier
# -------------------------------

echo "Running Variation 3: --select_transform fourier"
mkdir -p "./output_cluster_transform_fourier"

# Run KMeans.py
$PYTHON_PATH $KMEANS_SCRIPT \
    --used_pt "$USED_PT_BASE" \
    --extract "$EXTRACT" \
    --select_Cspace "rgb" \
    --select_transform "$QUERY_TRANSFORM_VANILLA" \
    --select_compress_ratio 1 \
    --output_name "cluster_transform_fourier.pt"

# Run photomosiac2.py
$PYTHON_PATH $PHOTOMOSAIC_SCRIPT \
    --tgt_pt "$TGT_PT" \
    --used_pt "cluster_transform_fourier.pt" \
    --n_row "$N_ROW" \
    --n_col "$N_COL" \
    $EXHIBIT \
    --query_Cspace "$QUERY_CSPACE_VANILLA" \
    --query_transform "fourier" \
    --query_compress_ratio "$QUERY_COMPRESS_RATIO_VANILLA" \
    $QUERY_IS_FITNESS \
    --evaluate_Cspace "$EVALUATE_CSPACE" \
    $EVALUATE_IS_FITNESS \
    $SAVE_PLOTS \
    --output_dir "./output_cluster_transform_fourier"

echo "Variation 3 completed. Results saved in ./output_cluster_transform_fourier"
echo ""

# -------------------------------
# Variation 4: --select_transform wavelet
# -------------------------------

echo "Running Variation 4: --select_transform wavelet"
mkdir -p "./output_cluster_transform_wavelet"

# Run KMeans.py
$PYTHON_PATH $KMEANS_SCRIPT \
    --used_pt "$USED_PT_BASE" \
    --extract "$EXTRACT" \
    --select_Cspace "rgb" \
    --select_transform "wavelet" \
    --select_compress_ratio 1 \
    --output_name "cluster_transform_wavelet.pt"

# Run photomosiac2.py
$PYTHON_PATH $PHOTOMOSAIC_SCRIPT \
    --tgt_pt "$TGT_PT" \
    --used_pt "cluster_transform_wavelet.pt" \
    --n_row "$N_ROW" \
    --n_col "$N_COL" \
    $EXHIBIT \
    --query_Cspace "$QUERY_CSPACE_VANILLA" \
    --query_transform "wavelet" \
    --query_compress_ratio "$QUERY_COMPRESS_RATIO_VANILLA" \
    $QUERY_IS_FITNESS \
    --evaluate_Cspace "$EVALUATE_CSPACE" \
    $EVALUATE_IS_FITNESS \
    $SAVE_PLOTS \
    --output_dir "./output_cluster_transform_wavelet"

echo "Variation 4 completed. Results saved in ./output_cluster_transform_wavelet"
echo ""

# -------------------------------
# Variations 5-9: --select_compress_ratio 2,4,8,16,32
# -------------------------------

COMPRESS_RATIOS=(2 4 8 16 32)

for ratio in "${COMPRESS_RATIOS[@]}"; do
    echo "Running Variation compress_ratio_$ratio"
    mkdir -p "./output_cluster_compress_${ratio}"
    
    # Run KMeans.py
    $PYTHON_PATH $KMEANS_SCRIPT \
        --used_pt "$USED_PT_BASE" \
        --extract "$EXTRACT" \
        --select_Cspace "rgb" \
        --select_transform "None" \
        --select_compress_ratio "$ratio" \
        --output_name "cluster_compress_${ratio}.pt"
    
    # Run photomosiac2.py
    $PYTHON_PATH $PHOTOMOSAIC_SCRIPT \
        --tgt_pt "$TGT_PT" \
        --used_pt "cluster_compress_${ratio}.pt" \
        --n_row "$N_ROW" \
        --n_col "$N_COL" \
        $EXHIBIT \
        --query_Cspace "$QUERY_CSPACE_VANILLA" \
        --query_transform "$QUERY_TRANSFORM_VANILLA" \
        --query_compress_ratio "$QUERY_COMPRESS_RATIO_VANILLA" \
        $QUERY_IS_FITNESS \
        --evaluate_Cspace "$EVALUATE_CSPACE" \
        $EVALUATE_IS_FITNESS \
        $SAVE_PLOTS \
        --output_dir "./output_cluster_compress_${ratio}"
    
    echo "Variation compress_ratio_$ratio completed. Results saved in ./output_cluster_compress_${ratio}"
    echo ""
done

# -------------------------------
# Variation 10: Omit --query_is_fitness
# -------------------------------

echo "Running Variation 10: Omit --query_is_fitness"
mkdir -p "./output_cluster_no_query_is_fitness"

# Run KMeans.py
$PYTHON_PATH $KMEANS_SCRIPT \
    --used_pt "$USED_PT_BASE" \
    --extract "$EXTRACT" \
    --select_Cspace "rgb" \
    --select_transform "None" \
    --select_compress_ratio 1 \
    --output_name "cluster_no_query_is_fitness.pt"

# Run photomosiac2.py without --query_is_fitness
$PYTHON_PATH $PHOTOMOSAIC_SCRIPT \
    --tgt_pt "$TGT_PT" \
    --used_pt "cluster_no_query_is_fitness.pt" \
    --n_row "$N_ROW" \
    --n_col "$N_COL" \
    $EXHIBIT \
    --query_Cspace "$QUERY_CSPACE_VANILLA" \
    --query_transform "$QUERY_TRANSFORM_VANILLA" \
    --query_compress_ratio "$QUERY_COMPRESS_RATIO_VANILLA" \
    --evaluate_Cspace "$EVALUATE_CSPACE" \
    $EVALUATE_IS_FITNESS \
    $SAVE_PLOTS \
    --output_dir "./output_cluster_no_query_is_fitness"

echo "Variation 10 completed. Results saved in ./output_cluster_no_query_is_fitness"
echo ""

echo "All variations have been executed successfully."

