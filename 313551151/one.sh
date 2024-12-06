#!/bin/bash

# Script: run_photomosaic_variations.sh
# Description: Executes photomosiac2.py with various parameter combinations.
# Usage: ./run_photomosaic_variations.sh
# Make sure to give execute permissions: chmod +x run_photomosaic_variations.sh

# Base command components
PYTHON_PATH="/home/ccwu/miniforge3/bin/python"
SCRIPT_PATH="/home/ccwu/PycharmProjects/DLP_pythonProject/DIP_final_project/photomosiac2.py"
TGT_PT="data_9_1024_1024.pt"
USED_PT="cifar10_data_50000_32_32.pt"
N_ROW=50
N_COL=50
EXHIBIT="--exhibit"
EVALUATE_CSPACE="rgb"
EVALUATE_IS_FITNESS="--evaluate_is_fitness"
SAVE_PLOTS="--save_plots"

# Vanilla Command
OUTPUT_DIR_VANILLA="./output_vanilla"
mkdir -p "$OUTPUT_DIR_VANILLA"
$PYTHON_PATH $SCRIPT_PATH \
  --tgt_pt $TGT_PT \
  --used_pt $USED_PT \
  --n_row $N_ROW \
  --n_col $N_COL \
  $EXHIBIT \
  --query_Cspace rgb \
  --query_transform none \
  --query_compress_ratio 1 \
  --query_is_fitness \
  --evaluate_Cspace $EVALUATE_CSPACE \
  $EVALUATE_IS_FITNESS \
  $SAVE_PLOTS \
  --output_dir "$OUTPUT_DIR_VANILLA"

# Variation 1: --query_Cspace yiq
OUTPUT_DIR_QUERY_CSPACE_YIQ="./output_query_Cspace_yiq"
mkdir -p "$OUTPUT_DIR_QUERY_CSPACE_YIQ"
$PYTHON_PATH $SCRIPT_PATH \
  --tgt_pt $TGT_PT \
  --used_pt $USED_PT \
  --n_row $N_ROW \
  --n_col $N_COL \
  $EXHIBIT \
  --query_Cspace yiq \
  --query_transform none \
  --query_compress_ratio 1 \
  --query_is_fitness \
  --evaluate_Cspace $EVALUATE_CSPACE \
  $EVALUATE_IS_FITNESS \
  $SAVE_PLOTS \
  --output_dir "$OUTPUT_DIR_QUERY_CSPACE_YIQ"

# Variation 2: --query_Cspace xyz
OUTPUT_DIR_QUERY_CSPACE_XYZ="./output_query_Cspace_xyz"
mkdir -p "$OUTPUT_DIR_QUERY_CSPACE_XYZ"
$PYTHON_PATH $SCRIPT_PATH \
  --tgt_pt $TGT_PT \
  --used_pt $USED_PT \
  --n_row $N_ROW \
  --n_col $N_COL \
  $EXHIBIT \
  --query_Cspace xyz \
  --query_transform none \
  --query_compress_ratio 1 \
  --query_is_fitness \
  --evaluate_Cspace $EVALUATE_CSPACE \
  $EVALUATE_IS_FITNESS \
  $SAVE_PLOTS \
  --output_dir "$OUTPUT_DIR_QUERY_CSPACE_XYZ"

# Variation 3: --query_transform fourier
OUTPUT_DIR_QUERY_TRANSFORM_FOURIER="./output_query_transform_fourier"
mkdir -p "$OUTPUT_DIR_QUERY_TRANSFORM_FOURIER"
$PYTHON_PATH $SCRIPT_PATH \
  --tgt_pt $TGT_PT \
  --used_pt $USED_PT \
  --n_row $N_ROW \
  --n_col $N_COL \
  $EXHIBIT \
  --query_Cspace rgb \
  --query_transform fourier \
  --query_compress_ratio 1 \
  --query_is_fitness \
  --evaluate_Cspace $EVALUATE_CSPACE \
  $EVALUATE_IS_FITNESS \
  $SAVE_PLOTS \
  --output_dir "$OUTPUT_DIR_QUERY_TRANSFORM_FOURIER"

# Variation 4: --query_transform wavelet
OUTPUT_DIR_QUERY_TRANSFORM_WAVELET="./output_query_transform_wavelet"
mkdir -p "$OUTPUT_DIR_QUERY_TRANSFORM_WAVELET"
$PYTHON_PATH $SCRIPT_PATH \
  --tgt_pt $TGT_PT \
  --used_pt $USED_PT \
  --n_row $N_ROW \
  --n_col $N_COL \
  $EXHIBIT \
  --query_Cspace rgb \
  --query_transform wavelet \
  --query_compress_ratio 1 \
  --query_is_fitness \
  --evaluate_Cspace $EVALUATE_CSPACE \
  $EVALUATE_IS_FITNESS \
  $SAVE_PLOTS \
  --output_dir "$OUTPUT_DIR_QUERY_TRANSFORM_WAVELET"

# Variations 5-9: --query_compress_ratio 2,4,8,16,32
COMPRESS_RATIOS=(2 4 8 16 32)
for ratio in "${COMPRESS_RATIOS[@]}"; do
  OUTPUT_DIR_COMPRESS="./output_query_compress_ratio_${ratio}"
  mkdir -p "$OUTPUT_DIR_COMPRESS"
  $PYTHON_PATH $SCRIPT_PATH \
    --tgt_pt $TGT_PT \
    --used_pt $USED_PT \
    --n_row $N_ROW \
    --n_col $N_COL \
    $EXHIBIT \
    --query_Cspace rgb \
    --query_transform none \
    --query_compress_ratio $ratio \
    --query_is_fitness \
    --evaluate_Cspace $EVALUATE_CSPACE \
    $EVALUATE_IS_FITNESS \
    $SAVE_PLOTS \
    --output_dir "$OUTPUT_DIR_COMPRESS"
done

# Variation 10: Omit --query_is_fitness
OUTPUT_DIR_NO_QUERY_IS_FITNESS="./output_no_query_is_fitness"
mkdir -p "$OUTPUT_DIR_NO_QUERY_IS_FITNESS"
$PYTHON_PATH $SCRIPT_PATH \
  --tgt_pt $TGT_PT \
  --used_pt $USED_PT \
  --n_row $N_ROW \
  --n_col $N_COL \
  $EXHIBIT \
  --query_Cspace rgb \
  --query_transform none \
  --query_compress_ratio 1 \
  --evaluate_Cspace $EVALUATE_CSPACE \
  $EVALUATE_IS_FITNESS \
  $SAVE_PLOTS \
  --output_dir "$OUTPUT_DIR_NO_QUERY_IS_FITNESS"
