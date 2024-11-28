#!/usr/bin/env python3

import torch
import argparse
import os
import numpy as np
import cudf
from cuml import KMeans
import torch.nn.functional as F
import pywt  # For wavelet transform
import time

def rgb_to_xyz(tensor):
    tensor = tensor / 255.0
    M = torch.tensor([[0.412453, 0.357580, 0.180423],
                      [0.212671, 0.715160, 0.072169],
                      [0.019334, 0.119193, 0.950227]], device=tensor.device)
    N, H, W, C = tensor.shape
    tensor = tensor.view(N, H * W, C)
    tensor = torch.matmul(tensor, M.T)
    tensor = tensor.view(N, H, W, C)
    return tensor

def rgb_to_yiq(tensor):
    tensor = tensor / 255.0
    M = torch.tensor([[0.299, 0.587, 0.114],
                      [0.595716, -0.274453, -0.321263],
                      [0.211456, -0.522591, 0.311135]], device=tensor.device)
    N, H, W, C = tensor.shape
    tensor = tensor.view(N, H * W, C)
    tensor = torch.matmul(tensor, M.T)
    tensor = tensor.view(N, H, W, C)
    return tensor

def apply_transform(tensor, transform):
    if transform is None or transform.lower() == 'none':
        return tensor
    elif transform.lower() == 'wavelet':
        N, H, W, C = tensor.shape
        transformed_tensors = []
        for c in range(C):
            channel = tensor[:, :, :, c]
            coeffs2 = pywt.dwt2(channel.cpu().numpy(), 'haar')
            LL, (LH, HL, HH) = coeffs2
            transformed_tensors.append(torch.from_numpy(LL))
        tensor_transformed = torch.stack(transformed_tensors, dim=3)
        return tensor_transformed.to(tensor.device)
    else:
        raise NotImplementedError(f"Transform '{transform}' is not implemented.")

def compress_tensor(tensor, compress_ratio):
    if compress_ratio <= 1:
        return tensor
    else:
        N, H, W, C = tensor.shape
        new_H = H // compress_ratio
        new_W = W // compress_ratio
        tensor = tensor.permute(0, 3, 1, 2)
        tensor = F.interpolate(tensor, size=(new_H, new_W), mode='area')
        tensor = tensor.permute(0, 2, 3, 1)
        return tensor

def preprocess_tensor(tensor, color_space, transform, compress_ratio, batch_size):
    N = tensor.shape[0]
    processed_tensors = []
    num_batches = (N + batch_size - 1) // batch_size
    for batch_idx in range(num_batches):
        print(f"Preprocessing batch {batch_idx + 1}/{num_batches}")
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, N)
        tensor_batch = tensor[start_idx:end_idx]
        if color_space.lower() == 'rgb':
            tensor_batch_cs = tensor_batch
        elif color_space.lower() == 'xyz':
            tensor_batch_cs = rgb_to_xyz(tensor_batch)
        elif color_space.lower() == 'yiq':
            tensor_batch_cs = rgb_to_yiq(tensor_batch)
        else:
            raise ValueError(f"Unknown color space: {color_space}")
        tensor_batch_transformed = apply_transform(tensor_batch_cs, transform)
        tensor_batch_compressed = compress_tensor(tensor_batch_transformed, compress_ratio)
        processed_tensors.append(tensor_batch_compressed)
    processed_tensor = torch.cat(processed_tensors, dim=0)
    return processed_tensor

def main():
    parser = argparse.ArgumentParser(description='KMeans Image Selection')
    parser.add_argument('--used_pt', type=str, required=True, help='Path to B.pt (pool images)')
    parser.add_argument('--extract', type=int, required=True, help='Number of images to extract')
    parser.add_argument('--select_Cspace', type=str, default='rgb', help='Color space used in selection (default: rgb)')
    parser.add_argument('--select_transform', type=str, default=None, help='Transformation used in selection (options: None, wavelet)')
    parser.add_argument('--select_compress_ratio', type=int, default=1, help='Compression ratio for selection (default: 1)')

    args = parser.parse_args()

    # Load B.pt
    print("Loading B.pt...")
    B = torch.load(args.used_pt).float()

    N_B, h, w, C = B.shape

    # Preprocess B
    print("Preprocessing B images...")
    batch_size = 1000
    B_processed = preprocess_tensor(B, args.select_Cspace, args.select_transform, args.select_compress_ratio, batch_size)

    # Flatten images for clustering
    print("Flattening images for clustering...")
    N_B_processed, H_processed, W_processed, C_processed = B_processed.shape
    B_flattened = B_processed.reshape(N_B_processed, -1).cpu().numpy().astype(np.float32)

    # Perform KMeans clustering using cuML
    cluster_start_time=time.time()
    print(f"Performing KMeans clustering with k={args.extract}...")
    kmeans_model = KMeans(n_clusters=args.extract, init='k-means++', random_state=42)

    kmeans_model.fit(B_flattened)

    # Get cluster centers and labels
    labels = kmeans_model.labels_
    cluster_centers = kmeans_model.cluster_centers_

    # Find the closest image to each cluster center
    print("Selecting images closest to cluster centers...")
    selected_indices = []
    for cluster_id in range(args.extract):
        cluster_mask = (labels == cluster_id)
        cluster_data = B_flattened[cluster_mask]
        if cluster_data.shape[0] == 0:
            print(f"Warning: Cluster {cluster_id} has no members.")
            continue
        center = cluster_centers[cluster_id]
        distances = np.linalg.norm(cluster_data - center, axis=1)
        min_idx = np.argmin(distances)
        original_idx = np.where(cluster_mask)[0][min_idx]
        selected_indices.append(original_idx)

    if len(selected_indices) < args.extract:
        print(f"Only {len(selected_indices)} images were selected out of {args.extract} requested.")

    # Select the images from the original B tensor
    B_selected = B[selected_indices]

    # Prepare output filename
    args_dict = vars(args)
    args_str = '_'.join([f"{k}={args_dict[k]}" for k in args_dict if k != 'used_pt'])
    output_filename = f"KMeans_{args_str}.pt"
    cluster_end_time = time.time()
    print("Takes", cluster_end_time - cluster_start_time)
    # Save the selected images
    torch.save(B_selected, output_filename)
    print(f"Saved selected images to {output_filename}")

if __name__ == '__main__':
    main()
