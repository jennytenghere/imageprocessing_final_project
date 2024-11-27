#!/usr/bin/env python3

import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
import torch.nn.functional as F

def main():
    parser = argparse.ArgumentParser(description='Photomosaic Result Map')
    parser.add_argument('--tgt_pt', type=str, required=True, help='Path to A.pt (target images)')
    parser.add_argument('--used_pt', type=str, required=True, help='Path to B.pt (pool images)')
    parser.add_argument('--n_row', type=int, required=True, help='Number of rows (Y)')
    parser.add_argument('--n_col', type=int, required=True, help='Number of columns (X)')
    parser.add_argument('--method', type=str, default='rgb_fitness', help='Method name (default: rgb_fitness)')
    parser.add_argument('--metric', type=str, default='rgb_fitness', help='Metric name (default: rgb_fitness)')
    parser.add_argument('--exhibit', action='store_true', help='If provided, show comparison images')
    args = parser.parse_args()

    # Load tensors
    A = torch.load(args.tgt_pt).float()  # Shape: (N_A, H, W, C)
    B = torch.load(args.used_pt).float()  # Shape: (N_B, h, w, C)
    B=B[:1000]

    # Handle device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    A = A.to(device)
    B = B.to(device)

    N_A, H, W, C = A.shape
    N_B, h, w, C_B = B.shape
    n_row = args.n_row
    n_col = args.n_col

    # Ensure channels are the same
    assert C == C_B, "Channel dimensions of A.pt and B.pt must be the same."

    # Resize images in A to match n_row * h and n_col * w
    target_height = n_row * h
    target_width = n_col * w

    if H != target_height or W != target_width:
        print(f'Resizing images in A from ({H}, {W}) to ({target_height}, {target_width})')
        # A is (N_A, H, W, C), we need to permute to (N_A, C, H, W) for interpolation
        A = A.permute(0, 3, 1, 2)  # Now (N_A, C, H, W)
        A = F.interpolate(A, size=(target_height, target_width), mode='bilinear', align_corners=False)
        # Permute back to (N_A, target_height, target_width, C)
        A = A.permute(0, 2, 3, 1)
        H, W = target_height, target_width

    # Step 1: Reshaping
    # Reshape A into (N_A, n_row, h, n_col, w, C)
    A_reshaped = A.reshape(N_A, n_row, h, n_col, w, C)
    # Permute to (N_A, n_row, n_col, h, w, C)
    A_patches = A_reshaped.permute(0, 1, 3, 2, 4, 5)
    # Reshape to (N_patches, h, w, C)
    N_patches = N_A * n_row * n_col
    A_patches = A_patches.reshape(N_patches, h, w, C)

    # Step 2: Image querying with batching over A_patches
    batch_size_patches = 250  # Adjust based on your GPU memory capacity
    num_batches = (N_patches + batch_size_patches - 1) // batch_size_patches

    best_match_indices_list = []
    loss_tensor_list = []

    print(f'Total patches: {N_patches}, Processing in batches of size: {batch_size_patches}')

    for batch_idx in range(num_batches):
        print(f'Processing batch {batch_idx+1}/{num_batches}')
        start_idx = batch_idx * batch_size_patches
        end_idx = min((batch_idx + 1) * batch_size_patches, N_patches)
        A_batch = A_patches[start_idx:end_idx]  # Shape: (batch_size_patches, h, w, C)
        batch_size_actual = A_batch.shape[0]

        if args.method == 'rgb_fitness':
            # Expand dimensions for broadcasting
            A_batch_expanded = A_batch.unsqueeze(1)  # Shape: (batch_size_patches, 1, h, w, C)
            B_expanded = B.unsqueeze(0)  # Shape: (1, N_B, h, w, C)
            # Compute differences
            diff = torch.abs(A_batch_expanded - B_expanded)  # Shape: (batch_size_patches, N_B, h, w, C)
            # Sum over h, w, C to get distances
            distances = diff.sum(dim=(2,3,4))  # Shape: (batch_size_patches, N_B)
            # Find best matches
            best_indices = distances.argmin(dim=1)  # Shape: (batch_size_patches,)
        else:
            # Implement other methods as needed
            raise NotImplementedError(f'Method {args.method} is not implemented.')

        # Step 3: Evaluation metric
        if args.metric == 'rgb_fitness':
            # Use the distances computed during querying
            min_distances = distances.min(dim=1).values  # Shape: (batch_size_patches,)
            loss_tensor_batch = min_distances
        else:
            # Implement other metrics as needed
            raise NotImplementedError(f'Metric {args.metric} is not implemented.')

        best_match_indices_list.append(best_indices)
        loss_tensor_list.append(loss_tensor_batch)

    # Concatenate results
    best_match_indices = torch.cat(best_match_indices_list, dim=0)  # Shape: (N_patches,)
    loss_tensor = torch.cat(loss_tensor_list, dim=0)  # Shape: (N_patches,)

    # Get the founded images
    Founded_patches = B[best_match_indices]  # Shape: (N_patches, h, w, C)

    # Step 4: Reshape the founded patches back to images
    Founded_patches = Founded_patches.reshape(N_A, n_row, n_col, h, w, C)
    # Permute and reshape to (N_A, n_row * h, n_col * w, C)
    Founded_images = Founded_patches.permute(0, 1, 3, 2, 4, 5).reshape(N_A, n_row * h, n_col * w, C)

    # Reshape loss_tensor to (N_A, n_row, n_col)
    loss_tensor = loss_tensor.reshape(N_A, n_row, n_col)

    # Move tensors back to CPU for saving and displaying
    A = A.cpu().numpy().astype(np.uint8)
    Founded_images = Founded_images.cpu().numpy().astype(np.uint8)
    loss_tensor = loss_tensor.cpu().numpy()

    # Save the output images and loss_tensor
    output_dir = 'output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    np.save(os.path.join(output_dir, 'original_images.npy'), A)
    np.save(os.path.join(output_dir, f'constructed_images_{args.method}.npy'), Founded_images)
    np.save(os.path.join(output_dir, f'loss_tensor_{args.metric}.npy'), loss_tensor)

    print(f'Saved output images and loss_tensor to {output_dir}')

    # If exhibit is true, plot and show the comparison
    if args.exhibit:
        for i in range(N_A):
            fig, axs = plt.subplots(1, 3, figsize=(15, 5))
            axs[0].imshow(A[i])
            axs[0].set_title('Original Image')
            axs[0].axis('off')

            axs[1].imshow(Founded_images[i])
            axs[1].set_title(f'Constructed Image ({args.method})')
            axs[1].axis('off')

            # Normalize loss_tensor to [0,1] for grayscale display
            loss_image = loss_tensor[i]
            loss_image_norm = (loss_image - loss_image.min()) / (loss_image.max() - loss_image.min() + 1e-8)
            # Upscale the loss image to match image size
            loss_image_upscaled = np.kron(loss_image_norm, np.ones((h, w)))
            axs[2].imshow(loss_image_upscaled, cmap='gray')
            axs[2].set_title(f'Loss Map ({args.metric})')
            axs[2].axis('off')

            plt.show()

if __name__ == '__main__':
    main()
