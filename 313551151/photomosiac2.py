#!/usr/bin/env python3

import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
import torch.nn.functional as F
import time


def rgb_to_xyz(tensor):
    # Convert RGB to XYZ color space
    # Normalize RGB values to [0, 1]
    tensor = tensor / 255.0
    # sRGB to XYZ conversion matrix
    M = torch.tensor([[0.412453, 0.357580, 0.180423],
                      [0.212671, 0.715160, 0.072169],
                      [0.019334, 0.119193, 0.950227]], device=tensor.device)
    # Reshape tensor for matrix multiplication
    N, H, W, C = tensor.shape
    tensor = tensor.view(N, H * W, C)
    tensor = torch.matmul(tensor, M.T)
    tensor = tensor.view(N, H, W, C)
    return tensor


def rgb_to_yiq(tensor):
    # Convert RGB to YIQ color space
    # Normalize RGB values to [0, 1]
    tensor = tensor / 255.0
    # RGB to YIQ conversion matrix
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
    elif transform.lower() == 'fourier':
        # Apply Fourier transform
        tensor = tensor.permute(0, 3, 1, 2)  # (N, C, H, W)
        tensor_fft = torch.fft.fft2(tensor)
        tensor = torch.abs(tensor_fft)
        tensor = tensor.permute(0, 2, 3, 1)  # (N, H, W, C)
        return tensor
    else:
        raise NotImplementedError(f"Transform '{transform}' is not implemented.")


def compress_tensor(tensor, compress_ratio):
    if compress_ratio <= 1:
        return tensor
    else:
        N, H, W, C = tensor.shape
        new_H = H // compress_ratio
        new_W = W // compress_ratio
        tensor = tensor.permute(0, 3, 1, 2)  # (N, C, H, W)
        tensor = F.interpolate(tensor, size=(new_H, new_W), mode='area')
        tensor = tensor.permute(0, 2, 3, 1)  # (N, new_H, new_W, C)
        return tensor


def preprocess_tensor(tensor, color_space, transform, compress_ratio, batch_size):
    N = tensor.shape[0]
    processed_tensors = []
    num_batches = (N + batch_size - 1) // batch_size
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, N)
        tensor_batch = tensor[start_idx:end_idx]
        # Color space conversion
        if color_space.lower() == 'rgb':
            tensor_batch_cs = tensor_batch
        elif color_space.lower() == 'xyz':
            tensor_batch_cs = rgb_to_xyz(tensor_batch)
        elif color_space.lower() == 'yiq':
            tensor_batch_cs = rgb_to_yiq(tensor_batch)
        else:
            raise ValueError(f"Unknown color space: {color_space}")
        # Apply transformation
        tensor_batch_transformed = apply_transform(tensor_batch_cs, transform)
        # Compression
        tensor_batch_compressed = compress_tensor(tensor_batch_transformed, compress_ratio)
        processed_tensors.append(tensor_batch_compressed)
    processed_tensor = torch.cat(processed_tensors, dim=0)
    return processed_tensor


def main():
    parser = argparse.ArgumentParser(description='Photomosaic Result Map')

    # Input arguments
    parser.add_argument('--tgt_pt', type=str, required=True, help='Path to A.pt (target images)')
    parser.add_argument('--used_pt', type=str, required=True, help='Path to B.pt (pool images)')
    parser.add_argument('--n_row', type=int, required=True, help='Number of rows (Y)')
    parser.add_argument('--n_col', type=int, required=True, help='Number of columns (X)')
    parser.add_argument('--exhibit', action='store_true', help='If provided, show comparison images')
    parser.add_argument('--B_limit', type=int, default=1000, help='Take the first n pictures out from B.pt,-1 means all')

    # Batch sizes
    parser.add_argument('--pre_batch', type=int, default=1000, help='Preprocessing stage batch size')
    parser.add_argument('--com_batch', type=int, default=250, help='Compare and evaluating stage batch size')
    parser.add_argument('--rend_batch', type=int, default=1000, help='Rendering stage batch size')

    # Querying parameters
    parser.add_argument('--query_Cspace', type=str, default='rgb', help='Color space used in querying (default: rgb)')
    parser.add_argument('--query_transform', type=str, default=None,
                        help='Transformation used in querying (options: None, fourier)')
    parser.add_argument('--query_compress_ratio', type=int, default=1,
                        help='Compression ratio for querying (default: 1)')
    parser.add_argument('--query_is_fitness', action='store_true', help='Use sum of absolute differences in querying')

    # Evaluating parameters
    parser.add_argument('--evaluate_Cspace', type=str, default='rgb',
                        help='Color space used in evaluating (default: rgb)')
    parser.add_argument('--evaluate_transform', type=str, default=None,
                        help='Transformation used in evaluating (options: None, fourier)')
    parser.add_argument('--evaluate_compress_ratio', type=int, default=1,
                        help='Compression ratio for evaluating (default: 1)')
    parser.add_argument('--evaluate_is_fitness', action='store_true',
                        help='Use sum of absolute differences in evaluating')
    parser.add_argument('--save_plots', action='store_true',
                        help='If provided, save the plots as high-resolution images')

    parser.add_argument('--output_dir', type=str, default='output')
    args = parser.parse_args()

    # Start timing
    total_start_time = time.time()


    # Load tensors
    A = torch.load(args.tgt_pt).float()
    B = torch.load(args.used_pt).float()
    if(args.B_limit != -1):B = B[:args.B_limit]

    # Handle device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    A = A.to(device)
    B = B.to(device)

    N_A, H, W, C = A.shape
    N_B, h, w, C_B = B.shape
    n_row = args.n_row
    n_col = args.n_col

    assert C == C_B, "Channel dimensions of A.pt and B.pt must be the same."

    # Resize images in A to match n_row * h and n_col * w
    target_height = n_row * h
    target_width = n_col * w

    if H != target_height or W != target_width:
        print(f'Resizing images in A from ({H}, {W}) to ({target_height}, {target_width})')
        A = A.permute(0, 3, 1, 2)
        A = F.interpolate(A, size=(target_height, target_width), mode='bilinear', align_corners=False)
        A = A.permute(0, 2, 3, 1)
        H, W = target_height, target_width

    # Preprocessing stage
    preprocessing_start_time = time.time()
    print('Starting preprocessing...')

    A_query = preprocess_tensor(A, args.query_Cspace, args.query_transform, args.query_compress_ratio, args.pre_batch)
    B_query = preprocess_tensor(B, args.query_Cspace, args.query_transform, args.query_compress_ratio, args.pre_batch)

    A_evaluate = preprocess_tensor(A, args.evaluate_Cspace, args.evaluate_transform, args.evaluate_compress_ratio,
                                   args.pre_batch)
    B_evaluate = preprocess_tensor(B, args.evaluate_Cspace, args.evaluate_transform, args.evaluate_compress_ratio,
                                   args.pre_batch)

    # Detach unnecessary tensors
    A.detach()
    B.detach()
    torch.cuda.empty_cache()

    preprocessing_end_time = time.time()
    preprocessing_time = preprocessing_end_time - preprocessing_start_time
    print(f'Preprocessing completed in {preprocessing_time:.2f} seconds.')

    # Querying stage
    querying_start_time = time.time()
    print('Starting querying...')

    # Reshape A_query into patches
    N_A_query, H_query, W_query, C_query = A_query.shape
    N_B_query, h_query, w_query, C_B_query = B_query.shape

    N_A_evaluate, H_evaluate, W_evaluate, C_evaluate = A_evaluate.shape
    N_B_evaluate, h_evaluate, w_evaluate, C_B_evaluate = B_evaluate.shape

    A_query_reshaped = A_query.reshape(N_A_query, n_row, h_query, n_col, w_query, C_query)
    A_query_patches = A_query_reshaped.permute(0, 1, 3, 2, 4, 5).reshape(-1, h_query, w_query, C_query)
    N_patches = A_query_patches.shape[0]

    batch_size = args.com_batch
    num_batches = (N_patches + batch_size - 1) // batch_size

    best_match_indices_list = []

    for batch_idx in range(num_batches):
        if (batch_idx % 10 == 0):
            print(f'Querying batch {batch_idx + 1}/{num_batches}')
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, N_patches)
        A_batch = A_query_patches[start_idx:end_idx]

        A_batch_expanded = A_batch.unsqueeze(1)
        B_query_expanded = B_query.unsqueeze(0)

        diff = A_batch_expanded - B_query_expanded

        if args.query_is_fitness:
            distances = torch.abs(diff).sum(dim=(2, 3, 4))
        else:
            distances = (diff ** 2).sum(dim=(2, 3, 4))

        best_indices = distances.argmin(dim=1)
        best_match_indices_list.append(best_indices)

    best_match_indices = torch.cat(best_match_indices_list, dim=0)

    querying_end_time = time.time()
    querying_time = querying_end_time - querying_start_time
    print(f'Querying completed in {querying_time:.2f} seconds.')


    # Evaluating stage
    evaluating_start_time = time.time()
    print('Starting evaluating...')

    A_evaluate_reshaped = A_evaluate.reshape(N_A_evaluate, n_row, h_evaluate, n_col, w_evaluate, C_evaluate)
    A_evaluate_patches = A_evaluate_reshaped.permute(0, 1, 3, 2, 4, 5).reshape(-1, h_evaluate, w_evaluate, C_evaluate)
    B_evaluate_patches = B_evaluate[best_match_indices]

    loss_tensor_list = []
    num_batches = (N_patches + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        if (batch_idx % 10 == 0):
            print(f'Evaluating batch {batch_idx + 1}/{num_batches}')
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, N_patches)
        A_batch = A_evaluate_patches[start_idx:end_idx]
        B_batch = B_evaluate_patches[start_idx:end_idx]

        diff = A_batch - B_batch

        if args.evaluate_is_fitness:
            losses = torch.abs(diff).sum(dim=(1, 2, 3))
        else:
            losses = (diff ** 2).sum(dim=(1, 2, 3))

        loss_tensor_list.append(losses)

    loss_tensor = torch.cat(loss_tensor_list, dim=0)

    evaluating_end_time = time.time()
    evaluating_time = evaluating_end_time - evaluating_start_time
    print(f'Evaluating completed in {evaluating_time:.2f} seconds.')

    # Compute and print mean loss per pixel
    total_loss = loss_tensor.sum().item()
    total_pixels = N_patches * h_evaluate * w_evaluate * C_evaluate
    mean_loss_per_pixel = total_loss / total_pixels
    print(f"Mean loss per pixel: {mean_loss_per_pixel:.6f}")

    # Prepare output log
    result_log_path = os.path.join(args.output_dir, 'result.log')
    with open(result_log_path, 'w') as log_file:
        log_file.write(f"Preprocessing time: {preprocessing_time:.2f} seconds\n")
        log_file.write(f"Querying time: {querying_time:.2f} seconds\n")
        log_file.write(f"Evaluating time: {evaluating_time:.2f} seconds\n")
        log_file.write(f"Mean loss per pixel: {mean_loss_per_pixel:.6f}\n")

    # Reshape and save tensors
    best_match_indices_reshaped = best_match_indices.reshape(N_A_query, n_row, n_col)
    loss_tensor_reshaped = loss_tensor.reshape(N_A_query, n_row, n_col)

    output_dir = args.output_dir
    assert os.path.exists(output_dir)

    output_log_path = os.path.join(args.output_dir, 'output.log')
    with open(output_log_path, 'w') as log_file:
        # Write all parameters
        log_file.write("Parameters:\n")
        for arg in vars(args):
            log_file.write(f"{arg}: {getattr(args, arg)}\n")
        log_file.write("\n")

    torch.save(best_match_indices_reshaped.cpu(), os.path.join(output_dir, 'best_match_indices.pt'))
    torch.save(loss_tensor_reshaped.cpu(), os.path.join(output_dir, 'loss_tensor.pt'))

    print(f'Saved best_match_indices.pt and loss_tensor.pt to {output_dir}')

    # Exhibit stage
    if not args.exhibit:
        total_end_time = time.time()
        print(f'Total time: {total_end_time - total_start_time:.2f} seconds.')
        return

    exhibit_start_time = time.time()
    print('Starting exhibit...')

    # Reload original A and B
    A = A.to(device)
    B = B.to(device)

    if H != target_height or W != target_width:
        A = A.permute(0, 3, 1, 2)
        A = F.interpolate(A, size=(target_height, target_width), mode='bilinear', align_corners=False)
        A = A.permute(0, 2, 3, 1)

    # Reshape A
    A_reshaped = A.reshape(N_A_query, n_row, h, n_col, w, C)
    A_patches = A_reshaped.permute(0, 1, 3, 2, 4, 5).reshape(-1, h, w, C)

    # Get matched patches
    B_matched_patches = B[best_match_indices]
    Founded_patches = B_matched_patches.reshape(N_A_query, n_row, n_col, h, w, C)
    Founded_images = Founded_patches.permute(0, 1, 3, 2, 4, 5).reshape(N_A_query, n_row * h, n_col * w, C)

    # Move to CPU
    A = A.cpu().numpy().astype(np.uint8)
    Founded_images = Founded_images.cpu().numpy().astype(np.uint8)
    loss_tensor_reshaped = loss_tensor_reshaped.cpu().numpy()

    # Plotting
    for i in range(N_A_query):
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        axs[0].imshow(A[i])
        axs[0].set_title('Original Image')
        axs[0].axis('off')

        axs[1].imshow(Founded_images[i])
        axs[1].set_title('Constructed Image')
        axs[1].axis('off')

        loss_image = loss_tensor_reshaped[i]/(h_evaluate* w_evaluate* C_B_evaluate)
        # Upscale the loss image to match image size
        loss_image_upscaled = np.kron(loss_image, np.ones((h, w)))
        # Use a colorful colormap and store the image object
        im = axs[2].imshow(loss_image_upscaled, cmap='viridis')
        axs[2].set_title('Loss Map')
        axs[2].axis('off')

        # Add a colorbar to the loss map with original loss values
        cbar = fig.colorbar(im, ax=axs[2], orientation='vertical', fraction=0.046, pad=0.04)
        cbar.set_label('Loss Value')

        plot_filename = os.path.join(output_dir, f'comparison_{i}.png')
        plt.savefig(plot_filename, dpi=300)
        print(f'Saved plot to {plot_filename}')
        plt.show()

    exhibit_end_time = time.time()
    print(f'Exhibit completed in {exhibit_end_time - exhibit_start_time:.2f} seconds.')

    total_end_time = time.time()
    print(f'Total time: {total_end_time - total_start_time:.2f} seconds.')


if __name__ == '__main__':
    main()
