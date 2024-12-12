import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
import torch.nn.functional as F
import time
import pywt
from PIL import Image

def rgb_to_xyz(tensor):
    # Convert RGB to XYZ color space
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
    # Convert RGB to YIQ color space
    tensor = tensor / 255.0
    M = torch.tensor([[0.299,     0.587,     0.114],
                      [0.595716, -0.274453, -0.321263],
                      [0.211456, -0.522591,  0.311135]], device=tensor.device)
    N, H, W, C = tensor.shape
    tensor = tensor.view(N, H * W, C)
    tensor = torch.matmul(tensor, M.T)
    tensor = tensor.view(N, H, W, C)
    return tensor
import torch

def rgb_to_lab(tensor):
    """
    Convert a batch of RGB images to the CIELAB (LAB) color space.

    Parameters:
    - tensor (torch.Tensor): Input tensor of shape (N, H, W, 3) with RGB values in [0, 255].

    Returns:
    - torch.Tensor: Output tensor of shape (N, H, W, 3) with LAB values.
    """
    # Ensure the input tensor is a float tensor
    tensor = tensor.clone().float()

    # Normalize RGB values to [0, 1]
    tensor = tensor / 255.0

    # Inverse gamma correction (sRGB to linear RGB)
    threshold = 0.04045
    linear_mask = tensor > threshold
    tensor_linear = torch.zeros_like(tensor)
    tensor_linear[linear_mask] = ((tensor[linear_mask] + 0.055) / 1.055) ** 2.4
    tensor_linear[~linear_mask] = tensor[~linear_mask] / 12.92

    # Define the RGB to XYZ conversion matrix (sRGB D65)
    M = torch.tensor([[0.4124564, 0.3575761, 0.1804375],
                      [0.2126729, 0.7151522, 0.0721750],
                      [0.0193339, 0.1191920, 0.9503041]], device=tensor.device)

    # Reshape tensor for matrix multiplication
    N, H, W, C = tensor_linear.shape
    tensor_linear = tensor_linear.view(N, H * W, C)

    # Convert linear RGB to XYZ
    tensor_xyz = torch.matmul(tensor_linear, M.T)
    tensor_xyz = tensor_xyz.view(N, H, W, C)

    # Define reference white point (D65)
    white_ref = torch.tensor([0.95047, 1.00000, 1.08883], device=tensor.device).view(1, 1, 1, 3)

    # Normalize XYZ by the white reference
    tensor_xyz_norm = tensor_xyz / white_ref

    # Define the f(t) function for LAB conversion
    epsilon = 0.008856  # (6/29)^3
    kappa = 903.3       # (29/3)^3

    # Apply the f(t) function
    mask = tensor_xyz_norm > epsilon
    tensor_f = torch.zeros_like(tensor_xyz_norm)
    tensor_f[mask] = tensor_xyz_norm[mask] ** (1/3)
    tensor_f[~mask] = (kappa * tensor_xyz_norm[~mask] + 16) / 116

    # Compute L*, a*, b*
    L = 116 * tensor_f[..., 1] - 16
    a = 500 * (tensor_f[..., 0] - tensor_f[..., 1])
    b = 200 * (tensor_f[..., 1] - tensor_f[..., 2])

    # Stack the LAB channels
    lab = torch.stack([L, a, b], dim=-1)

    return lab

def rgb_to_hsv(tensor):
    # Convert RGB to HSV color space (OpenCV-like)
    # RGB values are in [0,255]
    # The output HSV should have:
    # H in [0,179], S in [0,255], V in [0,255]
    tensor = tensor.clone()  # Avoid altering original
    tensor = tensor.float()
    # Normalize to [0,1]
    r = tensor[...,0] / 255.0
    g = tensor[...,1] / 255.0
    b = tensor[...,2] / 255.0

    max_val, _ = torch.max(torch.stack((r,g,b), dim=-1), dim=-1)
    min_val, _ = torch.min(torch.stack((r,g,b), dim=-1), dim=-1)
    diff = max_val - min_val

    # Hue calculation
    # Avoid division by zero by adding a small epsilon where needed
    epsilon = 1e-6
    hue = torch.zeros_like(max_val)
    mask = (diff > epsilon)
    # For pixels where diff > 0:
    # hue calculation as per standard formula:
    # For reference:
    # H = 60 * ( (G - B)/diff mod 6 ) if max_val == R
    # H = 60 * ( (B - R)/diff + 2 ) if max_val == G
    # H = 60 * ( (R - G)/diff + 4 ) if max_val == B
    # Then H normalized to [0,360), we then scale to [0,179]
    r_eq = (max_val == r)
    g_eq = (max_val == g)
    b_eq = (max_val == b)

    hue[r_eq & mask] = (60.0 * ((g[r_eq & mask] - b[r_eq & mask]) / (diff[r_eq & mask] + epsilon)) ) % 360
    hue[g_eq & mask] = (60.0 * ((b[g_eq & mask] - r[g_eq & mask]) / (diff[g_eq & mask] + epsilon)) + 120) % 360
    hue[b_eq & mask] = (60.0 * ((r[b_eq & mask] - g[b_eq & mask]) / (diff[b_eq & mask] + epsilon)) + 240) % 360

    # Scale hue to [0,179]
    hue = hue * (179.0/360.0)

    # Saturation
    sat = torch.zeros_like(max_val)
    sat[mask] = (diff[mask] / (max_val[mask] + epsilon)) * 255.0

    # Value
    val = max_val * 255.0

    hsv = torch.stack([hue, sat, val], dim=-1)
    return hsv

def apply_transform(tensor, transform):
    if transform is None or transform.lower() == 'none':
        return tensor
    elif transform.lower() == 'fourier':
        tensor = tensor.permute(0, 3, 1, 2)  # (N, C, H, W)
        tensor_fft = torch.fft.fft2(tensor)
        tensor_abs = torch.abs(tensor_fft)
        tensor_transformed = tensor_abs.permute(0, 2, 3, 1)  # (N, H, W, C)
        return tensor_transformed
    elif transform.lower() == 'wavelet':
        N, H, W, C = tensor.shape
        transformed_tensors = []
        for c in range(C):
            channel = tensor[:, :, :, c].cpu().numpy()  # Shape: (N, H, W)
            transformed_channels = []
            for i in range(N):
                coeffs2 = pywt.dwt2(channel[i], 'haar')
                LL, (LH, HL, HH) = coeffs2
                transformed_channels.append(LL.flatten())
            transformed_channel_tensor = torch.tensor(transformed_channels, device=tensor.device)
            transformed_tensors.append(transformed_channel_tensor)
        tensor_transformed = torch.stack(transformed_tensors, dim=1)  # (N, C, LL_H*LL_W)
        return tensor_transformed
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
        elif color_space.lower() == 'hsv':
            tensor_batch_cs = rgb_to_hsv(tensor_batch)
        elif color_space.lower() == 'lab':
            tensor_batch_cs = rgb_to_lab(tensor_batch)
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

    # Detach original A and B
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

        # Compute difference
        # If HSV, we must use the special formula
        if args.query_Cspace.lower() == 'hsv':
            # Extract channels
            A_h = A_batch_expanded[...,0]
            A_s = A_batch_expanded[...,1]
            A_v = A_batch_expanded[...,2]

            B_h = B_query_expanded[...,0]
            B_s = B_query_expanded[...,1]
            B_v = B_query_expanded[...,2]

            # Hue difference with circular adjustment
            diff_h = torch.abs(A_h - B_h)
            max_hue = 180.0
            half_hue = max_hue / 2.0
            mask = diff_h > half_hue
            diff_h[mask] = max_hue - diff_h[mask]

            # Saturation and Value differences
            diff_s = A_s - B_s
            diff_v = A_v - B_v

            # Compute per-pixel distance in HSV
            # weights (1,1,1) are already default
            # Distances per pixel: sqrt(diff_h^2 + diff_s^2 + diff_v^2)
            # Then sum over (H, W, C)
            pixel_dist = torch.sqrt(diff_h**2 + diff_s**2 + diff_v**2)
            # If query_is_fitness: sum of absolute differences is requested
            # But we have a Euclidean-like metric here.
            # The user request: "If query method is hsv, fix the distance calculating to the above"
            # The above definition uses Euclidean distance.
            # We'll ignore query_is_fitness in HSV mode as per instructions ("fix the distance ... always weight 1,1,1").
            distances = pixel_dist.sum(dim=(2,3))

        else:
            # Original functionality for other color spaces
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

    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

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
        # im = Image.fromarray(Founded_images[i]).resize((4000,2250))
        # im.save("your_file.jpeg")
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        axs[0].imshow(A[i])
        axs[0].set_title('Original Image')
        axs[0].axis('off')

        axs[1].imshow(Founded_images[i])
        axs[1].set_title('Constructed Image')
        axs[1].axis('off')

        loss_image = loss_tensor_reshaped[i]/(h_evaluate* w_evaluate* C_B_evaluate)
        loss_image_upscaled = np.kron(loss_image, np.ones((h, w)))
        im = axs[2].imshow(loss_image_upscaled, cmap='viridis')
        axs[2].set_title('Loss Map')
        axs[2].axis('off')

        cbar = fig.colorbar(im, ax=axs[2], orientation='vertical', fraction=0.046, pad=0.04)
        cbar.set_label('Loss Value')

        plot_filename = os.path.join(output_dir, f'comparison_{i}.png')
        plt.savefig(plot_filename, dpi=300)
        print(f'Saved plot to {plot_filename}')
        #plt.show()

    exhibit_end_time = time.time()
    print(f'Exhibit completed in {exhibit_end_time - exhibit_start_time:.2f} seconds.')

    total_end_time = time.time()
    print(f'Total time: {total_end_time - total_start_time:.2f} seconds.')

if __name__ == '__main__':
    main()
