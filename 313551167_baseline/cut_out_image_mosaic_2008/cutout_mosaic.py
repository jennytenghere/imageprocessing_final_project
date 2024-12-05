import numpy as np
from PIL import Image
import cv2 
from scipy.fft import fft2, ifft2
from scipy.optimize import linear_sum_assignment
import time
from tqdm import tqdm
import os
import multiprocessing
from multiprocessing import Pool
from functools import partial

class CutOutMosaic:
    def __init__(self, tile_size=50):
        self.tile_size = tile_size
        self.tiles = []
        self.tile_ffts = []
        self.num_processes = max(1, multiprocessing.cpu_count() - 1)

    def load_tiles(self, tiles_dir):
        """
        Load and preprocess tile images with better error handling
        """
        print("Loading tiles from:", tiles_dir)
        files = os.listdir(tiles_dir)
        
        with tqdm(total=len(files), desc="Processing tiles") as pbar:
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    try:
                        img_path = os.path.join(tiles_dir, file)
                        img = cv2.imread(img_path)
                        if img is None:
                            continue
                            
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        # Convert to YIQ color space
                        yiq = self.rgb_to_yiq(img)
                        yiq_resized = cv2.resize(yiq, (self.tile_size, self.tile_size))
                        
                        self.tiles.append(yiq_resized)
                        
                        # Compute FFT for each channel
                        ffts = []
                        for c in range(3):
                            fft = fft2(yiq_resized[:,:,c])
                            ffts.append(fft)
                        self.tile_ffts.append(ffts)
                        
                    except Exception as e:
                        print(f"Error processing {file}: {e}")
                    pbar.update(1)
                    
        print(f"Processed {len(self.tiles)} tiles")

    def rgb_to_yiq(self, rgb):
        """
        Convert RGB to YIQ color space using the correct transformation matrix
        from the paper
        """
        # Normalize RGB values to [0,1] range
        rgb_norm = rgb.astype(np.float32) / 255.0
        
        # Use the exact transformation matrix from the paper
        transform = np.array([
            [0.299, 0.587, 0.114],     # Y
            [0.596, -0.274, -0.322],   # I 
            [0.211, -0.523, 0.312]     # Q
        ])
        
        yiq = np.zeros_like(rgb_norm)
        for i in range(3):
            yiq[:,:,i] = (transform[i,0] * rgb_norm[:,:,0] + 
                         transform[i,1] * rgb_norm[:,:,1] + 
                         transform[i,2] * rgb_norm[:,:,2])
        return yiq

    def yiq_to_rgb(self, yiq):
        """
        Convert YIQ back to RGB using the correct inverse transformation
        """
        # Use the exact inverse transformation matrix from the paper
        transform = np.array([
            [1.000, 0.956, 0.621],
            [1.000, -0.272, -0.647], 
            [1.000, -1.105, 1.702]
        ])
        
        rgb = np.zeros_like(yiq)
        for i in range(3):
            rgb[:,:,i] = (transform[i,0] * yiq[:,:,0] + 
                         transform[i,1] * yiq[:,:,1] + 
                         transform[i,2] * yiq[:,:,2])
                         
        # Scale back to [0,255] range and clip
        rgb = rgb * 255.0
        return np.clip(rgb, 0, 255).astype(np.uint8)

    def compute_ssd_fft(self, target_tile, source_fft):
        """
        Compute Sum of Squared Differences using FFT
        """
        target_fft = [fft2(target_tile[:,:,i]) for i in range(3)]
        ssd = 0
        for i in range(3):
            diff = target_fft[i] - source_fft[i] 
            ssd += np.sum(np.abs(diff) ** 2)
        return ssd.real

    def compute_optimal_color_correction(self, target_tile, source_tile):
        """
        Compute optimal color correction parameters as described in the paper
        Returns separate scaling factors for luminance and chrominance
        """
        # Compute luminance (Y) scale factor
        y_scale = np.sum(target_tile[:,:,0] * source_tile[:,:,0]) / (np.sum(source_tile[:,:,0] ** 2) + 1e-6)
        
        # Compute chrominance (I,Q) scale factor
        c_scale = np.sum(target_tile[:,:,1:] * source_tile[:,:,1:]) / (np.sum(source_tile[:,:,1:] ** 2) + 1e-6)
        
        return np.array([y_scale, c_scale, c_scale])

    def process_tile_position(self, args):
        """
        Process a single tile position with improved color correction
        """
        i, j, target_tile = args
        costs = []
        
        for k, source_fft in enumerate(self.tile_ffts):
            # Compute basic SSD cost
            cost = self.compute_ssd_fft(target_tile, source_fft)
            
            # Apply color correction
            correction = self.compute_optimal_color_correction(target_tile, self.tiles[k])
            
            # Weight the cost by color correction factors
            cost = cost / (np.mean(correction) + 1e-6)
            costs.append(cost)
            
        return i, j, costs

    def create_mosaic(self, image_path, output_path):
        """
        Create mosaic with improved color handling and better global optimization
        """
        start_time = time.time()
        
        # Load and preprocess target image
        target = cv2.imread(image_path)
        if target is None:
            raise ValueError(f"Could not load image: {image_path}")
            
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
        target_yiq = self.rgb_to_yiq(target)
        
        h, w = target.shape[:2]
        n_tiles_h = h // self.tile_size
        n_tiles_w = w // self.tile_size
        n_tiles = n_tiles_h * n_tiles_w
        
        # Prepare arguments for parallel processing
        process_args = []
        for i in range(n_tiles_h):
            for j in range(n_tiles_w):
                y1, y2 = i * self.tile_size, (i + 1) * self.tile_size
                x1, x2 = j * self.tile_size, (j + 1) * self.tile_size
                target_tile = target_yiq[y1:y2, x1:x2]
                process_args.append((i, j, target_tile))
        
        # Initialize cost matrix
        cost_matrix = np.zeros((n_tiles, len(self.tiles)))
        
        print("\nComputing matching costs using", self.num_processes, "processes...")
        with Pool(processes=self.num_processes) as pool:
            results = []
            with tqdm(total=len(process_args)) as pbar:
                for result in pool.imap_unordered(self.process_tile_position, process_args):
                    i, j, costs = result
                    tile_idx = i * n_tiles_w + j
                    cost_matrix[tile_idx] = costs
                    pbar.update(1)
        
        print("\nSolving assignment problem...")
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        print("\nCreating final mosaic...")
        mosaic = np.zeros_like(target)
        
        with tqdm(total=len(row_ind)) as pbar:
            for tile_idx, source_idx in zip(row_ind, col_ind):
                i = tile_idx // n_tiles_w
                j = tile_idx % n_tiles_w
                
                y1, y2 = i * self.tile_size, (i + 1) * self.tile_size
                x1, x2 = j * self.tile_size, (j + 1) * self.tile_size
                
                source_tile = self.tiles[source_idx].copy()
                correction = self.compute_optimal_color_correction(
                    target_yiq[y1:y2, x1:x2], source_tile)
                
                # Apply color correction to each channel
                for c in range(3):
                    source_tile[:,:,c] *= correction[c]
                
                # Convert back to RGB
                rgb_tile = self.yiq_to_rgb(source_tile)
                mosaic[y1:y2, x1:x2] = rgb_tile
                pbar.update(1)
        
        # Save final result
        cv2.imwrite(output_path, cv2.cvtColor(mosaic, cv2.COLOR_RGB2BGR))
        
        execution_time = time.time() - start_time
        print(f"\nMosaic creation completed!")
        print(f"Total execution time: {execution_time:.2f} seconds")
        print(f"                     ({execution_time/60:.2f} minutes)")
        print(f"Using {self.num_processes} processes")

def create_cutout_mosaic(image_path, tiles_dir, output_path, tile_size=50):
    mosaic = CutOutMosaic(tile_size=tile_size)
    mosaic.load_tiles(tiles_dir)
    mosaic.create_mosaic(image_path, output_path)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Create a Cut-Out Image Mosaic')
    parser.add_argument('image', help='Path to input image')
    parser.add_argument('tiles', help='Directory containing tile images')
    parser.add_argument('output', help='Path for output mosaic')
    parser.add_argument('--tile-size', type=int, default=50, help='Size of mosaic tiles')
    
    args = parser.parse_args()
    
    create_cutout_mosaic(
        args.image,
        args.tiles,
        args.output,
        tile_size=args.tile_size
    )