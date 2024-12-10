import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
import multiprocessing as mp
from pathlib import Path
import time

class ImageTile:
    def __init__(self, image, avg_color):
        self.image = image
        self.avg_color = avg_color

class ImageSetMosaic:
    def __init__(self, tile_size=50, num_iterations=20):
        self.tile_size = tile_size
        self.num_iterations = num_iterations
        self.tiles = []
        self.tile_colors = []
        self.n_processes = max(1, mp.cpu_count() - 1)
        
    def load_tiles(self, tiles_dir):
        """Load and process tile images"""
        print(f"Loading tiles from {tiles_dir}")
        tiles_dir = Path(tiles_dir)
        
        for file_path in tqdm(list(tiles_dir.glob("*"))):
            if file_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                try:
                    # Read image
                    img = cv2.imread(str(file_path))
                    if img is None:
                        continue
                        
                    # Convert to RGB
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    # Make square
                    h, w = img.shape[:2]
                    size = min(h, w)
                    start_h = (h - size) // 2
                    start_w = (w - size) // 2
                    img = img[start_h:start_h+size, start_w:start_w+size]
                    
                    # Resize
                    img = cv2.resize(img, (self.tile_size, self.tile_size))
                    
                    # Calculate average color
                    avg_color = np.mean(img, axis=(0,1))
                    
                    # Store tile
                    self.tiles.append(ImageTile(img, avg_color))
                    self.tile_colors.append(avg_color)
                    
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
        
        self.tile_colors = np.array(self.tile_colors)
        print(f"Loaded {len(self.tiles)} tiles")
        
    def compute_direction_field(self, image, sigma=2.0):
        """Compute edge direction field"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        
        # Compute gradients
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Compute magnitude and direction
        magnitude = np.sqrt(gx**2 + gy**2)
        angle = np.arctan2(gy, gx)
        
        # Smooth fields
        magnitude = gaussian_filter(magnitude, sigma)
        cos_angle = gaussian_filter(np.cos(angle), sigma)
        sin_angle = gaussian_filter(np.sin(angle), sigma)
        smoothed_angle = np.arctan2(sin_angle, cos_angle)
        
        return smoothed_angle, magnitude
        
    def create_distance_field(self, shape, site, direction):
        """Create manhattan distance field for a site"""
        h, w = shape
        y, x = np.mgrid[0:h, 0:w]
        
        # Compute distances in rotated coordinate system
        dx = x - site[0]
        dy = y - site[1]
        cos_theta = np.cos(direction)
        sin_theta = np.sin(direction)
        x_rot = dx * cos_theta + dy * sin_theta
        y_rot = -dx * sin_theta + dy * cos_theta
        
        return np.abs(x_rot) + np.abs(y_rot)
        
    def process_voronoi_chunk(self, args):
        """Process a chunk of Voronoi diagram computation"""
        start_idx, end_idx, shape, sites, directions = args
        h, w = shape
        
        distance_field = np.full((h, w), np.inf)
        label_field = np.zeros((h, w), dtype=int)
        
        for i in range(start_idx, end_idx):
            dist = self.create_distance_field(shape, sites[i], directions[i])
            mask = dist < distance_field
            distance_field[mask] = dist[mask]
            label_field[mask] = i
            
        return distance_field, label_field
        
    def compute_voronoi_regions(self, shape, sites, directions):
        """Compute Voronoi regions using parallel processing"""
        n_sites = len(sites)
        chunk_size = max(1, n_sites // self.n_processes)
        chunks = []
        
        # Prepare chunks
        for i in range(0, n_sites, chunk_size):
            end_idx = min(i + chunk_size, n_sites)
            chunks.append((i, end_idx, shape, sites, directions))
            
        # Process in parallel
        with mp.Pool(self.n_processes) as pool:
            results = pool.map(self.process_voronoi_chunk, chunks)
            
        # Combine results
        final_distance = np.full(shape, np.inf)
        final_labels = np.zeros(shape, dtype=int)
        
        for dist, labels in results:
            mask = dist < final_distance
            final_distance[mask] = dist[mask]
            final_labels[mask] = labels[mask]
            
        return final_labels
        
    def find_best_tile(self, target_color, n_candidates=5):
        """Find best matching tile for target color"""
        distances = np.sqrt(np.sum((self.tile_colors - target_color)**2, axis=1))
        candidate_indices = np.argsort(distances)[:n_candidates]
        return self.tiles[np.random.choice(candidate_indices)]
        
    def place_tile(self, result, site, direction, tile):
        """Place a tile in the result image"""
        h, w = result.shape[:2]
        x, y = int(site[0]), int(site[1])
        
        if 0 <= x < w and 0 <= y < h:
            # Create rotation matrix
            M = cv2.getRotationMatrix2D(
                (self.tile_size//2, self.tile_size//2),
                np.degrees(direction),
                1.0
            )
            
            # Rotate tile
            rotated_tile = cv2.warpAffine(
                tile.image,
                M,
                (self.tile_size, self.tile_size),
                flags=cv2.INTER_LINEAR
            )
            
            # Calculate placement coordinates
            y1 = max(0, y - self.tile_size//2)
            y2 = min(h, y + self.tile_size//2)
            x1 = max(0, x - self.tile_size//2)
            x2 = min(w, x + self.tile_size//2)
            
            if y2 > y1 and x2 > x1:
                tile_region = rotated_tile[
                    self.tile_size//2-(y-y1):self.tile_size//2+(y2-y),
                    self.tile_size//2-(x-x1):self.tile_size//2+(x2-x)
                ]
                
                # Only update non-black pixels
                mask = np.any(tile_region > 0, axis=2)
                result[y1:y2, x1:x2][mask] = tile_region[mask]
                
    def create_mosaic(self, image, edge_weight=1.0):
        """Create mosaic using image tiles"""
        h, w = image.shape[:2]
        print(f"Creating mosaic for image of size {w}x{h}")
        
        # Compute direction field
        print("Computing direction field...")
        angle_field, magnitude = self.compute_direction_field(image)
        
        # Initialize sites
        n_tiles_x = w // self.tile_size
        n_tiles_y = h // self.tile_size
        x_coords = np.linspace(self.tile_size, w-self.tile_size, n_tiles_x)
        y_coords = np.linspace(self.tile_size, h-self.tile_size, n_tiles_y)
        xx, yy = np.meshgrid(x_coords, y_coords)
        sites = np.stack([
            xx.ravel() + np.random.normal(0, self.tile_size/4, xx.size),
            yy.ravel() + np.random.normal(0, self.tile_size/4, yy.size)
        ], axis=1)
        
        # Create edge mask
        edge_mask = (magnitude > np.percentile(magnitude, 90)).astype(float)
        edge_mask = gaussian_filter(edge_mask, self.tile_size/4)
        
        # Main iteration loop
        print("Starting CVD iterations...")
        for iteration in tqdm(range(self.num_iterations)):
            # Get directions for current sites
            site_directions = np.array([
                angle_field[min(h-1, max(0, int(y))), min(w-1, max(0, int(x)))]
                for x, y in sites
            ])
            
            # Compute Voronoi regions
            regions = self.compute_voronoi_regions((h, w), sites, site_directions)
            
            # Update sites
            new_sites = []
            for i in range(len(sites)):
                region_mask = (regions == i)
                y, x = np.nonzero(region_mask)
                if len(x) > 0:
                    centroid = np.array([np.mean(x), np.mean(y)])
                    
                    # Apply edge avoidance
                    cx, cy = int(centroid[0]), int(centroid[1])
                    if 0 <= cx < w and 0 <= cy < h:
                        edge_influence = edge_mask[cy, cx] * edge_weight
                        direction = angle_field[cy, cx]
                        offset = edge_influence * np.array([np.cos(direction), np.sin(direction)])
                        centroid = centroid + offset
                    
                    new_sites.append(centroid)
            
            sites = np.array(new_sites)
        
        # Create final mosaic
        print("Creating final mosaic...")
        result = np.zeros_like(image)
        
        # Place tiles
        for site in tqdm(sites):
            x, y = int(site[0]), int(site[1])
            if 0 <= x < w and 0 <= y < h:
                target_color = image[y, x]
                direction = angle_field[y, x]
                best_tile = self.find_best_tile(target_color)
                self.place_tile(result, site, direction, best_tile)
        
        return result

def create_image_mosaic(image_path, tiles_dir, output_path, tile_size=50, num_iterations=20, edge_weight=1.0):
    """Helper function to create image mosaic"""
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Create mosaic
    start_time = time.time()
    
    mosaic = ImageSetMosaic(tile_size=tile_size, num_iterations=num_iterations)
    mosaic.load_tiles(tiles_dir)
    result = mosaic.create_mosaic(image, edge_weight=edge_weight)
    
    elapsed_time = time.time() - start_time
    print(f"\nTotal processing time: {elapsed_time:.2f} seconds")
    
    # Save result
    plt.imsave(output_path, result)
    print(f"Mosaic saved to {output_path}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Create image mosaic')
    parser.add_argument('image', help='Input image path')
    parser.add_argument('tiles_dir', help='Directory containing tile images')
    parser.add_argument('--tile-size', type=int, default=50, help='Size of mosaic tiles')
    parser.add_argument('--iterations', type=int, default=20, help='Number of iterations')
    parser.add_argument('--edge-weight', type=float, default=1.0, help='Edge influence weight')
    parser.add_argument('--output', default='mosaic.png', help='Output image path')
    
    args = parser.parse_args()
    
    try:
        create_image_mosaic(
            args.image,
            args.tiles_dir,
            args.output,
            tile_size=args.tile_size,
            num_iterations=args.iterations,
            edge_weight=args.edge_weight
        )
    except Exception as e:
        print(f"Error: {e}")