import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
import time
import multiprocessing as mp
from functools import partial

class ParallelHausnerMosaic:
    def __init__(self, tile_size=10, num_iterations=20):
        self.tile_size = tile_size
        self.num_iterations = num_iterations
        self.debug = True
        self.n_processes = max(1, mp.cpu_count() - 1)  # Leave one core free
        
    def create_distance_field(self, shape, site, direction):
        """Create manhattan distance field for a site"""
        h, w = shape
        y, x = np.mgrid[0:h, 0:w]
        
        dx = x - site[0]
        dy = y - site[1]
        cos_theta = np.cos(direction)
        sin_theta = np.sin(direction)
        x_rot = dx * cos_theta + dy * sin_theta
        y_rot = -dx * sin_theta + dy * cos_theta
        
        return np.abs(x_rot) + np.abs(y_rot)

    def process_voronoi_chunk(self, chunk_data):
        """Process a chunk of the Voronoi diagram computation"""
        start_idx, end_idx, shape, sites, directions = chunk_data
        h, w = shape
        
        distance_field = np.full((h, w), np.inf)
        label_field = np.zeros((h, w), dtype=int)
        
        for i in range(start_idx, end_idx):
            dist = self.create_distance_field(shape, sites[i], directions[i])
            mask = dist < distance_field
            distance_field[mask] = dist[mask]
            label_field[mask] = i
            
        return distance_field, label_field

    def compute_voronoi_regions_parallel(self, shape, sites, directions):
        """Compute Voronoi regions using parallel processing"""
        n_sites = len(sites)
        chunk_size = max(1, n_sites // self.n_processes)
        chunks = []
        
        # Prepare chunks for parallel processing
        for i in range(0, n_sites, chunk_size):
            end_idx = min(i + chunk_size, n_sites)
            chunks.append((i, end_idx, shape, sites, directions))
        
        # Process chunks in parallel
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

    def render_tile_chunk(self, chunk_data):
        """Render a chunk of tiles"""
        start_idx, end_idx, sites, angle_field, image, h, w = chunk_data
        result = np.zeros((h, w, 3), dtype=np.uint8)
        
        for i in range(start_idx, end_idx):
            site = sites[i]
            x, y = int(site[0]), int(site[1])
            
            if 0 <= x < w and 0 <= y < h:
                # Get tile color and direction
                color = image[y, x]
                direction = angle_field[y, x]
                
                # Create and rotate tile
                tile = np.zeros((self.tile_size*2, self.tile_size*2, 3), dtype=np.uint8)
                center = self.tile_size
                pts = np.array([
                    [center-self.tile_size//2, center-self.tile_size//2],
                    [center+self.tile_size//2, center-self.tile_size//2],
                    [center+self.tile_size//2, center+self.tile_size//2],
                    [center-self.tile_size//2, center+self.tile_size//2]
                ], dtype=np.float32)
                
                M = cv2.getRotationMatrix2D((center, center), np.degrees(direction), 1.0)
                pts = cv2.transform(pts.reshape(1, -1, 2), M).reshape(-1, 2)
                cv2.fillPoly(tile, [pts.astype(np.int32)], color.tolist())
                
                # Place tile
                y1 = max(0, y-self.tile_size)
                y2 = min(h, y+self.tile_size)
                x1 = max(0, x-self.tile_size)
                x2 = min(w, x+self.tile_size)
                
                if y2 > y1 and x2 > x1:
                    tile_region = tile[
                        self.tile_size-(y-y1):self.tile_size+(y2-y),
                        self.tile_size-(x-x1):self.tile_size+(x2-x)
                    ]
                    mask = np.any(tile_region > 0, axis=2)
                    result[y1:y2, x1:x2][mask] = tile_region[mask]
        
        return result

    def create_mosaic(self, image, edge_weight=1.0):
        """Create decorative mosaic with parallel processing"""
        h, w = image.shape[:2]
        print(f"Creating mosaic using {self.n_processes} processes")
        
        # Compute edge direction field
        print("Computing direction field...")
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        magnitude = np.sqrt(gx**2 + gy**2)
        angle = np.arctan2(gy, gx)
        
        magnitude = gaussian_filter(magnitude, 2.0)
        cos_angle = gaussian_filter(np.cos(angle), 2.0)
        sin_angle = gaussian_filter(np.sin(angle), 2.0)
        angle_field = np.arctan2(sin_angle, cos_angle)
        
        # Initialize sites
        print("Initializing sites...")
        n_tiles_x = w // self.tile_size
        n_tiles_y = h // self.tile_size
        x_coords = np.linspace(self.tile_size, w-self.tile_size, n_tiles_x)
        y_coords = np.linspace(self.tile_size, h-self.tile_size, n_tiles_y)
        xx, yy = np.meshgrid(x_coords, y_coords)
        sites = np.stack([
            xx.ravel() + np.random.normal(0, self.tile_size/4, xx.size),
            yy.ravel() + np.random.normal(0, self.tile_size/4, yy.size)
        ], axis=1)
        
        # Edge mask for edge avoidance
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
            
            # Compute Voronoi regions in parallel
            regions = self.compute_voronoi_regions_parallel((h, w), sites, site_directions)
            
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
        
        # Render final mosaic in parallel
        print("Rendering final mosaic...")
        n_sites = len(sites)
        chunk_size = max(1, n_sites // self.n_processes)
        chunks = []
        
        for i in range(0, n_sites, chunk_size):
            end_idx = min(i + chunk_size, n_sites)
            chunks.append((i, end_idx, sites, angle_field, image, h, w))
        
        with mp.Pool(self.n_processes) as pool:
            chunk_results = pool.map(self.render_tile_chunk, chunks)
        
        # Combine results
        result = np.zeros_like(image)
        for chunk_result in chunk_results:
            mask = np.any(chunk_result > 0, axis=2)
            result[mask] = chunk_result[mask]
        
        return result

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Create parallel Hausner-style mosaic')
    parser.add_argument('image', help='Input image path')
    parser.add_argument('--tile-size', type=int, default=10, help='Size of mosaic tiles')
    parser.add_argument('--iterations', type=int, default=20, help='Number of iterations')
    parser.add_argument('--edge-weight', type=float, default=1.0, help='Edge influence weight')
    parser.add_argument('--output', default='mosaic.png', help='Output image path')
    
    args = parser.parse_args()
    
    # Load image
    image = cv2.imread(args.image)
    if image is None:
        raise ValueError(f"Could not load image: {args.image}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Create mosaic
    start_time = time.time()
    
    mosaic = ParallelHausnerMosaic(
        tile_size=args.tile_size,
        num_iterations=args.iterations
    )
    result = mosaic.create_mosaic(image, edge_weight=args.edge_weight)
    
    elapsed_time = time.time() - start_time
    print(f"Total processing time: {elapsed_time:.2f} seconds")
    
    # Save result
    plt.imsave(args.output, result)
    print(f"Mosaic saved to {args.output}")