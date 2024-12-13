import sys
import os
import time
from PIL import Image, ImageOps
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count
import numpy as np

# Configuration parameters
ENLARGEMENT = 1     # the mosaic image will be this many times wider and taller than the original
MAX_WORKERS = cpu_count()  

class TileProcessor:
    def __init__(self, tiles_directory=None, use_cifar10=False, cifar10_path=None):
        self.tiles_directory = tiles_directory
        self.use_cifar10 = use_cifar10
        self.cifar10_path = cifar10_path

    def __process_tile(self, tile_path):
        try:
            img = Image.open(tile_path)
            img = ImageOps.exif_transpose(img)

            # Tiles must be square, so get the largest square that fits inside the image
            w = img.size[0]
            h = img.size[1]
            min_dimension = min(w, h)
            w_crop = (w - min_dimension) / 2
            h_crop = (h - min_dimension) / 2
            img = img.crop((w_crop, h_crop, w - w_crop, h - h_crop))

            # Resize the image to the target tile size
            tile_img = img.resize((tile_size, tile_size), Image.LANCZOS)
            return tile_img.convert('RGB')
        except:
            return None

    def process_cifar10_data(self):
        print(f"Loading CIFAR-10 tiles from {self.cifar10_path}")
        tiles = []
        
        try:
            data = np.load(self.cifar10_path)
            images = data['data']
            
            if len(images.shape) == 2:
                images = images.reshape(-1, 3, 32, 32)
                images = images.transpose(0, 2, 3, 1)
            
            with tqdm(total=len(images), desc="Processing CIFAR-10 tiles") as pbar:
                for img_array in images:
                    img = Image.fromarray(img_array)
                    img = img.resize((tile_size, tile_size), Image.LANCZOS)
                    tiles.append(img)
                    pbar.update(1)
            
            print(f'Processed {len(tiles)} CIFAR-10 tiles')
            return tiles
            
        except Exception as e:
            print(f"Error loading CIFAR-10 data: {e}")
            return []

    def get_tiles(self):
        if self.use_cifar10 and self.cifar10_path:
            return self.process_cifar10_data()
            
        tiles = []
        print('Reading tiles from {}...'.format(self.tiles_directory))

        # Get all tile paths
        tile_paths = []
        for root, _, files in os.walk(self.tiles_directory):
            for tile_name in files:
                if tile_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    tile_paths.append(os.path.join(root, tile_name))

        if not tile_paths:
            print("No valid image files found in the tiles directory")
            return []

        # Process tiles using thread pool
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Submit all tasks and create future-to-path mapping
            future_to_path = {executor.submit(self.__process_tile, path): path 
                            for path in tile_paths}
            
            # Process completed tasks with progress bar
            with tqdm(total=len(tile_paths), desc="Processing tiles") as pbar:
                for future in as_completed(future_to_path):
                    tile = future.result()
                    if tile:
                        tiles.append(tile)
                    pbar.update(1)

        print('Processed {} tiles.'.format(len(tiles)))
        return tiles

class TargetImage:
    def __init__(self, image_path):
        self.image_path = image_path

    def get_data(self):
        print('Processing main image...')
        img = Image.open(self.image_path)
        w = img.size[0] * ENLARGEMENT
        h = img.size[1] * ENLARGEMENT
        large_img = img.resize((w, h), Image.LANCZOS)
        
        # Crop to ensure we have complete tiles
        w_diff = (w % tile_size)/2
        h_diff = (h % tile_size)/2
        if w_diff or h_diff:
            large_img = large_img.crop((w_diff, h_diff, w - w_diff, h - h_diff))

        return large_img.convert('RGB')

def find_best_match(args):
    block_data, tile_data_list = args
    best_distance = float('inf')
    best_tile_idx = 0

    for idx, curr_tile_data in enumerate(tile_data_list):
        distance = sum(abs(a - b) for p1, p2 in zip(block_data, curr_tile_data)
                      for a, b in zip(p1, p2))
        if distance < best_distance:
            best_distance = distance
            best_tile_idx = idx

    return best_tile_idx, best_distance

def create_mosaic(original_img, tiles):
    """
    Create mosaic with multithreading support
    """
    print('Building mosaic...')
    
    # Create output image
    result = Image.new('RGB', original_img.size)
    
    # Convert tiles to data arrays
    tile_data = [list(tile.getdata()) for tile in tiles]
    
    W = original_img.size[0]
    H = original_img.size[1]
    
    # Prepare blocks data
    blocks_data = []
    positions = []
    
    for y in range(0, H - tile_size + 1, tile_size):
        for x in range(0, W - tile_size + 1, tile_size):
            block = original_img.crop((x, y, x + tile_size, y + tile_size))
            blocks_data.append((list(block.getdata()), tile_data))
            positions.append((x, y))

    total_blocks = len(blocks_data)
    total_loss = 0
    total_pixels = 0

    # Process blocks using thread pool
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_block = {executor.submit(find_best_match, block_data): idx 
                          for idx, block_data in enumerate(blocks_data)}
        
        # Process completed tasks with progress bar
        with tqdm(total=total_blocks, desc="Creating mosaic") as pbar:
            for future in as_completed(future_to_block):
                block_idx = future_to_block[future]
                best_tile_idx, distance = future.result()
                
                # Place the best matching tile
                x, y = positions[block_idx]
                result.paste(tiles[best_tile_idx], (x, y))
                
                # Update metrics
                total_loss += distance
                total_pixels += tile_size * tile_size * 3
                
                pbar.update(1)

    # Save the result
    print('Finished creating mosaic!')
    
    # Calculate average loss
    min_loss_per_pixel = total_loss / total_pixels
    print(f'\nMetrics:')
    print(f'- Total L1 Loss: {total_loss:,}')
    print(f'- Total Pixels (RGB channels): {total_pixels:,}')
    print(f'- Min Loss Per Pixel: {min_loss_per_pixel:.2f}')
    
    return result

def show_error(msg):
    print('ERROR: {}'.format(msg))

def create_photomosaic(img_path, output_path='mosaic.jpeg', tiles_path=None, use_cifar10=False, cifar10_path=None):
    """
    Create a photomosaic from an input image using either a directory of tiles or CIFAR-10 dataset
    """
    # Start timing
    start_time = time.time()
    
    # Process images
    print("Step 1: Processing target image")
    image_data = TargetImage(img_path).get_data()
    
    print("\nStep 2: Processing tile images")
    tiles_data = TileProcessor(tiles_path, use_cifar10, cifar10_path).get_tiles()
    
    if tiles_data:
        print("\nStep 3: Creating mosaic")
        result = create_mosaic(image_data, tiles_data)
        
        # Save the result
        print(f'Saving mosaic to {output_path}...')
        result.save(output_path)
        print(f'Saved mosaic to {output_path}')
        
        # Calculate and display total execution time
        end_time = time.time()
        execution_time = end_time - start_time
        
        print(f'\n - Total execution time: {execution_time:.2f} seconds')
        print(f'                           ({execution_time/60:.2f} minutes)')
        
        # Display image information
        print(f'\nImage details:')
        print(f'- Original size: {image_data.size[0]}x{image_data.size[1]} pixels')
        print(f'- Tile size: {tile_size}x{tile_size} pixels')
        print(f'- Number of tiles used: {len(tiles_data)}')
        print(f'- Total tiles placed: {(image_data.size[0] // tile_size) * (image_data.size[1] // tile_size)}')
    else:
        show_error("No valid tiles found")

if __name__ == '__main__':
    if len(sys.argv) < 4:
        show_error('''Usage: 
        For tile directory: {} <image> <tiles directory> <tile size> [output path]
        For CIFAR-10: {} <image> --cifar10 <cifar10_path> <tile size> [output path]
        '''.format(sys.argv[0], sys.argv[0]))
    else:
        source_image = sys.argv[1]
        output_path = 'mosaic.jpeg' 
        
        if sys.argv[2] == '--cifar10':
            use_cifar10 = True
            cifar10_path = sys.argv[3]
            tile_size = int(sys.argv[4])
            if len(sys.argv) > 5:
                output_path = sys.argv[5]
                
            if not os.path.isfile(cifar10_path):
                show_error(f"Unable to find CIFAR-10 file '{cifar10_path}'")
            else:
                create_photomosaic(source_image, output_path, use_cifar10=True, cifar10_path=cifar10_path)
        else:
            tile_dir = sys.argv[2]
            tile_size = int(sys.argv[3])
            if len(sys.argv) > 4:
                output_path = sys.argv[4]
                
            if not os.path.isfile(source_image):
                show_error(f"Unable to find image file '{source_image}'")
            elif not os.path.isdir(tile_dir):
                show_error(f"Unable to find tile directory '{tile_dir}'")
            else:
                create_photomosaic(source_image, output_path, tiles_path=tile_dir)