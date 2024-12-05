import sys
import os
import time
from PIL import Image, ImageOps
from tqdm import tqdm

# Configuration parameters
TILE_SIZE = 50      # height/width of mosaic tiles in pixels
ENLARGEMENT = 1     # the mosaic image will be this many times wider and taller than the original
OUT_FILE = 'mosaic.jpeg'

class TileProcessor:
    def __init__(self, tiles_directory):
        self.tiles_directory = tiles_directory

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
            tile_img = img.resize((TILE_SIZE, TILE_SIZE), Image.LANCZOS)
            return tile_img.convert('RGB')
        except:
            return None

    def get_tiles(self):
        tiles = []
        print('Reading tiles from {}...'.format(self.tiles_directory))

        # Get total number of files for progress bar
        total_files = sum([len(files) for r, d, files in os.walk(self.tiles_directory)])
        
        with tqdm(total=total_files, desc="Processing tiles") as pbar:
            for root, subFolders, files in os.walk(self.tiles_directory):
                for tile_name in files:
                    tile_path = os.path.join(root, tile_name)
                    tile = self.__process_tile(tile_path)
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
        w_diff = (w % TILE_SIZE)/2
        h_diff = (h % TILE_SIZE)/2
        if w_diff or h_diff:
            large_img = large_img.crop((w_diff, h_diff, w - w_diff, h - h_diff))

        return large_img.convert('RGB')

def calculate_l1_distance(block_data, tile_data):
    distance = 0
    for i in range(len(block_data)):
        # Get RGB values from tuples
        r1, g1, b1 = block_data[i]
        r2, g2, b2 = tile_data[i]
        # Calculate L1 distance for RGB values
        distance += abs(r1 - r2) + abs(g1 - g2) + abs(b1 - b2)
    return distance

def create_mosaic(original_img, tiles):
    """
    Create mosaic following the paper's O(WHTwh) implementation
    """
    print('Building mosaic...')
    
    # Create output image
    result = Image.new('RGB', original_img.size)
    
    # Convert tiles to data arrays
    tile_data = [list(tile.getdata()) for tile in tiles]
    
    W = original_img.size[0]
    H = original_img.size[1]
    total_blocks = (W // TILE_SIZE) * (H // TILE_SIZE)
    
    # Setup progress bar for blocks processing
    with tqdm(total=total_blocks, desc="Creating mosaic") as pbar:
        # Iterate through each block in the original image O(WH)
        for y in range(0, H - TILE_SIZE + 1, TILE_SIZE):
            for x in range(0, W - TILE_SIZE + 1, TILE_SIZE):
                # Get current block data
                block = original_img.crop((x, y, x + TILE_SIZE, y + TILE_SIZE))
                block_data = list(block.getdata())
                
                # Find best matching tile O(T)
                best_distance = float('inf')
                best_tile_idx = 0
                
                # Compare with each tile O(T * w * h)
                for idx, curr_tile_data in enumerate(tile_data):
                    distance = calculate_l1_distance(block_data, curr_tile_data)
                    if distance < best_distance:
                        best_distance = distance
                        best_tile_idx = idx
                
                # Place the best matching tile
                result.paste(tiles[best_tile_idx], (x, y))
                pbar.update(1)
    
    print('Saving mosaic...')
    result.save(OUT_FILE)
    print('Finished! Output is in', OUT_FILE)

def show_error(msg):
    print('ERROR: {}'.format(msg))

def create_photomosaic(img_path, tiles_path):
    # Start timing
    start_time = time.time()
    
    # Process images
    print("Step 1: Processing target image")
    image_data = TargetImage(img_path).get_data()
    
    print("\nStep 2: Processing tile images")
    tiles_data = TileProcessor(tiles_path).get_tiles()
    
    if tiles_data:
        print("\nStep 3: Creating mosaic")
        create_mosaic(image_data, tiles_data)
        
        # Calculate and display total execution time
        end_time = time.time()
        execution_time = end_time - start_time
        
        print(f'\nTotal execution time: {execution_time:.2f} seconds')
        print(f'                     ({execution_time/60:.2f} minutes)')
        
        # Display image information
        print(f'\nImage details:')
        print(f'- Original size: {image_data.size[0]}x{image_data.size[1]} pixels')
        print(f'- Tile size: {TILE_SIZE}x{TILE_SIZE} pixels')
        print(f'- Number of tiles used: {len(tiles_data)}')
        print(f'- Total tiles placed: {(image_data.size[0] // TILE_SIZE) * (image_data.size[1] // TILE_SIZE)}')
    else:
        show_error("No images found in tiles directory '{}'".format(tiles_path))

if __name__ == '__main__':
    if len(sys.argv) < 3:
        show_error('Usage: {} <image> <tiles directory>\r'.format(sys.argv[0]))
    else:
        source_image = sys.argv[1]
        tile_dir = sys.argv[2]
        
        if not os.path.isfile(source_image):
            show_error("Unable to find image file '{}'".format(source_image))
        elif not os.path.isdir(tile_dir):
            show_error("Unable to find tile directory '{}'".format(tile_dir))
        else:
            create_photomosaic(source_image, tile_dir)