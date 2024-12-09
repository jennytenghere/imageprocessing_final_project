# Photomosaic Generator - Baseline 1 (Rectangle Tile)

This implementation is based on the paper ["Generating photomosaics: an empirical study" (A. Finkelstein, M. Range)](https://dl.acm.org/doi/pdf/10.1145/298151.298213), focusing on the basic rectangular tile approach using L1 distance for tile matching.

## Algorithm

The algorithm uses L1 distance to find the best matching tile for each block in the original image. For each pixel, the L1 distance is calculated as:

$$
d = |A_r(i,j) - T_r(i,j)| + |A_g(i,j) - T_g(i,j)| + |A_b(i,j) - T_b(i,j)|
$$

where:
- `Ar(i,j), Ag(i,j), Ab(i,j)` are the RGB values of the original image
- `Tr(i,j), Tg(i,j), Tb(i,j)` are the RGB values of the tile image
- The total distance is summed over all pixels in the block


## Requirements

- Python 3.x
- PIL (Python Imaging Library)
```bash
pip install Pillow tqdm
```

## Usage

```bash
python photomosaicTran.py <original_image> <tiles_directory> <tile size>
```

### Parameters

- `original_image`: Path to the image you want to convert into a mosaic
- `tiles_directory`: Directory containing the tile images
- `tile size` : Size of each mosaic tile in pixels

### Configuration

You can modify these parameters in the code:
- `ENLARGEMENT`: Output image size multiplier (default: 8)

## Output

The program will generate a file named `mosaic.jpeg` in the current directory.

## Metrics

### Min Loss Per Pixel

The Min Loss Per Pixel metric evaluates how well the mosaic represents the original image at the pixel level. It is calculated as:

$$
MLPP = \frac{\sum_{blocks} min\_distance}{total\_pixels \times 3}
$$

where:
- `min_distance` is the minimum L1 distance between a block and its best matching tile
- `total_pixels` is the number of pixels in the matched area
- multiply by 3 accounts for RGB channels
- Range: [0, 255], lower values indicate better matches

For each block, the minimum L1 distance is:

$$
min\_distance = min_{t \in tiles} \sum_{i,j} (|A_r(i,j) - T_r(i,j)| + |A_g(i,j) - T_g(i,j)| + |A_b(i,j) - T_b(i,j)|)
$$

### Execution time
The total execution time is broken down into three main phases:

1. **Target Image Processing**: $O(WH)$
   - Loading and resizing the original image
   - Converting to RGB format

2. **Tile Processing**: $O(T \times w \times h)$
   - Reading and processing tile images
   - Resizing to standard tile size
   - Converting to RGB format
   Where T is number of tiles, w and h are tile dimensions

3. **Mosaic Creation**: $O(WH \times T)$
   - Computing L1 distances
   - Finding best matching tiles
   - Assembling final mosaic

Total time complexity: $O(WH \times T)$, dominated by the mosaic creation phase

The actual runtime is reported in both seconds and minutes, allowing for:
- Performance benchmarking
- Resource planning for large-scale mosaic generation
- Optimization of processing parameters
## Resource Requirements

- Memory usage depends on the size of the input image and number of tiles
- Disk space required for the output image will be approximately `(original_width * ENLARGEMENT) * (original_height * ENLARGEMENT) * 3` bytes


## Acknowledgments
Based on the algorithm described in ["Generating photomosaics: an empirical study" (A. Finkelstein, M. Range)](https://dl.acm.org/doi/pdf/10.1145/298151.298213)