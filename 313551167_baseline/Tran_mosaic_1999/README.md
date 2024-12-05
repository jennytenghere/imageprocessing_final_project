# Photomosaic Generator - Baseline 1 (Rectangle Tile)

This implementation is based on the paper [Generating photomosaics: an empirical study](https://dl.acm.org/doi/pdf/10.1145/298151.298213), focusing on the basic rectangular tile approach using L1 distance for tile matching.

## Algorithm

The algorithm uses L1 distance to find the best matching tile for each block in the original image. For each pixel, the L1 distance is calculated as:

$$
d = |A_r(i,j) - T_r(i,j)| + |A_g(i,j) - T_g(i,j)| + |A_b(i,j) - T_b(i,j)|
$$

where:
- `Ar(i,j), Ag(i,j), Ab(i,j)` are the RGB values of the original image
- `Tr(i,j), Tg(i,j), Tb(i,j)` are the RGB values of the tile image
- The total distance is summed over all pixels in the block

### Time Complexity
$$
O(W\times H \times T), \text{ where } W : \text{Width}, H:\text{Height ****}, T:\text{numbers of tiles}
$$
## Requirements

- Python 3.x
- PIL (Python Imaging Library)
```bash
pip install Pillow tqdm
```

## Usage

```bash
python mosaic.py <original_image> <tiles_directory>
```

### Parameters

- `original_image`: Path to the image you want to convert into a mosaic
- `tiles_directory`: Directory containing the tile images

### Configuration

You can modify these parameters in the code:
- `TILE_SIZE`: Size of each mosaic tile in pixels (default: 50)
- `ENLARGEMENT`: Output image size multiplier (default: 8)
- `WORKER_COUNT`: Number of parallel processes (default: CPU count - 1)

## Output

The program will generate a file named `mosaic.jpeg` in the current directory.


## Resource Requirements

- Memory usage depends on the size of the input image and number of tiles
- CPU usage scales with `WORKER_COUNT`
- Disk space required for the output image will be approximately `(original_width * ENLARGEMENT) * (original_height * ENLARGEMENT) * 3` bytes


## Author

Nicholas Tran

## Acknowledgments
Based on the algorithm described in "Generating photomosaics: an empirical study" (A. Finkelstein, M. Range)