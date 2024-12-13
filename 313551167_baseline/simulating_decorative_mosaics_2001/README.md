# Photomosaic Generator - Baseline 2 (Irregular Tile)

This implementation is based on the paper [Simulating Decorative Mosaics, Alejo Hausner 2001](https://dl.acm.org/doi/pdf/10.1145/383259.383327)


## Algorithm

This implementation generates photomosaics using a Directional Voronoi Diagram technique. It optimizes tile placement by iteratively refining Voronoi regions influenced by edge detection and user-defined parameters. This advanced approach balances aesthetic quality with computational efficiency.


## Key Features

- **Directional Voronoi Tiling**: Places tiles based on edge-aligned Voronoi regions to improve detail preservation.
- **Edge-Aware Refinement**: Utilizes edge detection and direction fields to guide tile placement.
- **Parallelized Voronoi Computation**: Leverages multiprocessing to speed up the generation process.
- **Configurable Parameters**: Tile size, iteration count, and edge influence weight can be customized.

## Algorithm Overview

1. **Tile Preparation**:
   - Tiles are resized to a uniform size.
   - Average color is precomputed for matching.

2. **Edge and Direction Fields**:
   - Sobel gradients calculate edge magnitudes and directions.
   - Gaussian smoothing refines the fields.

3. **Voronoi Diagram Generation**:
   - Tiles are assigned to regions via a distance field computation, considering edge alignment.

4. **Tile Matching and Placement**:
   - The best-matching tile for each region is chosen based on Euclidean color distance.
   - Tiles are rotated and placed according to the region’s direction.

5. **Iterative Refinement**:
   - Voronoi regions are iteratively updated to align better with edges and image features.

## Requirements

- **Python 3.x**
- **Dependencies**: 
   - OpenCV (`cv2`)
   - NumPy
   - SciPy
   - tqdm
   - Matplotlib

Install dependencies with:
```bash
pip install opencv-python-headless numpy scipy tqdm matplotlib
```

## Usage
```bash
usage: photomosaic.py [-h] (--tiles-dir TILES_DIR | --use-cifar10) [--cifar10-path CIFAR10_PATH] [--tile-size TILE_SIZE] [--iterations ITERATIONS]
                      [--edge-weight EDGE_WEIGHT] [--output OUTPUT]
                      image

Create image mosaic

positional arguments:
  image                 Input image path

optional arguments:
  -h, --help            show this help message and exit
  --tiles-dir TILES_DIR
                        Directory containing tile images
  --use-cifar10         Use CIFAR-10 dataset as tiles
  --cifar10-path CIFAR10_PATH
                        Path to cifar10_train.npz file
  --tile-size TILE_SIZE
                        Size of mosaic tiles
  --iterations ITERATIONS
                        Number of iterations
  --edge-weight EDGE_WEIGHT
                        Edge influence weight
  --output OUTPUT       Output image path
```

### Example
#### Use the [Oxford 102 Flower](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/) as the tile directory
```bash
python photomosaic.py ../data/minion.jpg --tiles-dir ../data/102flowers --tile-size 20 --output mosaic_102flowers.png
```
#### Use the CIFAR10 dataset
```bash
python photomosaic.py ../data/minion.jpg --use-cifar10 --cifar10-path ../data/cifar10_train.npz --tile-size 16 --output mosaic_cifar10_16.png
```

### Output
#### **Original Image**
  
![alt text](../data/minion.jpg)

#### **Photomosaic Image with CIFAR10**
![mosaic](./mosaic_cifar10_16.png)

#### **Photomosaic Image with Oxfords 102 flowers**
![102flowers](./mosaic_102flowers.png)

## Performance and Complexity

### Runtime

The program's runtime is influenced by:
- **Image Size**: Larger images require more Voronoi computations.
- **Tile Count**: More tiles increase matching time.
- **Iterations**: Higher iteration counts improve detail but increase processing time.

## Acknowledgments
Based on the algorithm described in Hausner, Alejo. "Simulating decorative mosaics." Proceedings of the 28th annual conference on Computer graphics and interactive techniques. 2001.

