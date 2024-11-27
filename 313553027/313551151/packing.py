import os
import argparse
import torch
from PIL import Image
import numpy as np

def load_and_resize_images(image_dir, height, width):
    images = []
    for filename in os.listdir(image_dir):
        filepath = os.path.join(image_dir, filename)
        try:
            with Image.open(filepath) as img:
                img = img.resize((width, height))
                img = np.array(img, dtype=np.uint8)
                if img.shape[-1] == 3:  # Ensure it's an RGB image
                    images.append(img)
        except Exception as e:
            print(f"Skipping {filename}: {e}")
    return np.array(images, dtype=np.uint8)

def main():
    parser = argparse.ArgumentParser(description="Pack images into a tensor.")
    parser.add_argument("-d", "--directory", type=str, required=True, help="Directory containing images.")
    parser.add_argument("-ht", "--height", type=int, required=True, help="Height to resize images.")
    parser.add_argument("-w", "--width", type=int, required=True, help="Width to resize images.")
    args = parser.parse_args()

    images = load_and_resize_images(args.directory, args.height, args.width)
    if len(images) == 0:
        print("No valid images found.")
        return

    image_tensor = torch.tensor(images, dtype=torch.uint8)
    output_path = f"data_{len(images)}_{args.height}_{args.width}.pt"
    torch.save(image_tensor, output_path)
    print(f"Saved tensor of shape {image_tensor.shape} to {output_path}")

if __name__ == "__main__":
    main()

