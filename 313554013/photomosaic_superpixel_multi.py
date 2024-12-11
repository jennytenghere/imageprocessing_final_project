import os
import cv2
import numpy as np
import argparse
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

def process_superpixel(idx, target_image, labels, mask, x, y, w, h, image_folder):
    part = target_image[labels == idx]
    part_mean = np.mean(part, axis=0)
    min_distance = 195075

    files = os.listdir(opt.image_pool)
    for i in files:
        tile = cv2.imread(os.path.join(image_folder, i))

        tile_h, tile_w = tile.shape[:2]
        scale_factor = max(h / tile_h, w / tile_w)
        new_tile_w = int(tile_w * scale_factor)
        new_tile_h = int(tile_h * scale_factor)
        
        mask_region = mask[y:y+h, x:x+w]
        mask_region[mask_region > 0] = 1
        tile_resized = cv2.resize(tile, (new_tile_w, new_tile_h), interpolation=cv2.INTER_AREA)

        top = new_tile_h // 2 - h // 2
        bottom = top + h
        left = new_tile_w // 2 - w // 2
        right = left + w
        tile_cropped = tile_resized[top:bottom, left:right]

        bgr_mean = np.mean(tile_cropped[mask_region>0], axis=0)
        distance = np.sum((part_mean - bgr_mean) ** 2)
        if distance < min_distance:
            min_distance = distance
            final_tile = tile_cropped  * mask_region[..., None]
            final_h = h
            final_w = w

    return x, y, final_w, final_h, final_tile

def create_superpixel_list_new(target_image, image_folder, region_size=50, ruler=50):
    slic_image = cv2.ximgproc.createSuperpixelSLIC(target_image, algorithm=cv2.ximgproc.SLIC,
                                            region_size=region_size, ruler=ruler)
    slic_image.iterate(10)
    labels = slic_image.getLabels()
    idx_num = len(np.unique(labels))

    output_image = np.zeros((target_image.shape[0], target_image.shape[1], 3), np.float64)
    tasks = []

    max_processes = os.cpu_count() // 2
    with ThreadPoolExecutor(max_workers=max_processes) as executor:
        for idx in tqdm(range(idx_num)):
            mask = np.zeros_like(labels, dtype=np.uint8)
            mask[labels == idx] = 255
            x, y, w, h = cv2.boundingRect(mask)
            if w == 0 or h == 0:
                continue
            tasks.append(executor.submit(process_superpixel, idx, target_image, labels, mask, x, y, w, h, image_folder))

        for future in tqdm(as_completed(tasks), total=len(tasks)):
            x, y, w, h, image_tile = future.result()
            output_image[y:y+h, x:x+w] += image_tile
            
    return output_image.astype(np.uint8)

def main(opt):
    target_image = cv2.imread(opt.target_root)

    output_image = create_superpixel_list_new(target_image, opt.image_pool, region_size=opt.region_size, ruler=opt.ruler)
    
    os.makedirs(opt.output_dir, exist_ok=True)
    cv2.imshow('fig', output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite(os.path.join(opt.output_dir, f'output.jpg'), output_image) 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_pool", type=str, help="The path of image pool.", default="./image_pool")
    parser.add_argument("--target_root", type=str, help="The path of target image.", default="./test.jpg")
    parser.add_argument("--output_dir", type=str, help="The path of output image.", default="./output")
    parser.add_argument("--region_size", type=int, help="Region size of superpixel.", default=60)
    parser.add_argument("--ruler", type=int, help="Region ruler of superpixel.", default=100)

    opt = parser.parse_args()
    main(opt)
