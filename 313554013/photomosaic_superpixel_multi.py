import os
import cv2
import numpy as np
import argparse
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

def load_images(folder_path, resize_width, resize_height):
    resize_image_list = []
    files = os.listdir(folder_path)  
    for i in tqdm(files):     
        image = cv2.imread(os.path.join(folder_path, i))  
        if image.shape[0]  > image.shape[1]:
            image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        image = cv2.resize(image, (resize_width, resize_height), interpolation=cv2.INTER_LINEAR)
        resize_image_list.append(image)
    return resize_image_list

def process_superpixel(idx, target_image, labels, loaded_images, mask, x, y, w, h, files, image_folder):
    part = target_image[labels == idx]
    part_mean = np.mean(part, axis=0)
    min_distance = 195075

    mask_region = mask[y:y+h, x:x+w]
    mask_region[mask_region > 0] = 1

    bgr_mean_list = []
    for i in loaded_images:
        i_resized = cv2.resize(i, (w, h), interpolation=cv2.INTER_AREA)
        i_masked = np.zeros_like(i_resized)
        for c in range(3):
            i_masked[:, :, c] = mask_region * i_resized[:, :, c]
        bgr_mean_list.append(np.mean(i_masked, axis=(1, 0)))

    image_idx = 0
    for i, k in enumerate(bgr_mean_list):
        distance = ((part_mean[0] - k[0]) ** 2 +
                    (part_mean[1] - k[1]) ** 2 +
                    (part_mean[2] - k[2]) ** 2)
        if distance < min_distance:
            min_distance = distance
            image_idx = i

    image_tile = files[image_idx]
    image_tile = cv2.imread(os.path.join(image_folder, image_tile))
    image_tile_resized = cv2.resize(image_tile, (w, h), interpolation=cv2.INTER_AREA)

    return x, y, w, h, mask_region, image_tile_resized

def create_superpixel_list_new(target_image, loaded_images, image_folder, files, region_size=50, ruler=50):
    slic_image = cv2.ximgproc.createSuperpixelSLIC(target_image, algorithm=cv2.ximgproc.SLIC,
                                            region_size=region_size, ruler=ruler)
    slic_image.iterate(10)
    labels = slic_image.getLabels()
    idx_num = len(np.unique(labels))

    output_image = np.zeros((target_image.shape[0], target_image.shape[1], 3), np.float64)
    tasks = []

    max_processes = os.cpu_count()
    with ThreadPoolExecutor(max_workers=max_processes) as executor:
        for idx in tqdm(range(idx_num)):
            mask = np.zeros_like(labels, dtype=np.uint8)
            mask[labels == idx] = 255
            x, y, w, h = cv2.boundingRect(mask)
            if w == 0 or h == 0:
                continue
            tasks.append(executor.submit(process_superpixel, idx, target_image, labels, loaded_images, mask, x, y, w, h, files, image_folder))

        for future in tqdm(as_completed(tasks), total=len(tasks)):
            x, y, w, h, mask_region, image_tile_resized = future.result()
            # for c in range(3):
            #     output_image[y:y+h, x:x+w, c] += mask_region * image_tile_resized[:, :, c]
            output_image[y:y+h, x:x+w] += mask_region[..., None] * image_tile_resized
            
    return output_image.astype(np.uint8)

def main(opt):
    target_image = cv2.imread(opt.target_root)
    w = target_image.shape[1]
    h = target_image.shape[0]
    target_image = cv2.resize(target_image, (w//2,h//2), interpolation=cv2.INTER_LINEAR)
    try:
        loaded_data = np.load(opt.npz_dir)
    except:
        resize_image_list = load_images(opt.image_pool, resize_width=150, resize_height=100)
        np.savez(opt.npz_dir, *resize_image_list)
        loaded_data = np.load(opt.npz_dir)

    loaded_images = [loaded_data[key] for key in loaded_data]
    
    files = os.listdir(opt.image_pool)
    output_image = create_superpixel_list_new(target_image, loaded_images, opt.image_pool,
                                                   files, region_size=opt.region_size, ruler=opt.ruler)
    
    os.makedirs(opt.output_dir, exist_ok=True)
    cv2.imshow('fig', output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite(os.path.join(opt.output_dir, 'output.jpg'), output_image) 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz_dir", type=str, help="The path where npz files are stored.", default="./img.npz")
    parser.add_argument("--image_pool", type=str, help="The path of image pool.", default="./image_pool")
    parser.add_argument("--target_root", type=str, help="The path of target image.", default="./test.jpg")
    parser.add_argument("--output_dir", type=str, help="The path of output image.", default="./output")
    parser.add_argument("--region_size", type=int, help="Region size of superpixel.", default=60)
    parser.add_argument("--ruler", type=int, help="Region ruler of superpixel.", default=80)

    opt = parser.parse_args()
    main(opt)