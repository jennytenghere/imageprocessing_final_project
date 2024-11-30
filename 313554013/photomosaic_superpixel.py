import os
import cv2
import numpy as np
from tqdm import tqdm

def create_superpixel_list(target_image, bgr_mean_list, region_size=50, ruler=50):
    slic_image = cv2.ximgproc.createSuperpixelSLIC(target_image, algorithm=cv2.ximgproc.SLIC,
                                            region_size=region_size, ruler=ruler)
    slic_image.iterate(10)
    labels = slic_image.getLabels()
    idx_num = len(np.unique(labels))

    selected_image = []
    for idx in tqdm(range(idx_num)):
        part = target_image[labels==idx]
        bgr = np.mean(part, axis=0)
        min_distance = 195075
        for i in range(len(bgr_mean_list)):
            k = bgr_mean_list[i]
            distance = ((bgr[0]-k[0])**2+
                        (bgr[1]-k[1])**2+
                        (bgr[2]-k[2])**2)
            if distance<min_distance:
                min_distance = distance
                image_idx = i 
        selected_image.append(image_idx) 
    return labels, selected_image

def concat_tile_image(files, labels, image_list, image_folder, WIDTH = 4000, HEIGHT=2250):
    output_image = np.zeros((HEIGHT, WIDTH, 3), np.float64)
    for idx, image_index in tqdm(enumerate(image_list)):
        image = files[image_index]
        image = cv2.imread(os.path.join(image_folder,image))
        
        mask = np.zeros_like(labels, dtype=np.uint8)
        mask[labels==idx]=255
        x, y, w, h = cv2.boundingRect(mask)
        if w == 0 or h == 0:
            continue

        image_resized = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
        for c in range(3):
            mask_region = mask[y:y+h, x:x+w]
            mask_region[mask_region>0]=1
            output_image[y:y+h, x:x+w, c] += mask_region * (image_resized[:, :, c])

    # output_image = cv2.cvtColor(output_image.astype(np.uint8), cv2.COLOR_BGR2RGB)
    return output_image.astype(np.uint8)


def main():
    image_folder = '/Users/rightpunch/Desktop/image_pool'
    target_image_dir = '/Users/rightpunch/Desktop/相片/_DSC4580.jpg'

    target_image = cv2.imread(target_image_dir)
    target_image = cv2.pyrDown(target_image)
    original_h, original_w = target_image.shape[:2]

    loaded_data = np.load('IP_HW/Final/resize_50.npz')
    loaded_images = [loaded_data[key] for key in loaded_data]

    bgr_mean_list = []
    for i in tqdm(loaded_images):
        bgr_mean_list.append(np.mean(i, axis=(1,0)))

    files = os.listdir(image_folder)
    labels, selected_image = create_superpixel_list(target_image, bgr_mean_list, region_size=30, ruler=50)
    output_image = concat_tile_image(files, labels, selected_image, image_folder, original_w, original_h) 
    cv2.imshow('fig', output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite('replace.jpg', output_image) 

if __name__ == "__main__":
    main()