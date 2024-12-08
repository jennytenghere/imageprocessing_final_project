import argparse
import cv2
from functools import partial
from multiprocessing import Pool
import numpy as np
import os
from PIL import Image, ImageDraw, ImageChops
import random
import sys
import time
from tqdm import tqdm
import matplotlib.pyplot as plt

from concurrent.futures import ThreadPoolExecutor
# from colorsys import rgb_to_hsv, hsv_to_rgb

# rotate, resize and crop the image
def process_image(args):
    try:
        image_file, args, width, height, h0, h1, w0, w1 = args
        image = cv2.imread(os.path.join(args.image_folder, image_file))
        # rotate the image if it is vertical
        if image.shape[0]>image.shape[1]:
            image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        if image is None:
            raise ValueError(f"Error reading image: {os.path.join(args.image_folder, image_file)}")
        # The image will be cropped while keeping the center unchanged
        # e.g. all the photos from my smartphone is 4000*2252 -> --original_width 4000 --orignal_height 2250
        image = cv2.resize(image[h0:h1, w0:w1], (width, height), interpolation=cv2.INTER_LINEAR)
    except Exception as e:
        print(image_file)
    return image

# concate all the patches
def concat_tile_image(args, files, image_list, h0, h1, w0, w1):
    # the size of each patch
    height = int(args.original_height // args.concat_scale)# 160
    width = int(args.original_width // args.concat_scale)# 90
    # multithreading
    # resize, rotate and crop the images
    with Pool() as pool:
        processed_images = list(
            tqdm(
                pool.imap(
                    process_image,
                    [(files[idx], args, width, height, h0, h1, w0, w1) for idx in image_list]
                ),
                total=len(image_list)
            )
        )
    # concate the images
    concat_list = []
    for idx, image in enumerate(processed_images):
        # first image of the row
        if idx % args.piece_scale == 0:
            row_list = [image]
        # not the first image of the row
        else:
            row_list.append(image)
            # last image of the row
            if idx % args.piece_scale == (args.piece_scale - 1):
                row_img = np.hstack(row_list)
                concat_list.append(row_img)
        # last one
        if idx == len(processed_images) - 1 and idx % args.piece_scale < args.piece_scale - 1:
            n_blank = args.piece_scale - (len(processed_images) % args.piece_scale)
            row_list.extend([np.zeros((height, width, 3), dtype=np.uint8)] * n_blank)
            row_img = np.hstack(row_list)
            concat_list.append(row_img)
        if idx >= args.piece_scale * args.piece_scale:
            break
    # concate each row
    concat_img = np.vstack(concat_list)
    # save the output image
    cv2.imwrite(args.output_image_dir, concat_img)

    # calculate the loss between the input image and the output image
    adjust_height = int(args.original_height//args.piece_scale*args.piece_scale)
    adjust_width = int(args.original_width//args.piece_scale*args.piece_scale)
    concat_img_resize = cv2.resize(concat_img, (adjust_width, adjust_height))
    fitness = np.mean((target_image.astype(np.float32) - concat_img_resize.astype(np.float32)) ** 2)
    print("fitness:", fitness)
    print("Average fitness:", fitness / (target_image.shape[0] * target_image.shape[1] * 3))

# calculate the avg of each channel
def compute_bgr_mean(file_path, args):
    image = cv2.imread(file_path)
    # using LAB color space if --lab
    if args.lab:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    # rotate the image if it is vertical
    if image.shape[0]>image.shape[1]:
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    # The image will be cropped while keeping the center unchanged 
    width_center = image.shape[1]//2
    height_center = image.shape[0]//2
    image = image[height_center-args.original_height//2 : height_center-args.original_height//2+args.original_height,
                    width_center-args.original_width//2 :  width_center-args.original_width//2+args.original_width]
    # if the image doesn't exist
    if image is None:
        raise ValueError(f"Error reading image: {file_path}")
    # return the avg of each channel
    return np.mean(image, axis=(0, 1))

# calculate the four avg of each channel (divide the image into four)
def compute_bgr_mean_four(file_path, args):
    # same as compute_bgr_mean(), but using four mean values
    bgr_mean_list= []
    image = cv2.imread(file_path)
    if args.lab:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    if image.shape[0]>image.shape[1]:
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    width_center = image.shape[1]//2
    height_center = image.shape[0]//2
    image = image[height_center-args.original_height//2 : height_center-args.original_height//2+args.original_height,
                    width_center-args.original_width//2 :  width_center-args.original_width//2+args.original_width]
    
    h = args.original_height
    w = args.original_width
    h_half, w_half = h // 2, w // 2

    top_left = image[:h_half, :w_half]
    top_right = image[:h_half, w_half:]
    bottom_left = image[h_half:, :w_half]
    bottom_right = image[h_half:, w_half:]

    top_left_mean = np.mean(top_left, axis=(0, 1))
    top_right_mean = np.mean(top_right, axis=(0, 1))
    bottom_left_mean = np.mean(bottom_left, axis=(0, 1))
    bottom_right_mean = np.mean(bottom_right, axis=(0, 1))
    
    if image is None:
        raise ValueError(f"Error reading image: {file_path}")
    return [top_left_mean, top_right_mean, bottom_left_mean, bottom_right_mean]

# rotate, resize and crop the iamge
def resize_image(file_path, args):
    image = cv2.imread(file_path)
    if args.lab:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    if image.shape[0]>image.shape[1]:
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    width_center = image.shape[1]//2
    height_center = image.shape[0]//2
    image = image[height_center-args.original_height//2 : height_center-args.original_height//2+args.original_height,
                    width_center-args.original_width//2 :  width_center-args.original_width//2+args.original_width]
    image = cv2.resize(image, (args.original_width//args.piece_scale, args.original_height//args.piece_scale))
    return image

# baseline
# calculate the difference of the pixel values at corresponding positions within the patch
def diff_score(img1, img2):
    if (img1 is None) or (img2 is None):
        return np.inf
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    #diff_score = np.sum(np.sqrt(np.sum((img1 - img2)**2, axis=2)))
    diff_score = np.sum((img1 - img2) ** 2) 
    return diff_score

# multithreading
def compute_distance_for_images(part, loaded_image_list):
    return np.array([diff_score(part, img) for img in loaded_image_list])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for final project.")
    # common args
    parser.add_argument("--original_width", type=int, default=4000, help="The width of images in the dataset. (automatically cropped)")
    parser.add_argument("--original_height", type=int, default=2250, help="The width of images in the dataset. (automatically cropped).")
    parser.add_argument("--target_image_dir", type=str, default='./target/20230824_115144.jpg', help="The path of the target image.")
    parser.add_argument("--output_image_dir", type=str,  default='./output.png', help="The path of the output image.")
    # args of photo masaic art 
    parser.add_argument("--piece_scale", type=int, default=50, help="The number of divisions (of one side).")
    parser.add_argument("--concat_scale", type=int, default=25, help="The scaling factor to reduce the image when concatenating.")
    parser.add_argument("--image_folder", type=str, default='./japan/', help="The directory of the dataset folder.")
    parser.add_argument("--use_npz", action="store_true", default=False, help="Use npz file.  (This can speed up processing if the same parameters have been used before. Same database, original_width, original_height, piece_scale, four_mean, and lab)") # --use_npz
    parser.add_argument("--npz_dir", type=str, default='bgr_mean_list.npz', help="The path of the npz file.")
    parser.add_argument("--four_mean", action="store_true", default=False, help="Use four mean values to calculate similarity. (more precise but requires more time)") # --four_mean
    parser.add_argument("--lab", action="store_true", default=False, help="Use LAB color space instead of RGB color space. (without --lab, using RGB color space)")
    parser.add_argument("--baseline", action="store_true", default=False, help="Baseline.")
    # args of random circle/square art 
    parser.add_argument("--circle", action="store_true", default=False, help="Construct the target image using random dots.")
    parser.add_argument("--square", action="store_true", default=False, help="Construct the target image using random squares.")
    parser.add_argument("--min_radius_size", type=int, default=30, help="Min radius of the circle or square. ( min_radius_size ~ max(min_radius_size, max_radius_size-iteration//alpha) )")
    parser.add_argument("--max_radius_size", type=int, default=500, help="Max radius of the circle or square. ( min_radius_size ~ max(min_radius_size, max_radius_size-iteration//alpha) )")
    parser.add_argument("--alpha", type=int, default=2, help="The parameter to control the maximum radius reduction per iteration. ( min_radius_size ~ max(min_radius_size, max_radius_size-iteration//alpha) )")
    parser.add_argument("--iteration", type=int, default=5000, help="one iteration = one circle/square")
    args = parser.parse_args()
    
    target_image = cv2.imread(args.target_image_dir)
    # center of the image
    height_center = target_image.shape[0]//2
    width_center = target_image.shape[1]//2
    # crop the image
    h0 = height_center-args.original_height//2
    h1 =  height_center-args.original_height//2+args.original_height
    w0 = width_center-args.original_width//2
    w1 = width_center-args.original_width//2+args.original_width
    target_image = target_image[h0:h1, w0:w1]
    
    if args.baseline:
        # print("baseline----------------------------------------")
        # while True:
        #     if args.original_width % args.piece_scale == 0 and args.original_height % args.piece_scale == 0 and \
        #         args.original_width % args.concat_scale == 0 and args.original_height % args.concat_scale == 0:
        #         break
        #     else:
        #         print("Bad scale number. They need to be the common factors of original_width and original_height.")
        #         sys.exit(1)

        bgr_mean_list = []
        files = [os.path.join(args.image_folder, f) for f in os.listdir(args.image_folder)]
        # multithreading
        # resize the images in the database as the size of the patches
        if not args.use_npz:
            resize_image_partial = partial(resize_image, args=args)
            with Pool() as pool:
                resize_image_list = list(tqdm(pool.imap(resize_image_partial, files), total=len(files)))
            np.savez(args.npz_dir, *resize_image_list)
        # the size of each patch
        resize_height = int(args.original_height // args.piece_scale)
        resize_width = int(args.original_width // args.piece_scale)
        # load the npz file (resized images)
        loaded_data = np.load(args.npz_dir)
        loaded_image_list = [loaded_data[key] for key in loaded_data]
        # the buffer of selected images (index)
        selected_image = []
        # adjust the size of the target image
        adjust_height = int(args.original_height//args.piece_scale*args.piece_scale)
        adjust_width = int(args.original_width//args.piece_scale*args.piece_scale)
        target_image = cv2.resize(target_image, (adjust_width, adjust_height))
        # find the best image for each patch
        for y in tqdm(range(0, adjust_height, resize_height)):
            for x in range(0, adjust_width, resize_width):
                part = target_image[y:y + resize_height, x:x + resize_width]
                if args.lab:
                        part = cv2.cvtColor(part, cv2.COLOR_BGR2LAB)
                distance = np.array([diff_score(part, s) for s in loaded_image_list])
                selected_image.append(np.argmin(distance))
        files = os.listdir(args.image_folder)
        # concat all the selected images
        concat_tile_image(args, files, selected_image, h0, h1, w0, w1)
        
    elif args.circle:
        # read the image and crop as args.original_width and args.original_height
        img = Image.open(args.target_image_dir)
        img = img.crop((w0, h0, w1, h1))
        # initial white image for saving the final result
        halftone_img = Image.new("RGB", img.size, "white")
        draw = ImageDraw.Draw(halftone_img)
        original_np = np.array(img, dtype=np.int32)
        halftone_np = np.array(halftone_img, dtype=np.int32)
        # the shape radius of each radius
        radius_values = [random.randint(args.min_radius_size, max(args.min_radius_size, args.max_radius_size-i//args.alpha)) for i in range(args.iteration)]
        x_values = [random.randint(0, img.width - 1) for i in range(args.iteration)]
        y_values = [random.randint(0, img.height - 1) for i in range(args.iteration) ]
        
        for i in tqdm(range(args.iteration)):
            x = x_values[i]
            y = y_values[i]
            radius = radius_values[i]
            # the region of where the circle will be added
            left_up = (max(x - radius, 0), max(y - radius, 0))
            right_down = (min(x + radius, img.width - 1), min(y + radius, img.height - 1))
            x_min, y_min = left_up
            x_max, y_max = right_down
            region_original = original_np[y_min:y_max, x_min:x_max]
            region_halftone = halftone_np[y_min:y_max, x_min:x_max]
            # the difference between the target image and current image
            origin_difference = np.mean(np.abs(region_original - region_halftone))
            # tmp numpy for simulation
            tmp_np = halftone_np.copy()
            # get the color
            color = original_np[y, x]
            # mask of the circle
            rr, cc = np.ogrid[:y_max-y_min, :x_max-x_min]
            mask = (rr - (y - y_min))**2 + (cc - (x - x_min))**2 <= radius**2
            # draw the circle on the tmp numpy
            tmp_np[y_min:y_max, x_min:x_max][mask] = color  
            # calculate the different after adding the circle
            new_difference = np.mean(np.abs(region_original - tmp_np[y_min:y_max, x_min:x_max]))

            # if the different is smaller, then update the result
            if new_difference < origin_difference:
                halftone_np[y_min:y_max, x_min:x_max][mask] = color
                draw.ellipse([left_up, right_down], fill=tuple(color), outline=tuple(color))

        # save the result
        final_img = Image.fromarray(halftone_np.astype(np.uint8))
        final_img.save(args.output_image_dir)
    elif args.square:
        # read the image and crop as args.original_width and args.original_height
        img = Image.open(args.target_image_dir)
        img = img.crop((w0, h0, w1, h1))
        # initial white image for saving the final result
        halftone_img = Image.new("RGB", img.size, "white")
        draw = ImageDraw.Draw(halftone_img)
        original_np = np.array(img, dtype=np.int32)
        halftone_np = np.array(halftone_img, dtype=np.int32)
        # the shape radius of each radius
        radius_values = [random.randint(args.min_radius_size, max(args.min_radius_size, args.max_radius_size-i//args.alpha)) for i in range(args.iteration)]
        x_values = [random.randint(0, img.width - 1) for i in range(args.iteration)]
        y_values = [random.randint(0, img.height - 1) for i in range(args.iteration) ]
        
        for i in tqdm(range(args.iteration)):
            x = x_values[i]
            y = y_values[i]
            radius = radius_values[i]
            # the region of where the square will be added
            left_up = (max(x - radius, 0), max(y - radius, 0))
            right_down = (min(x + radius, img.width - 1), min(y + radius, img.height - 1))
            x_min, y_min = left_up
            x_max, y_max = right_down
            region_original = original_np[y_min:y_max, x_min:x_max]
            region_halftone = halftone_np[y_min:y_max, x_min:x_max]
            # the difference between the target image and current image
            origin_difference = np.mean(np.abs(region_original - region_halftone))
            # tmp numpy for simulation
            tmp_np = halftone_np.copy()
            # get the color
            color = original_np[y, x]
            # mask of the square
            rr, cc = np.ogrid[:y_max-y_min, :x_max-x_min]
            # mask = (rr - (y - y_min))**2 + (cc - (x - x_min))**2 <= radius**2
            # draw the square on the tmp numpy
            tmp_np[y_min:y_max, x_min:x_max] = color  
            # calculate the different after adding the square
            new_difference = np.mean(np.abs(region_original - tmp_np[y_min:y_max, x_min:x_max]))

            # if the different is smaller, then update the result
            if new_difference < origin_difference:
                halftone_np[y_min:y_max, x_min:x_max] = color
                draw.rectangle([left_up, right_down], fill=tuple(color), outline=tuple(color))

        # save the result
        final_img = Image.fromarray(halftone_np.astype(np.uint8))
        final_img.save(args.output_image_dir)
    else:
        # while True:
        #     if args.original_width % args.piece_scale == 0 and args.original_height % args.piece_scale == 0 and \
        #         args.original_width % args.concat_scale == 0 and args.original_height % args.concat_scale == 0:
        #         break
        #     else:
        #         print("Bad scale number. They need to be the common factors of original_width and original_height.")
        #         sys.exit(1)

        # the adjusted size of the target image
        adjust_height = int(args.original_height//args.piece_scale*args.piece_scale)
        adjust_width = int(args.original_width//args.piece_scale*args.piece_scale)
        # resize the image
        target_image = cv2.resize(target_image, (adjust_width, adjust_height))

        bgr_mean_list = []
        files = [os.path.join(args.image_folder, f) for f in os.listdir(args.image_folder)]
        
        # multithreading
        # calculate the avg value of each channel for each image
        if not args.use_npz:
            # if --four_mean, devide the image into four and calculate four avg values
            if args.four_mean:
                compute_bgr_mean_partial = partial(compute_bgr_mean_four, args=args)
                with Pool() as pool:
                    bgr_mean_list = list(tqdm(pool.imap(compute_bgr_mean_partial, files), total=len(files)))
                np.savez(args.npz_dir, *bgr_mean_list)
            # using entire image
            else:
                compute_bgr_mean_partial = partial(compute_bgr_mean, args=args)
                with Pool() as pool:
                    bgr_mean_list = list(tqdm(pool.imap(compute_bgr_mean_partial, files), total=len(files)))
                np.savez(args.npz_dir, *bgr_mean_list)

        # the size of each patch
        resize_height = int(args.original_height // args.piece_scale)
        resize_width = int(args.original_width // args.piece_scale)
        # load the npz file (avg values of each image)
        loaded_data = np.load(args.npz_dir)
        loaded_bgr_mean_list = [loaded_data[key] for key in loaded_data]
        # buffer for storing selected image's index
        selected_image = []

        # if --four_mean
        if args.four_mean:
            # for each patch
            for y in tqdm(range(0, adjust_height, resize_height)):
                for x in range(0, adjust_width, resize_width):
                    part = target_image[y:y+resize_height, x:x+resize_width]
                    if args.lab:
                        part = cv2.cvtColor(part, cv2.COLOR_BGR2LAB)
                    h, w, _ = part.shape
                    h_half, w_half = h // 2, w // 2
                    top_left = part[:h_half, :w_half]
                    top_right = part[:h_half, w_half:]
                    bottom_left = part[h_half:, :w_half]
                    bottom_right = part[h_half:, w_half:]

                    top_left_mean = np.mean(top_left, axis=(0, 1))
                    top_right_mean = np.mean(top_right, axis=(0, 1))
                    bottom_left_mean = np.mean(bottom_left, axis=(0, 1))
                    bottom_right_mean = np.mean(bottom_right, axis=(0, 1))

                    # calculate the loss batween the target image's patch and all the images in the database
                    min_distance = 195075
                    for i, file_name in enumerate(files):  
                        k = loaded_bgr_mean_list[i]
                        distance = ((top_left_mean[0]-k[0][0])**2+(top_left_mean[1]-k[0][1])**2+(top_left_mean[2]-k[0][2])**2+
                                    (top_right_mean[0]-k[1][0])**2+(top_right_mean[1]-k[1][1])**2+(top_right_mean[2]-k[1][2])**2+
                                    (bottom_left_mean[0]-k[2][0])**2+(bottom_left_mean[1]-k[2][1])**2+(bottom_left_mean[2]-k[2][2])**2+
                                    (bottom_right_mean[0]-k[3][0])**2+(bottom_right_mean[1]-k[3][1])**2+(bottom_right_mean[2]-k[3][2])**2)
                        if distance<min_distance:
                            min_distance = distance
                            index = i 
                    selected_image.append(index) 
        # if --four_mean=False, calculate the avg of the entire image
        else:
            for y in tqdm(range(0, adjust_height, resize_height)):
                for x in range(0, adjust_width, resize_width):
                    part = target_image[y:y+resize_height, x:x+resize_width]
                    if args.lab:
                        part = cv2.cvtColor(part, cv2.COLOR_BGR2LAB)
                    bgr = np.mean(part, axis=(1,0))
                    min_distance = 195075
                    for i, file_name in enumerate(files):  
                        k = loaded_bgr_mean_list[i]
                        # calculate the loss between
                        distance = ((bgr[0]-k[0])**2+
                                    (bgr[1]-k[1])**2+
                                    (bgr[2]-k[2])**2)
                        if distance<min_distance:
                            min_distance = distance
                            index = i 
                    selected_image.append(index) 

        # concate and save the output image
        files = os.listdir(args.image_folder)
        concat_tile_image(args, files, selected_image, h0, h1, w0, w1)

