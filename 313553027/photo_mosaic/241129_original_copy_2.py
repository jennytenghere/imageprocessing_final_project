import argparse
import cv2
from functools import partial
import numpy as np
from tqdm import tqdm
import os
import sys
import time
from PIL import Image
from multiprocessing import Pool

def process_image(args):
    try:
        image_file, args, width, height, h0, h1, w0, w1 = args
        image = cv2.imread(os.path.join(args.image_folder, image_file))
        if image.shape[0]>image.shape[1]:
            image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        # print(image.shape)
        if image is None:
            raise ValueError(f"Error reading image: {os.path.join(args.image_folder, image_file)}")
        image = cv2.resize(image[h0:h1, w0:w1], (width, height), interpolation=cv2.INTER_LINEAR)
    except Exception as e:
        print(image_file)
    
    return image


# def process_image(args):
#     image_file, args, width, height = args
#     try:
#         image = Image.open(args.image_folder + image_file)
#         # width, height
#         width_center = image.size[0]//2 # 1151 
#         height_center = image.size[1]//2 #2000
#         image = image.crop((width_center-args.original_width//2, height_center-args.original_height//2, 
#                             width_center-args.original_width//2+args.original_width, height_center+args.original_height+args.original_height)) 
#         image = image.resize((width, height), Image.BILINEAR)
#         image = np.array(image)
#         return image
    
#     except Exception as e:
#         raise ValueError(f"Error processing image: {args.image_folder + image_file}, Error: {e}")


def concat_tile_image(args, files, image_list, h0, h1, w0, w1):
    height = int(args.original_height / args.concat_scale)
    width = int(args.original_width / args.concat_scale)

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
    concat_list = []
    for idx, image in enumerate(processed_images):
        if idx % args.piece_scale == 0:
            row_list = [image]
        else:
            row_list.append(image)
            if idx % args.piece_scale == (args.piece_scale - 1):
                row_img = np.hstack(row_list)
                concat_list.append(row_img)
        if idx == len(processed_images) - 1 and idx % args.piece_scale < args.piece_scale - 1:
            n_blank = args.piece_scale - (len(processed_images) % args.piece_scale)
            row_list.extend([np.zeros((height, width, 3), dtype=np.uint8)] * n_blank)
            row_img = np.hstack(row_list)
            concat_list.append(row_img)
        if idx >= args.piece_scale * args.piece_scale:
            break

    concat_img = np.vstack(concat_list)
    cv2.imwrite(args.output_image_dir, concat_img)

    concat_img_resize = cv2.resize(concat_img, (args.original_width, args.original_height))
    fitness = np.sum(np.abs(target_image.astype(np.int32) - concat_img_resize.astype(np.int32)))
    print("fitness:", fitness)
    print("Average fitness:", fitness / (target_image.shape[0] * target_image.shape[1] * 3))

# def concat_tile_image(args, files, image_list):
#     height = int(args.original_height / args.concat_scale)
#     width = int(args.original_width / args.concat_scale)
#     concat_list = []

#     for idx, image_index in enumerate(tqdm(image_list)):
#         # print(idx)
#         image = files[image_index]
#         image = cv2.imread(args.image_folder+image)
#         # print(image.shape)
#         image = cv2.resize(image[1:2251, 0:4000], (width, height), interpolation=cv2.INTER_LINEAR)
#         # print(image.shape)

#         if idx % args.piece_scale == 0:
#             row_list = [image]
#         else:
#             row_list = row_list + [image]
#             if idx % args.piece_scale == (args.piece_scale -1):
#                 row_img = np.hstack(row_list)
#                 # cv2.imshow('a', row_img)
#                 # cv2.waitKey(0)
#                 # cv2.destroyAllWindows()
#                 concat_list = concat_list +[row_img]
#         if idx == len(image_list) - 1:
#             if idx % args.piece_scale < args.piece_scale -1:
#                 n_blank = args.piece_scale - (len(image_list) % args.piece_scale)
#                 row_list = row_list + [np.zeros( (width, height, 3) )] * n_blank
#                 row_img = np.hstack(row_list)
#                 concat_list = concat_list + [row_img]            
#         if idx >= args.piece_scale * args.piece_scale:
#             break
#     concat_img = np.vstack(concat_list)
#     cv2.imwrite(args.output_image_dir, concat_img)
    
#     concat_img_resize = cv2.resize(concat_img, (4000, 2250))
#     fitness = np.sum(np.abs(target_image.astype(np.int32) - concat_img_resize.astype(np.int32)))
#     print("fitness:",fitness)
#     print("Average fitness:", fitness / (target_image.shape[0]*target_image.shape[1]*3))

def compute_bgr_mean(file_path, args):
    image = cv2.imread(file_path)
    if image.shape[0]>image.shape[1]:
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    width_center = image.shape[1]//2 # 1151
    height_center = image.shape[0]//2 #2000
    image = image[height_center-args.original_height//2 : height_center-args.original_height//2+args.original_height,
                    width_center-args.original_width//2 :  width_center-args.original_width//2+args.original_width]
    if image is None:
        raise ValueError(f"Error reading image: {file_path}")
    return np.mean(image, axis=(0, 1))

def compute_bgr_mean_four(file_path, args):
    bgr_mean_list= []
    image = cv2.imread(file_path)
    if image.shape[0]>image.shape[1]:
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    width_center = image.shape[1]//2 # 1151
    height_center = image.shape[0]//2 #2000
    image = image[height_center-args.original_height//2 : height_center-args.original_height//2+args.original_height,
                    width_center-args.original_width//2 :  width_center-args.original_width//2+args.original_width]
    
    h = args.original_height
    w = args.original_width
    h_half, w_half = h // 2, w // 2
    # print(image.shape, h, w, h_half, w_half, file_path)
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

# def process_block(args):
#     y, x, resize_height, resize_width, target_image, loaded_bgr_mean_array = args
#     part = target_image[y:y+resize_height, x:x+resize_width]
#     bgr = np.mean(part, axis=(0, 1))
#     distances = np.sum((loaded_bgr_mean_array - bgr)**2, axis=1)
#     return np.argmin(distances)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for photo masaic art.")
    parser.add_argument("--piece_scale", type=int, default=50, help="How much should the images be scaled down.")
    parser.add_argument("--concat_scale", type=int, default=25, help="How much should the images be scaled down during concatenation.")
    parser.add_argument("--original_width", type=int, default=4000, help="Original image's width.")
    parser.add_argument("--original_height", type=int, default=2250, help="Original image's height.")
    parser.add_argument("--image_folder", type=str, default='./japan/', help="Image folder.")
    parser.add_argument("--target_image_dir", type=str, default='./target/20230824_115144.jpg', help="The directory of the target image")
    parser.add_argument("--use_npz", action="store_true", default=False, help="Use npz file.") # --use_npz
    parser.add_argument("--npz_dir", type=str, default='bgr_mean_list.npz', help="The directory of the npz file")
    parser.add_argument("--output_image_dir", type=str,  required=True, help="The directory of the output image")
    parser.add_argument("--four_mean", action="store_true", default=False, help="Use four mean values to calculate similarity.")

    args = parser.parse_args()
    
    while True:
        if args.original_width % args.piece_scale == 0 and args.original_height % args.piece_scale == 0 and \
            args.original_width % args.concat_scale == 0 and args.original_height % args.concat_scale == 0:
            break
        else:
            print("Bad scale number")
            sys.exit(1)

    target_image = cv2.imread(args.target_image_dir)
    # print(target_image.shape)
    
    height_center = target_image.shape[0]//2 # 1151
    width_center = target_image.shape[1]//2 # 2000

    h0 = height_center-args.original_height//2 
    h1 =  height_center-args.original_height//2+args.original_height
    w0 = width_center-args.original_width//2
    w1 = width_center-args.original_width//2+args.original_width
    target_image = target_image[h0:h1, w0:w1]
    adjust_height = args.original_height//args.piece_scale*args.piece_scale
    adjust_width = args.original_width//args.piece_scale*args.piece_scale
    # target_image = target_image.resize()

    bgr_mean_list = []
    folder_path = './japan/'
    files = os.listdir(folder_path)
    # 10 min
    # if not args.use_npz:
    #     for i in tqdm(files):
    #         start_time = time.time()
    #         with Image.open(folder_path+i) as image:
    #         #image = cv2.imread(folder_path+i)
    #             read_time = time.time() - start_time
    #             start_time = time.time()
    #             mean_bgr = np.mean(image, axis=(1,0))
    #             mean_time = time.time() - start_time
    #             start_time = time.time()
    #             bgr_mean_list.append(mean_bgr)
    #             append_time = time.time() - start_time
    #             print(read_time, mean_time, append_time)
    #     np.savez(args.npz_dir, *bgr_mean_list)
    files = [os.path.join(folder_path, f) for f in os.listdir(folder_path)]
    if not args.use_npz:
        compute_bgr_mean_partial = partial(compute_bgr_mean_four, args=args)
        with Pool() as pool:
            bgr_mean_list = list(tqdm(pool.imap(compute_bgr_mean_partial, files), total=len(files)))
        np.savez(args.npz_dir, *bgr_mean_list)
    # else:
    #     compute_bgr_mean_partial = partial(compute_bgr_mean, args=args)
    #     with Pool() as pool:
    #         bgr_mean_list = list(tqdm(pool.imap(compute_bgr_mean_partial, files), total=len(files)))
    

    resize_height = int(args.original_height / args.piece_scale)
    resize_width = int(args.original_width / args.piece_scale)

    # replace = np.zeros((original_height, original_width, 3), np.uint8)

    loaded_data = np.load(args.npz_dir)
    loaded_bgr_mean_list = [loaded_data[key] for key in loaded_data]

    selected_image = []
    
    loaded_bgr_mean_array = np.array(loaded_bgr_mean_list)  # Shape: (m, 3)

    # for y in tqdm(range(0, args.original_height, resize_height)):
    #     for x in range(0, args.original_width, resize_width):
    #         # 計算當前區塊的 BGR 平均值
    #         part = target_image[y:y+resize_height, x:x+resize_width]
    #         bgr = np.mean(part, axis=(0, 1))  # Shape: (3,)
            
    #         # 計算與所有候選圖片的歐氏距離（向量化操作）
    #         distances = np.sum((loaded_bgr_mean_array - bgr)**2, axis=1)  # Shape: (m,)
            
    #         # 找到最小距離的索引
    #         index = np.argmin(distances)
    #         selected_image.append(index)
    
    # block_args = [
    #     (y, x, resize_height, resize_width, target_image, loaded_bgr_mean_array)
    #     for y in range(0, args.original_height, resize_height)
    #     for x in range(0, args.original_width, resize_width)
    # ]
    # print("here")
    # with Pool() as pool:
    #     selected_image = list(tqdm(pool.imap(process_block, block_args), total=len(block_args)))
    # print("loaded_bgr_mean_list", loaded_bgr_mean_list)
    # print("loaded_bgr_mean_list[0]", loaded_bgr_mean_list[0])
    if args.four_mean:
        for y in tqdm(range(0, args.original_height, resize_height)):
            for x in range(0, args.original_width, resize_width):
                part = target_image[y:y+resize_height, x:x+resize_width]
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
                
    else:
        for y in tqdm(range(0, args.original_height, resize_height)):
            for x in range(0, args.original_width, resize_width):
                part = target_image[y:y+resize_height, x:x+resize_width]
                bgr = np.mean(part, axis=(1,0))
                min_distance = 195075
                for i, file_name in enumerate(files):  
                    k = loaded_bgr_mean_list[i]
                    distance = ((bgr[0]-k[0])**2+
                                (bgr[1]-k[1])**2+
                                (bgr[2]-k[2])**2)
                    if distance<min_distance:
                        min_distance = distance
                        index = i 
                selected_image.append(index) 
    
    
    #         # replace[y:y+resize_height, x:x+resize_width] = choice

    files = os.listdir(args.image_folder)
    concat_tile_image(args, files, selected_image, h0, h1, w0, w1)

