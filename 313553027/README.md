# imageprocessing_final_project
## Final result: 241208_photomosaic.py
### Command line example:
1. Photomosaic art (mine):  
241208_photomosaic.py --original_width 4000 --original_height 2250 --target_image_dir './target/20230824_115144.jpg' --output_image_dir './output.png' --piece_scale 50 --concat_scale 25 --image_folder './japan/' --npz_dir 'bgr_mean_list.npz'
2. Photomosaic art (not the first time):
   If the databse, --original_width, --original_height, --piece_scale, --four_mean, --lab, and --baseline are not changed.  
   241208_photomosaic.py --original_width 4000 --original_height 2250 --target_image_dir './target/20230824_115144.jpg' --output_image_dir './output.png' --piece_scale 50 --concat_scale 25 --image_folder './japan/' --npz_dir 'bgr_mean_list.npz' --use_npz
3. Photomosaic art (baseline):  
241208_photomosaic.py --original_width 4000 --original_height 2250 --target_image_dir './target/20230824_115144.jpg' --output_image_dir './output.png' --piece_scale 50 --concat_scale 25 --image_folder './japan/' --npz_dir 'bgr_mean_list.npz' --baseline
4. Photomosaic art (divide the image into four then calculate the avg pixel values):  
241208_photomosaic.py --original_width 4000 --original_height 2250 --target_image_dir './target/20230824_115144.jpg' --output_image_dir './output.png' --piece_scale 50 --concat_scale 25 --image_folder './japan/' --npz_dir 'bgr_mean_list.npz' --four_mean
5. Photomosaic art (use LAB color space instead of RGB color space):  
241208_photomosaic.py --original_width 4000 --original_height 2250 --target_image_dir './target/20230824_115144.jpg' --output_image_dir './output.png' --piece_scale 50 --concat_scale 25 --image_folder './japan/' --npz_dir 'bgr_mean_list.npz' --lab
6. Random circle:  
241208_photomosaic.py --original_width 4000 --original_height 2250 --circle --min_radius_size 30 --max_radius_size 500 --alpha 2 --iteration 5000
7. Random square:  
241208_photomosaic.py --original_width 4000 --original_height 2250 --square --min_radius_size 30 --max_radius_size 500 --alpha 2 --iteration 5000
