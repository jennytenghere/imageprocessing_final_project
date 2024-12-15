# imageprocessing_final_project
## Final result: 241208_photomosaic.py  
### 一些影片中來不及講的東西:  
有試過多種loss function，但都無法明確數值化視覺效果。  
很多視覺效果比較好的圖反而loss比較高。  
常見的找相似圖片的作法是計算每個pixel位置之間的差異，  
但我用少張數(約5000張)圖片的實驗結果顯示，用每一張圖的平均pixel value算相似度，做出來的圖片效果較好。  
原因可能是因為肉眼在看的時候不會這麼在意細節，比較接近看局部的平均顏色。  
在資料數少的情況下，色彩的分布多樣性不夠多所以用常見的作法效果會變糟。  
將圖片切成4份，每張圖用4個平均值算相似度是折衷的作法，可以更符合邊緣、顏色變化(細節)。  

用多個圓點或方塊拼貼出圓圖的項目有做幾點改良:
1. cropping要貼上圓點或方塊的區域，計算加上這個圓點或方塊後loss會不會降低，有降低才加進圖中。避免視覺效果下降。
2. 用alpha控制越後面的iteration，隨機半徑的最大值越小。使可以用更少的iteration達到相同的效果。
3. 每個iteration的隨機半徑用一次random function生成，不要每個iteration都呼叫。
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
