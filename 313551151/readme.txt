我不小心做出了以fitness作為選擇照片的程式（不是比對最接近的平均值，而是計算所有小照片裡，每個像素色差加起來最低，作為選擇的照片），並且會用到cuda
packing.py
Usage:python packing.py -d <image_folder_dir> -w <target_width> -h <target_height>
功能：resize image_folder_dir 裡面的每張照片至target_height*target_width，並且存為data_{len(images)}_{args.height}_{args.width}.pt


photomosiac_result_map.py
Usage:python photomosiac_result_map.py --tgt_pt A.pt --used_pt B.pt --n_row Y --ncol X --method rgb_fitness --metric rbg_fitness --exhibit

功能：plot出photomosiac的結果比對圖（原圖、拼貼圖、灰階loss圖，越亮loss越大），由n_col跟n_row張小照片拼貼而成的，拼貼尋找的方法是method(預設為rbg_fitness)，loss的計算方式是metric(預設為rbg_fitness)，如果要顯示比對圖，記得在後面加--exhibit
A.pt:你想拼出來的圖的pt（建議先把你要的圖放在一個資料夾，用packing.py生成）
B.pt:小照片的pt（可以用packing.py或是跑cifar10_packing.py生成）
rgb_fitness:計算rbg值
