import cv2 as cv
from utiles import file_related
from pathlib import Path
import numpy as np
image_file_path = Path(r'H:\code\python\net_detection_and_tracking\test_harris_img')
out_put_file_path = Path(r'H:\code\python\net_detection_and_tracking\test_output\harris_corner_detector')
if __name__ == '__main__':
    origin_img_files = file_related.get_filenames_of_path(image_file_path)
    for cur_img_file in origin_img_files:
        img_name = cur_img_file.stem
        img = cv.imread(str(cur_img_file))
        gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        gray_img = np.float32(gray_img)
        ks = np.linspace(0.01,0.15,10)
        for a_block_size in range(60,70):
            for a_k_size in range(11, 31, 2):
                for a_k in ks:
                    dst = cv.cornerHarris(gray_img, a_block_size, a_k_size, a_k)
                    img[dst > 0.005 * dst.max()] = [0, 0, 255]
                    cv.imwrite(f'{str(out_put_file_path)}\{img_name}_{a_block_size}_{a_k_size}_{a_k}.jpg', img)
    pass