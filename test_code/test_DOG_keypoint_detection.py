import numpy as np
import cv2 as cv
from pathlib import Path
from utiles import file_related

if __name__ == '__main__':
   ori_img_path = Path(r'H:\code\python\net_detection_and_tracking\data_0.1\wj-original_image_diff_moving_direction_29241030\small_test_DOG')
   out_put_path = Path(r'H:\code\python\net_detection_and_tracking\test_output\test_DOG')
   img_files = file_related.get_filenames_of_path(ori_img_path)
   for a_img_file in img_files:
      img_name = a_img_file.stem
      cur_img = cv.imread(str(a_img_file))
      gray = cv.cvtColor(cur_img, cv.COLOR_BGR2GRAY)
      sift = cv.SIFT_create()
      kp = sift.detect(gray, None)
      display_img = cv.drawKeypoints(gray, kp, cur_img)
      cv.imwrite(f'{str(out_put_path)}/{img_name}.jpg', display_img)
      pass

