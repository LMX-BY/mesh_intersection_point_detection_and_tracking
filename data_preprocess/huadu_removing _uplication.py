import cv2 as cv
import pathlib

import numpy as np

from utiles import file_related

ori_img_file_path = pathlib.Path(r'G:\dataSet\huadu_localization_pool\video\camera_ji02')
output_image_path = r'G:\dataSet\huadu_localization_pool\video\camera_ji02_du_removed'


def get_file_id(file_name):
    test = float(file_name.stem)
    return float(file_name.stem)


if __name__ == '__main__':
    ori_image_files = file_related.get_filenames_of_path(ori_img_file_path)
    sorted_ori_image_files = sorted(ori_image_files, key=get_file_id)
    cur_base_img = None
    cur_base_img_file = None
    idiv_img_count = 0
    d_threshold = 15
    pixcel = 680 * 320
    for file_idx, a_ori_img_file in enumerate(sorted_ori_image_files):
        cur_img = cv.imread(str(a_ori_img_file))
        if cur_base_img_file is None:
            cur_base_img_file = a_ori_img_file
            cur_base_img = cur_img
            idiv_img_count = idiv_img_count + 1
            cv.imwrite(f'{output_image_path}/{idiv_img_count}.jpg', cur_base_img)
            continue
        delta = np.sum(np.abs(cur_img - cur_base_img)) / 3 / pixcel
        if delta > d_threshold:
            cur_base_img_file = a_ori_img_file
            cur_base_img = cur_img
            idiv_img_count = idiv_img_count + 1
            cv.imwrite(f'{output_image_path}/{idiv_img_count}.jpg', cur_base_img)
        print(f'delta is {delta}')


