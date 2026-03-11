import cv2 as cv
import numpy as np
from pathlib import Path
from utiles import file_related
import os
if __name__ == '__main__':
    point_label_file_path = Path(fr'H:\code\python\net_detection_and_tracking\data_0.1\tracking\label\1')
    ori_img_file_path = Path(r'H:\code\python\net_detection_and_tracking\data_0.1\wj'
                             r'-original_image_diff_moving_direction_29241030\img_all_size_uniformed')
    results_save_path = Path(
        fr'H:\code\python\net_detection_and_tracking\test_output\tracking_results\test_point_label_display_1')



    if not os.path.exists(results_save_path):
        # 创建文件夹（支持多级目录创建）
        os.makedirs(results_save_path)
    point_label_files = file_related.get_filenames_of_path(point_label_file_path)
    for cur_point_label_file in point_label_files:
        label_file_name = cur_point_label_file.stem
        img_file_name = ori_img_file_path / (label_file_name + '.jpg')
        cur_img = cv.imread(str(img_file_name))
        img_height, img_width = cur_img.shape[:2]
        labels = file_related.read_json(cur_point_label_file)['labels']
        for a_label in labels:
            point = a_label['points']
            id = a_label['id']
            radius = 2
            cv.circle(cur_img, (int(point[0]), int(point[1])), 2, (0, 0, 255), thickness=radius)
            text = str(id)
            #text_org = (int(point[0] - radius-2), int(point[1] - radius-2))  # 左上角坐标
            text_org = (int(point[0]), int(point[1]))
            cv.putText(
                cur_img,
                text,
                text_org,
                cv.FONT_HERSHEY_SIMPLEX,  # 字体
                0.8,  # 字号
                (0, 0, 255),  # 红色文字
                2,  # 线宽
            )
        cv.imwrite(str(results_save_path / f'{label_file_name}_points_with_id.jpg'), cur_img)