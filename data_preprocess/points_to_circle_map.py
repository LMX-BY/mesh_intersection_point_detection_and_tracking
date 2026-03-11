import os
from pathlib import Path
from utiles import file_related
import numpy as np
import cv2 as cv

if __name__ == '__main__':
    img_width = 960
    img_height = 576
    circle_map_width = 960
    circle_map_height = 576
    circle_radius = 6
    ln_net_point = 'net_point'
    label_file_path = Path(r'H:\code\python\net_detection_and_tracking\data_0.1\all_label_data_net_point\test')
    heatmap_output_path = Path(fr'H:\code\python\net_detection_and_tracking\data_0.1\all_label_data_net_point\{label_file_path.stem}_r{circle_radius}')
    orig_img_path = Path(r'H:\code\python\net_detection_and_tracking\data_0.1\wj-original_image_diff_moving_direction_29241030\img_all')
    if not os.path.exists(heatmap_output_path):
        os.makedirs(heatmap_output_path)
    label_files = file_related.get_filenames_of_path(label_file_path)
    for cur_label_file in label_files:
        file_name = cur_label_file.stem
        img_file_name = orig_img_path / f'{file_name}.jpg'
        cur_orig_img = cv.imread(str(img_file_name))
        orig_img_width, orig_img_height = cur_orig_img.shape[1], cur_orig_img.shape[0]
        labels = file_related.read_json(cur_label_file)['labels']
        width_scale_rate = circle_map_width / orig_img_width
        height_scale_rate = circle_map_height / orig_img_height
        heat_map_name = cur_label_file.stem
        cur_circle_map = np.zeros(shape=(circle_map_height, circle_map_width), dtype=np.uint8)
        for a_label in labels:
            if a_label['label'] != ln_net_point:
                continue
            a_point = a_label['points']
            cv.circle(cur_circle_map, (int(a_point[0]* width_scale_rate) , int(a_point[1] * height_scale_rate)), circle_radius, 1, thickness=-1)

        # normalized_heatmap = cv.normalize(resized_filtered_heatmap, None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX,
        #                                   dtype=cv.CV_32F)

        np.save(f'{str(heatmap_output_path / heat_map_name)}_heatmap.npy', cur_circle_map)

        pass
