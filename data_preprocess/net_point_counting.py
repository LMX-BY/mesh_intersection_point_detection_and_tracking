import os
from pathlib import Path
from utiles import file_related
import numpy as np
import cv2 as cv

if __name__ == '__main__':
    ln_net_point = 'net_point'
    label_file_path = Path(r'H:\code\python\net_detection_and_tracking\data_0.1\all_label_data_net_point\tracking_930')
    label_files = file_related.get_filenames_of_path(label_file_path)
    cur_point_num = 0
    for cur_label_file in label_files:
        labels = file_related.read_json(cur_label_file)['labels']
        for a_label in labels:
            if a_label['label'] != ln_net_point:
                continue
            cur_point_num = cur_point_num + 1

    print(f'cur_point_num = {cur_point_num}')

    pass
