from pathlib import Path
from utiles import file_related
import numpy as np
import cv2 as cv


if __name__ == '__main__':
    img_width = 960
    img_height = 576
    heatmap_width = 480
    heatmap_height = 288
    sigma = 1
    k_size = 41
    only_classficication = False
    binary = True
    ln_net_point = 'net_point'
    label_file_path = Path(r'H:\code\python\net_detection_and_tracking\data_0.1\test_labels')
    heatmap_output_path = Path(r'H:\code\python\net_detection_and_tracking\data_0.1\test_heatmap_half\binary_heatmap')
    label_files = file_related.get_filenames_of_path(label_file_path)
    gaussian_kernel = cv.getGaussianKernel(k_size, sigma)
    for cur_label_file in label_files:
        labels = file_related.read_json(cur_label_file)['labels']
        heat_map_name = cur_label_file.stem
        cur_heat_map = np.zeros(shape=(img_height, img_width))
        for a_label in labels:
            if a_label['label'] != ln_net_point:
                continue
            a_point = a_label['points']
            cur_heat_map[int(np.around(a_point[1])), int(np.around(a_point[0]))] = 1
        if only_classficication:
            np.save(f'{str(heatmap_output_path / heat_map_name)}_heatmap.npy', cur_heat_map)
            continue
        heatmap_with_boarder = cv.copyMakeBorder(cur_heat_map, k_size, k_size, k_size, k_size, cv.BORDER_CONSTANT, value=0)
        boardered_col = heatmap_with_boarder.shape[0]
        boardered_row = heatmap_with_boarder.shape[1]

        # !!注意直接用GaussianBlur时默认的boarder处理会不符合预期，靠近图片边框的值会偏大，影响后续训练
        filtered_heatmap_with_boarder = cv.GaussianBlur(heatmap_with_boarder, (k_size, k_size), sigma, sigma)
        filtered_heatmap = filtered_heatmap_with_boarder[k_size:boardered_col-k_size, k_size:boardered_row-k_size]

        resized_filtered_heatmap = cv.resize(filtered_heatmap,[heatmap_width, heatmap_height])
        normalized_heatmap = cv.normalize(resized_filtered_heatmap, None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX,
                                          dtype=cv.CV_32F)
        if binary:
            normalized_heatmap = np.where(normalized_heatmap != 0, 1, normalized_heatmap)

        np.save(f'{str(heatmap_output_path/heat_map_name)}_heatmap.npy', normalized_heatmap)

        pass
