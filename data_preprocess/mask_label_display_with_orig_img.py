import cv2 as cv
import numpy as np
from pathlib import Path
from utiles import file_related
if __name__ == '__main__':
    # 目前支持.npy的mask标签
    label_file_path = Path(r'H:\code\python\net_detection_and_tracking\data_0.1\all_label_data_net_point\augmented_validation_r6')
    orig_img_path = Path(r'H:\code\python\net_detection_and_tracking\data_0.1\wj'
                    r'-original_image_diff_moving_direction_29241030\augmented_img')
    display_img_save_path = r'H:\code\python\net_detection_and_tracking\test_output\mask_label_display_with_orig_img'
    label_files = file_related.get_filenames_of_path(label_file_path)

    for a_label_file in label_files:
        cur_label_file_name = a_label_file.stem
        label_file_name_str_split = cur_label_file_name.split('_')
        img_name = ''
        for file_name_index, a_word in enumerate(label_file_name_str_split):
            if a_word == 'heatmap':
                break
            if file_name_index == 0:
                img_name = img_name + a_word
                continue
            img_name = img_name + '_' + a_word

        cur_img_file_name = orig_img_path / f'{img_name}.jpg'
        mask = np.load(str(a_label_file))
        ori_img = cv.imread(str(cur_img_file_name))

        colored_net_img = np.zeros_like(ori_img)
        test1 = (mask == 1)
        #mask_for_point = np.all(mask == 1, axis=-1)
        colored_net_img[test1] = [0, 97, 255]

        comb_img = cv.addWeighted(ori_img, 0.5, colored_net_img, 0.5, 0)
        cv.imwrite(f'{display_img_save_path}/{img_name}.png', comb_img)
        pass

