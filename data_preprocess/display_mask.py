import cv2 as cv
import numpy as np
if __name__ == '__main__':
    source_img_file = r'H:\code\python\net_detection_and_tracking\data_0.1\mask_label_for_display\Net-3\images\train\HCVR_ch1_main_20240320103747_20240320120000_41067.jpg'
    mask_file = r'H:\code\python\net_detection_and_tracking\data_0.1\mask_label_for_display\Net-3\mask\train\HCVR_ch1_main_20240320103747_20240320120000_41067.png'
    display_img_path = r'H:\code\python\net_detection_and_tracking\test_output\mask_display_results'
    mask_img = cv.imread(mask_file)
    source_img = cv.imread(source_img_file)
    img_height, img_width = source_img.shape[:2]

    colored_net_img = np.zeros_like(source_img)
    colored_line_img = np.zeros_like(source_img)
    mask_for_point = np.all(mask_img == [1, 1, 1], axis=-1)
    mask_for_line = np.all(mask_img == [2, 2, 2], axis=-1)
    colored_net_img[mask_for_point] = [0, 97, 255]
    colored_line_img[mask_for_line] = [0, 0, 255]

    comb_img_0 = cv.addWeighted(source_img, 0.5, colored_net_img, 0.5, 0)
    comb_img_1 = cv.addWeighted(source_img, 0.5, colored_line_img, 0.5, 0)
    comb_img_2 = cv.addWeighted(comb_img_0, 0.5, comb_img_1, 0.5, 0)
    cv.imwrite(f'{display_img_path}/HCVR_ch1_main_20240320103747_20240320120000_41067.png',comb_img_2)



