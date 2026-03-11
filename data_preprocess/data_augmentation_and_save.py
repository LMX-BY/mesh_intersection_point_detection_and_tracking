from data_loader import net_point_and_lane_data_loader
from utiles import file_related
from pathlib import Path
import torch.utils.data as data
import cv2 as cv
import numpy as np


def AddSaltPepperNoise(src, rate):
    srcCopy = src.copy()
    height, width = srcCopy.shape[0:2]
    noiseCount = int(rate * height * width / 2)
    # add salt noise
    X = np.random.randint(width, size=(noiseCount,))
    Y = np.random.randint(height, size=(noiseCount,))
    srcCopy[Y, X] = 255
    # add black peper noise
    X = np.random.randint(width, size=(noiseCount,))
    Y = np.random.randint(height, size=(noiseCount,))
    srcCopy[Y, X] = 0
    return srcCopy


scale_rate = [0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.2, 1.4, 1.6, 1.8, 2.0]

if __name__ == '__main__':
    ori_img_path = Path(r'H:\code\python\net_detection_and_tracking\data_0.1\wj'
                        r'-original_image_diff_moving_direction_29241030\img_all_size_uniformed')
    ori_label_path = Path(r'H:\code\python\net_detection_and_tracking\data_0.1\all_label_data_net_point'
                          r'\test_r6')
    img_results_save_path = Path(r'H:\code\python\net_detection_and_tracking\data_0.1\wj'
                                 r'-original_image_diff_moving_direction_29241030\augmented_img')
    label_save_path = Path(r'H:\code\python\net_detection_and_tracking\data_0.1\all_label_data_net_point'
                           r'\augmented_test_r6')
    label_files = file_related.get_filenames_of_path(ori_label_path)

    train_dataset = net_point_and_lane_data_loader.NetPointAndLaneDataLoader(ori_img_path, label_files)

    for index in range(len(train_dataset)):
        img, target, ori_img, cur_label_file_name = train_dataset[index]
        label_file_name_str_split = cur_label_file_name.split('_')
        img_name = ''
        for file_name_index, a_word in enumerate(label_file_name_str_split):
            if a_word == 'heatmap':
                break
            if file_name_index == 0:
                img_name = img_name + a_word
                continue
            img_name = img_name + '_' + a_word
        ori_img_width = ori_img.shape[1]
        ori_img_height = ori_img.shape[0]
        for a_scale_rate in scale_rate:
            reconstructed_img = np.zeros_like(ori_img)
            reconstructed_mask = np.zeros_like(target)
            # 放缩
            img_with_scaling = cv.resize(ori_img, None, fx=a_scale_rate, fy=a_scale_rate)
            mask_with_scaling = cv.resize(target, None, fx=a_scale_rate, fy=a_scale_rate)
            mask_with_scaling[mask_with_scaling!=0] = 1
            # 添加椒盐噪声
            img_with_salt_pepper = AddSaltPepperNoise(img_with_scaling, 0.05)
            # 放回原尺寸的图像中
            scale_img_width = img_with_scaling.shape[1]
            scale_img_height = img_with_scaling.shape[0]
            start_x = int(np.abs(np.ceil((ori_img_width - scale_img_width) / 2)))
            start_y = int(np.abs(np.ceil((ori_img_height - scale_img_height) / 2)))
            if a_scale_rate <= 1:
                reconstructed_img[start_y:start_y + scale_img_height, start_x:start_x + scale_img_width] = img_with_salt_pepper
                reconstructed_mask[start_y:start_y + scale_img_height, start_x:start_x + scale_img_width] = mask_with_scaling
            else:
                reconstructed_img = img_with_salt_pepper[start_y:start_y + ori_img_height, start_x:start_x + ori_img_width]
                reconstructed_mask = mask_with_scaling[start_y:start_y + ori_img_height, start_x:start_x + ori_img_width]
            cv.imwrite(f'{str(img_results_save_path)}\\{img_name}_sr{int(a_scale_rate*10)}.jpg' , reconstructed_img)
            np.save(f'{str(label_save_path)}\\{img_name}_sr{int(a_scale_rate*10)}_heatmap.npy', reconstructed_mask)
        # 添加椒盐噪声

        # 生成图像随机旋转

        pass

    pass
