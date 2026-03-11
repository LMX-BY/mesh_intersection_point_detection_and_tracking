from pathlib import Path
import shutil
from utiles import file_related
import numpy as np
import json
import cv2 as cv
source_img_path = Path(r"H:\code\python\net_detection_and_tracking\data_0.1\wj"
                               r"-original_image_diff_moving_direction_29241030\imgs_train")
error_img_path = Path(r'H:\code\python\net_detection_and_tracking\error_images')
# copy_image_path = Path(r'H:\code\python\net_detection_and_tracking\data_0.1\wj-original_image_diff_moving_direction_29241030\imgs_train')
label_path = Path(r'H:\code\python\net_detection_and_tracking\data_0.1\test_Moore_data_labels\2024-11-12_08_02_46-export\data\20241011143651\net\train_img\moving_right\scene_4')
valid_label_name = ['net_point','lane']
# distance threshold between the same point in sequence frame
net_point_distance_threshold = 60
error_point_color_before = (0, 0, 255)
error_point_color_current = (0, 255, 0)
error_point_vector_color = (255, 0, 0)
error_id_text_font_face = cv.FONT_HERSHEY_SIMPLEX
error_id_text_font_scale = 0.3
error_id_text_color = (255, 0, 0) # 蓝色文本
error_id_text_thickness = 2
if __name__ == '__main__':
    origin_img_files = file_related.get_filenames_of_path(source_img_path)
    #source_img_path.walk(on_error=print)


    # copy nested images to a target path
    # all_nested_img = sorted(source_img_path.glob('**/*.jpg'))
    # for a_img in all_nested_img:
    #     shutil.copy(a_img, copy_image_path)
    initial_state = 1
    before_net_points_dict = {}
    before_file_name = None
    before_labels = None

    for cur_labeled_file in label_path.iterdir():

        cur_file_name = cur_labeled_file.stem
        cur_image_file_path = fr'{source_img_path}\{cur_file_name}.jpg'
        img = cv.imread(cur_image_file_path)
        cur_label_file_data = file_related.read_json(cur_labeled_file)
        cur_labels = cur_label_file_data['labels']
        cur_net_points_dict = {}

        # check label names
        for cur_label in cur_labels:
            if cur_label['label'] not in valid_label_name:
                print(f'label name error: {cur_label["label"]} in {cur_file_name}')
                continue

        for cur_label in cur_labels:
            cur_net_points_dict[cur_label['id']] = cur_label['points']

        # initialize a before data

        if initial_state == 1:
            before_file_name = cur_file_name
            before_labels = cur_labels
            before_net_points_dict = cur_net_points_dict
            initial_state = 0
            continue

        distance_list = []
        has_distance_error = False
        for cur_label in cur_labels:
            a_cur_net_point_ID = cur_label['id']
            if cur_label['label'] != 'net_point':
                continue
            a_cur_net_point = cur_label['points']
            if a_cur_net_point_ID in before_net_points_dict:
                a_before_net_point = before_net_points_dict[a_cur_net_point_ID]
                # calculate distance between current and previous net points
                distance = np.linalg.norm(np.array(a_cur_net_point) - np.array(a_before_net_point))
                distance_list.append(distance)
                if distance > net_point_distance_threshold:
                    has_distance_error = True
                    test_array = np.array(a_before_net_point)
                    cv.circle(img, np.array(a_before_net_point).astype(np.int32), 4, error_point_color_before)
                    cv.circle(img, np.array(a_cur_net_point).astype(np.int32), 4, error_point_color_current)
                    cv.putText(img, str(a_cur_net_point_ID),np.array(a_cur_net_point).astype(np.int32), error_id_text_font_face, error_id_text_font_scale, error_id_text_color, error_id_text_thickness)
                    cv.line(img, np.array(a_before_net_point).astype(np.int32), np.array(a_cur_net_point).astype(np.int32), error_point_vector_color)
                    print(f'distance error: ID:{a_cur_net_point_ID} with distance {distance} in file {cur_file_name}')
                pass
        if has_distance_error:
            cv.imwrite(f'{str(error_img_path)}/{cur_file_name}.png', img)
        distance_list = np.array(distance_list)
        mean = np.mean(distance_list)
        variance = np.var(distance_list)
        print(f'mean: {mean}, variance: {variance}')

        # reset before data
        before_file_name = cur_file_name
        before_labels = cur_labels
        before_net_points_dict = cur_net_points_dict


        pass

    pass
