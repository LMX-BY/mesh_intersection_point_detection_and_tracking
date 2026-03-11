import cv2
import torch
import torch.utils.data as data
from pathlib import Path
from utiles import file_related
import numpy as np


class NetPointAndLaneDataLoader(data.Dataset):
    def __init__(self, img_path, label_files, transform=None):
        self.img_path = img_path
        self.label_files = label_files
        self.transform = transform
        pass

    def __getitem__(self, index):
        cur_label_file = self.label_files[index]
        cur_label_file_name = cur_label_file.stem
        label_file_name_str_split = cur_label_file_name.split('_')
        img_name = ''
        for file_name_index, a_word in enumerate(label_file_name_str_split):
            if a_word == 'heatmap':
                break
            if file_name_index == 0:
                img_name = img_name + a_word
                continue
            img_name = img_name + '_' + a_word
        cur_img_file_name = self.img_path / f'{img_name}.jpg'

        # load data from file
        #ori_img = cv2.imread(str(cur_img_file_name), cv2.IMREAD_GRAYSCALE)
        ori_img = cv2.imread(str(cur_img_file_name))

        heat_map = np.load(str(cur_label_file))

        if self.transform is not None:
            img, target = self.transform(ori_img, heat_map)
        else:
            img, target = ori_img, heat_map

        return img, target, ori_img, cur_label_file_name

    def __len__(self):
        return len(self.label_files)


def detection_collate(batch):
    targets = []
    transformed_imgs = []
    file_names = []
    ori_imgs = []
    for _, sample in enumerate(batch):
        transformed_imgs.append(sample[0])
        targets.append(sample[1])
        ori_imgs.append(sample[2])
        file_names.append(sample[3])

    # try:
    #     torch.stack(transformed_imgs, 0), torch.stack(targets, 0), file_names
    # except:
    #     pass

    return torch.stack(transformed_imgs, 0), torch.stack(targets, 0), file_names, ori_imgs


if __name__ == '__main__':
    # test

    # Path setting
    img_path = Path(r'H:\code\python\net_detection_and_tracking\data_0.1\wj'
                    r'-original_image_diff_moving_direction_29241030\imgs_train')
    heatmap_path = Path(r'H:\code\python\net_detection_and_tracking\data_0.1\test_heatmap')

    heatmap_files = file_related.get_filenames_of_path(heatmap_path)

    # Create DataLoader Object

    dataloader = NetPointAndLaneDataLoader(img_path, heatmap_files)
    a_img, a_label, a_file_name = dataloader[0]

    pass
