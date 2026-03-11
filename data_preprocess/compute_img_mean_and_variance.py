from pathlib import Path
from utiles import file_related
import cv2 as cv
import torch
from torchvision import transforms
img_mean = torch.tensor([0.5832, 0.5959, 0.2992])
img_variance = torch.tensor([0.1738, 0.1815, 0.1696])
img_gray_mean = torch.tensor([0.5069])
img_gray_variance = torch.tensor([0.1629])
if __name__ == '__main__':
    imgs_path = Path(r'H:\code\python\net_detection_and_tracking\data_0.1\wj'
                     r'-original_image_diff_moving_direction_29241030\img_all_size_uniformed')
    img_files = file_related.get_filenames_of_path(imgs_path)
    img_list = []
    img_gray_list = []
    to_tensor = transforms.ToTensor()
    for a_img_file in img_files:
        a_img = cv.imread(str(a_img_file))
        a_img_gray = cv.cvtColor(a_img, cv.COLOR_BGR2GRAY)
        img_gray_list.append(to_tensor(a_img_gray))
        img_list.append(to_tensor(a_img))
    tensor_imgs = torch.stack([img_t for img_t in img_list], dim=3)
    mean = tensor_imgs.view(3, -1).mean(dim=1)
    variance = tensor_imgs.view(3, -1).std(dim=1)

    tensor_imgs_gray = torch.stack([img_t for img_t in img_gray_list], dim=3)
    mean_gray = tensor_imgs_gray.view(1, -1).mean(dim=1)
    variance_gray = tensor_imgs_gray.view(1, -1).std(dim=1)
    print(f'mean: {mean}')
    print(f'variance: {variance}')
    pass
