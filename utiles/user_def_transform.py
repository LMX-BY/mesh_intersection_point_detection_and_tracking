import random

import torch
from torchvision.transforms import functional as F
from torchvision import transforms as pytorch_tranF
import cv2 as cv


class Compose(object):
    """组合多个transform函数"""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ComposeOnlyImg(object):
    """组合多个transform函数"""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image):
        for t in self.transforms:
            image = t(image)
        return image


class ToTensor(object):
    """将PIL图像转为Tensor"""

    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target


class ToTensorOnlyImg(object):
    """将PIL图像转为Tensor"""

    def __call__(self, image):
        image = F.to_tensor(image)
        return image


class ToTensorWithDataAndTarget(object):
    """将PIL图像转为Tensor"""

    def __call__(self, image, target):
        image = F.to_tensor(image)
        target = torch.tensor(target).unsqueeze(0)
        return image, target


class MedianFilter(object):
    """将PIL图像转为Tensor"""

    def __call__(self, image, target):
        image = cv.medianBlur(image, 5)
        # image = F.to_tensor(image)
        #target = F.to_tensor(target)
        return image, target


class Normalize(object):
    def __init__(self, mean, std):
        self.pytorch_transform = pytorch_tranF.Normalize(mean, std)

    def __call__(self, image, target):
        image = self.pytorch_transform(image)
        return image, target


class NormalizeOnlyImg(object):
    def __init__(self, mean, std):
        self.pytorch_transform = pytorch_tranF.Normalize(mean, std)

    def __call__(self, image):
        image = self.pytorch_transform(image)
        return image


class Grayscale(object):
    def __call__(self, image, target):
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        return image, target


class RandomHorizontalFlip(object):
    """随机水平翻转图像以及bboxes"""

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)  # 水平翻转图片
            bbox = target["boxes"]
            # bbox: xmin, ymin, xmax, ymax
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]  # 翻转对应bbox坐标信息
            target["boxes"] = bbox
        return image, target
