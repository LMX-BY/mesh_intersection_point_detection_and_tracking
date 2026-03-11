import numpy as np
import cv2 as cv
from pathlib import Path


def generate_heatmap(height, width, landmark, sigma):
    X, Y = np.meshgrid(np.arange(height), np.arange(width))
    heatmap = np.exp(-(((X - landmark[0]) ** 2) + ((Y - landmark[1]) ** 2)) / (2 * sigma ** 2))

    return heatmap


if __name__ == '__main__':
    heatmap_output_path = Path(r'H:\code\python\net_detection_and_tracking\test_output\heat_map')
    # test our function
    height = 100
    width = 100
    landmark = np.array([50, 50])  # landmark in the center of the image
    sigma = 3  # standard deviation
    heatmap = generate_heatmap(height, width, landmark, sigma)
    normalized_heatmap = cv.normalize(heatmap, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX,
                                           dtype=cv.CV_32F)
    cv.imwrite(f'{str(heatmap_output_path)}/heatmap.png', normalized_heatmap)
