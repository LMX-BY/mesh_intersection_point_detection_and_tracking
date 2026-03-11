import cv2 as cv
import numpy as np
if __name__ == '__main__':
    test_array = np.array([10,10,10,10.2,10.9,0.6,5,6,4]).astype(np.uint8)
    ret, threshold = cv.threshold(test_array, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    print(ret)