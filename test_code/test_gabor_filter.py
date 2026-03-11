import cv2 as cv
import numpy as np
from pathlib import Path
if __name__ == '__main__':
    p1 = np.array([173,161])
    p2 = np.array([163,190])
    dis = np.linalg.norm(p1-p2)
    filter_path = Path(r'H:\code\python\net_detection_and_tracking\test_output\gabor_test\filter')
    filtered_img_path = Path(r'H:\code\python\net_detection_and_tracking\test_output\gabor_test\filtered_img')
    input_img_path = Path(r"H:\code\python\net_detection_and_tracking\test_output\gabor_test\img\HCVR_ch1_main_20240320103747_20240320120000_36980.jpg")
    ksize = (61, 61)  # Size of the filter
    sigma = np.linspace(1,20, 20)  # Standard deviation of the gaussian function
    theta = np.linspace(0, np.pi, 20)
    #lambd = np.linspace(1.0 * np.pi / 32,  1.0 * np.pi,10)
    lambd = np.linspace(20,  40,10)# Wavelength of the sinusoidal factor
    gamma = 0.5  # Spatial aspect ratio
    psi = 0

    cur_img = cv.imread(str(input_img_path))
    cur_img_name = input_img_path.stem
    cur_img_gray = cv.cvtColor(cur_img, cv.COLOR_BGR2GRAY)
    for a_sigma in sigma:
        for a_lambd in lambd:
            for a_theta in theta:
                gaborKernel = cv.getGaborKernel(ksize, a_sigma, a_theta, a_lambd, gamma, psi)
                #normalized_array = cv.normalize(gaborKernel,None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
                filtered_img = cv.filter2D(cur_img_gray, -1, gaborKernel)
                normalized_filtered_img = cv.normalize(filtered_img, None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
                #cv.imwrite(str(out_put_path / f'gabor_filter_sigma_{a_sigma}_lamda_{a_lambd}.jpg'), normalized_array)
                cv.imwrite(f'{str(filtered_img_path)}/{cur_img_name}_{a_sigma}_{a_lambd}_{a_theta}.bmp', filtered_img)

    # for a_sigma in sigma:
    #     for a_theta in theta:
    #         gaborKernel = cv.getGaborKernel(ksize, a_sigma, a_theta, a_lambd, gamma, psi)
    #         #normalized_array = cv.normalize(gaborKernel,None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
    #         filtered_img = cv.filter2D(cur_img_gray, -1, gaborKernel)
    #         #cv.imwrite(str(out_put_path / f'gabor_filter_sigma_{a_sigma}_lamda_{a_lambd}.jpg'), normalized_array)
    #         cv.imwrite(f'{str(filtered_img_path)}/{cur_img_name}_{a_sigma}_{a_lambd}_{a_theta}.bmp', filtered_img)
    #normalized_array_img = cv.cvtColor(normalized_array, cv.COLOR_BGR2RGB)
    #cv.imwrite(str(out_put_path / 'gabor_filter.jpg'), normalized_array)
    pass