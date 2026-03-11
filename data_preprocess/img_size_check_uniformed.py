from utiles import file_related
from pathlib import Path
import cv2 as cv
if __name__ == '__main__':
    input_path = Path(r'G:\dataSet\huadu_localization_pool\video\gopro_ji2')
    output_path = Path(r'G:\dataSet\huadu_localization_pool\video\gopro_g2_small_size')
    img_files = file_related.get_filenames_of_path(input_path)
    for a_img_file in img_files:
        img = cv.imread(str(a_img_file))
        name = a_img_file.name
        if img.shape != (540, 960, 3):
            img = cv.resize(img, (960, 540))
        cv.imwrite(str(output_path / name), img)
        pass
