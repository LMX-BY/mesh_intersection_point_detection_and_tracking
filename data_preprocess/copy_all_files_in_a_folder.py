import os
import shutil
from pathlib import Path

def get_files_with_extension(directory, extension):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(extension):
                yield os.path.join(root, file)


if __name__ == '__main__':
    file_path = r'H:\code\wangjin\point json\data\20241011143651\net\test_img'
    output_path = r'H:\code\python\net_detection_and_tracking\data_0.1\all_label_data_net_point\test'
    all_files = get_files_with_extension(file_path, '.json')
    for a_file in all_files:
        a_path = Path(a_file)
        name = a_path.name
        shutil.copy(a_file, f'{output_path}/{name}')
        pass

    pass
