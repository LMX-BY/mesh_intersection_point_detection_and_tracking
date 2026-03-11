from utiles import file_related
from pathlib import Path
import numpy as np
import shutil
if __name__ == '__main__':
    input_file_path = Path(r'H:\code\python\net_detection_and_tracking\data_0.1\all_label_data_net_point\train')
    output_file_path = Path(r'H:\code\python\net_detection_and_tracking\data_0.1\all_label_data_net_point\train'
                            r'\validation')
    file_names = file_related.get_filenames_of_path(input_file_path)
    validation_size = 100
    sample_index = np.random.choice(len(file_names), validation_size, replace=False)
    file_names = np.array(file_names)
    sampled_file_names = file_names[sample_index]
    for a_file in sampled_file_names:
        file_name = a_file.name
        shutil.move(a_file, f'{str(output_file_path)}/{file_name}')
        pass
    pass
