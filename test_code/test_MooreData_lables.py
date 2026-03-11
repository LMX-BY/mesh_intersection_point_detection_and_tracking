import pathlib
from utiles import file_related
general_json_file_path = pathlib.Path(r'H:\code\python\net_detection_and_tracking\data_0.1\test_Moore_data_labels'
                                      r'\json_1\originData.json')
coco_format_file_path = pathlib.Path(r'H:\code\python\net_detection_and_tracking\data_0.1\test_Moore_data_labels'
                                     r'\coco_2\originData.json')
if __name__ == '__main__':
    label_content = file_related.read_json(coco_format_file_path)
    pass
