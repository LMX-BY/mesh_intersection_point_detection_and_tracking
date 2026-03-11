import os
import pathlib
import json
from typing import Dict, List
import time


def get_filenames_of_path(path: pathlib.Path, ext: str = "*") -> List[pathlib.Path]:
    """
    Returns a list of files in a directory/path. Uses pathlib.
    """
    filenames = [file for file in path.glob(ext) if file.is_file()]
    assert len(filenames) > 0, f"No files found in path: {path}"
    return filenames


def read_json(path: pathlib.Path) -> dict:
    with open(str(path), "r") as fp:  # fp is the file pointer
        file = json.loads(s=fp.read())

    return file


def save_json(obj, path: pathlib.Path) -> None:
    with open(path, "w") as fp:  # fp is the file pointer
        json.dump(obj=obj, fp=fp, indent=4, sort_keys=False)


def create_folder_with_cur_time_info(parent_str, folder_name):
    # !! 未判断parent路径是否存在
    curr_local_time = time.localtime()
    time_str = time.strftime('%Y%m%d_%H%M', curr_local_time)
    path_folder_name = f'{parent_str}/{folder_name}_{time_str}'
    if not os.path.exists(path_folder_name):
        os.makedirs(path_folder_name)
    return path_folder_name
