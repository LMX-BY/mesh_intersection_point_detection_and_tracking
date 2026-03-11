import cv2
import pathlib
video_file_path = r'G:\dataSet\huadu_localization_pool\video\gopro_ji1ji2.mp4'
output_image_path = r'G:\dataSet\huadu_localization_pool\video\gopro_ji1ji2'
video_name = 'camera_ji01'
start_count = 0
rate = 1
if __name__ == '__main__':
    pass
    video_capture = cv2.VideoCapture(video_file_path)
    count = 0
    while(True):
        ret, frame = video_capture.read()
        print(f'cur count is {count}')
        if ret:
            if count < start_count:
                count = count + 1
                continue
            if count % rate == 0:
                cv2.imwrite(f'{output_image_path}/{count}.jpg', frame)
        else:
            break
        count = count + 1

