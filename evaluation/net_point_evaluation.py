# 输入mask和阈值分割后的预测掩膜，输出precision,recall和F1-scorey以及中间结果包括TP，FP，FN等
# （1）预测掩膜连通域分割；（2）mask掩膜连通域提取。计算连通域和连通域之间的匹配分数，将预测掩膜中
# ！！！coco评价的问题,标签类型有问题，现有标签转换为coco的比较麻烦？，coco在计算时必须要用到bbox？？coco的mask似乎
# 方案二：将预测掩膜进行连通域划分，转换为点，然后做点匹配，来确定正样本和负样本，对于覆盖了多个点的连通域，进行移除。
# 目前的点标注只有原始的，缺乏数据增强的点标注，是否需要生成。（需要）
# 如何判断一个连通域覆盖了多个网点？？
from skimage import measure
import numpy as np
from utiles import file_related
from pathlib import Path
import cv2

FP_type = 1
TP_type = 2
class NetPointDetectionEvaluation(object):
    def __init__(self):
        self.total_TP = 0
        self.total_detection_points = 0
        self.total_labeled_points = 0

    def reset(self):
        self.total_TP = 0
        self.total_detection_points = 0
        self.total_labeled_points = 0

    def evaluate(self, detected_mask, labeled_mask, ev_error_threshold):
        return_detection_eva_type_map = True
        # 获取连通域
        detected_points_size, detected_indexed_mask, detected_stats, detected_centroids = cv2.connectedComponentsWithStats(
            detected_mask, connectivity=8)
        # 需要把背景部分的连通域剔除，其最大的特点在于覆盖范围很大
        # 导入标记掩膜并提取连通域
        labels_size, labels_indexed_mask, labels_stats, labels_centroids = cv2.connectedComponentsWithStats(
            labeled_mask,
            connectivity=8)
        img_height, img_width = detected_mask.shape
        # 初始化正样本
        total_detected_points_size = detected_points_size - 1
        TP = 0
        FP = 0
        # 显示用
        detection_eva_type_map = None
        if return_detection_eva_type_map:
            detection_eva_type_map = np.zeros((img_height, img_width), dtype=np.uint8)
        # 找到标记掩膜中的背景索引
        label_background_index = None
        for point_index in range(labels_size - 1):
            cur_label_point_info = labels_stats[point_index]
            # 判断是否是背景
            if cur_label_point_info[2] > img_width / 2 or cur_label_point_info[3] > img_height / 2:
                label_background_index = point_index
        assert label_background_index is not None
        # 验证预测掩膜中每个前景连通域是TP还是FP
        before_detection_eva_type = None
        for point_index in range(total_detected_points_size):
            cur_detected_point_info = detected_stats[point_index]
            # 1代表FP，2代表TP,构建detection_type_map
            if before_detection_eva_type is not None and return_detection_eva_type_map:
                before_detected_point_info = detected_stats[point_index-1]
                for col_index in range(before_detected_point_info[0],
                                       before_detected_point_info[0] + before_detected_point_info[2]):
                    for row_index in range(before_detected_point_info[1],
                                           before_detected_point_info[1] + before_detected_point_info[3]):
                        if detected_mask[row_index][col_index] == 1:
                            detection_eva_type_map[row_index][col_index] = before_detection_eva_type
            # 判断是否是背景
            if cur_detected_point_info[2] > img_width / 2 or cur_detected_point_info[3] > img_height / 2:
                continue
            # 判断检测的连通域是否包括了两个点
            cur_corresp_label_index = None
            detection_two_point_region = False
            for col_index in range(cur_detected_point_info[0], cur_detected_point_info[0] + cur_detected_point_info[2]):
                for row_index in range(cur_detected_point_info[1],
                                       cur_detected_point_info[1] + cur_detected_point_info[3]):
                    label_index = labels_indexed_mask[row_index][col_index]
                    if label_index == label_background_index:
                        continue
                    if cur_corresp_label_index is None:
                        cur_corresp_label_index = label_index
                        continue
                    if cur_corresp_label_index != label_index:
                        detection_two_point_region = True
                        break
            if cur_corresp_label_index is None:
                FP = FP + 1
                before_detection_eva_type = FP_type
                continue
            # 判断中心点是否在误差范围内
            if detection_two_point_region:
                FP = FP + 1
                before_detection_eva_type = FP_type
                continue
            cur_detected_point_centroid = detected_centroids[point_index]
            corresponding_label_point_centroid = labels_centroids[cur_corresp_label_index]
            error_distance = np.linalg.norm(cur_detected_point_centroid - corresponding_label_point_centroid)
            if error_distance < ev_error_threshold:
                before_detection_eva_type = TP_type
                TP = TP + 1
            else:
                before_detection_eva_type = FP_type
                FP = FP + 1
        self.total_detection_points = self.total_detection_points + total_detected_points_size
        self.total_labeled_points = self.total_labeled_points + (labels_size - 1)
        self.total_TP = self.total_TP + TP
        return detection_eva_type_map

    def summary(self):
        if self.total_detection_points == 0:
            precision_all = 0
        else:
            precision_all = self.total_TP / self.total_detection_points
        recall_all = self.total_TP / self.total_labeled_points
        if precision_all + recall_all == 0:
            F1_measure = 0
        else:
            F1_measure = 2 * (precision_all * recall_all) / (precision_all + recall_all)
        print(f'recall_all: {recall_all}')
        print(f'precision_all: {precision_all}')
        print(f'F1_measure: {F1_measure}')
        return {'recall': recall_all,
                'precision': precision_all,
                'F1_measure': F1_measure}


if __name__ == '__main__':
    # 导入预测掩膜图像和坐标标记
    pre_mask_file_path = Path(r'H:\code\python\net_detection_and_tracking\test_output\test_results_display')
    point_label_file_path = Path(r'H:\code\python\net_detection_and_tracking\data_0.1\all_label_data_net_point\test_r6')
    error_threshold = 6  # 放缩或放大的图片是否需要考虑更小或更大的错误阈值，需要的
    pre_mask_files = file_related.get_filenames_of_path(pre_mask_file_path)
    TP_all_files = 0
    FP_all_files = 0
    detected_point_size_all_file = 0
    labeled_point_size_all_files = 0
    evaluator = NetPointDetectionEvaluation(error_threshold)
    for cur_pre_mask_file in pre_mask_files:
        # 解析标记文件
        cur_pre_mask_file_name = cur_pre_mask_file.stem
        cur_pre_mask_file_split = str(cur_pre_mask_file_name).split('_')
        cur_label_file_name = cur_pre_mask_file_split[0]
        for name_index in range(5):
            cur_label_file_name = cur_label_file_name + '_' + cur_pre_mask_file_split[name_index + 1]
        cur_label_file_path = point_label_file_path / f'{cur_label_file_name}_heatmap.npy'
        # 导入二值预测掩膜
        pre_mask_gray = cv2.imread(str(cur_pre_mask_file), 0)
        ret, pre_mask_binary = cv2.threshold(pre_mask_gray, 127, 255, cv2.THRESH_BINARY)
        label_mask = np.load(str(cur_label_file_path))
        evaluator.evaluate(pre_mask_binary, label_mask)

    evaluator.summary()
