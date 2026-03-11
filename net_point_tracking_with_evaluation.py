# 导入序列图片，输出序列网点，输出跟踪轨迹
from pathlib import Path
import cv2 as cv
import torch
from model import unet_with_seperate_conv
from utiles import user_def_transform
import torch.nn.functional as nnFun
from data_preprocess import compute_img_mean_and_variance
from utiles import file_related
from model import act_fn_config
import time
from scipy.spatial import KDTree
import numpy as np
import sys
from scipy.optimize import linear_sum_assignment
import os
import motmetrics as mm

#mot_metrics = ['idfp', 'idfn', 'idtp', 'idp', 'idr', 'idf1', 'num_unique_objects', 'mostly_tracked', 'partially_tracked', 'mostly_lost']
mot_metrics = mm.metrics.motchallenge_metrics
#track_let:[[ID],[[pt],[pt-2],[pt-3]...]]
#before_deteced_points:[ID,points]

def tracking_and_id_assignation_icp(before_detected_points_with_id, cur_detected_points, cur_max_id):
    before_detection_points = before_detected_points_with_id[:, 1:]
    cur_transformed_points = before_detection_points.copy()
    max_it_size = 5
    H1 = None
    kn_indices_first = None
    # ICP使前后两帧网点匹配
    size_icp_matched = 0
    for cur_it in range(max_it_size):
        kd_tree = KDTree(data=cur_transformed_points)
        dd, kn_indices_first = kd_tree.query(cur_detected_points, k=1)
        if cur_it == max_it_size - 1:
            break
        before_matched_points = cur_transformed_points[kn_indices_first]
        H1, H_mask1 = cv.findHomography(before_matched_points, cur_detected_point_centroids, method=cv.RANSAC,
                                        ransacReprojThreshold=5)
        cur_size_icp_matched = np.sum(H_mask1)
        if cur_size_icp_matched == size_icp_matched or cur_it == max_it_size - 1:
            break
        size_icp_matched = cur_size_icp_matched
        for tp_index in range(len(cur_transformed_points)):
            a_transformed_point = cur_transformed_points[tp_index]
            homo_a_transformed_point = np.array([a_transformed_point[0], a_transformed_point[1], 1])
            cur_transformed_points[tp_index] = (H1 @ homo_a_transformed_point.T)[0:2]
    # 根据最佳转换后的匹配误差过滤掉错误匹配
    cur_id = np.ones_like(kn_indices_first, dtype=np.int32)
    cur_before_id_map = np.ones_like(kn_indices_first, dtype=np.int32) * -1
    error_hs = []

    for row_index, kn_index in enumerate(kn_indices_first):
        a_cur_point = cur_detected_point_centroids[row_index]
        a_before_point = cur_transformed_points[kn_index]
        homo_cur_point = np.array([a_cur_point[0], a_cur_point[1], 1])
        homo_before_point = np.array([a_before_point[0], a_before_point[1], 1])
        H_error = np.linalg.norm(H1 @ homo_cur_point.T - homo_before_point)
        error_hs.append(H_error)
    error_hs = np.asarray(error_hs)
    neg_error_hs = -error_hs
    error_hs_aug = np.concatenate((neg_error_hs, error_hs), axis=0)
    std_h = np.std(error_hs_aug)

    for row_index, kn_index in enumerate(kn_indices_first):
        H_error = error_hs[row_index]
        if H_error < std_h:
            cur_id[row_index] = before_detected_points_with_id[kn_index][0]
            cur_before_id_map[row_index] = kn_index
        else:
            cur_id[row_index] = cur_max_id
            cur_max_id = cur_max_id + 1
    cur_id = np.expand_dims(cur_id, axis=1)
    cur_detected_points_with_id = np.concatenate((cur_id, cur_detected_points), axis=1)
    return cur_detected_points_with_id, cur_before_id_map, cur_max_id, cur_transformed_points


def tracking_and_id_assignation_using_fundamental_mat(before_detected_points_with_id, cur_detected_points, cur_max_id):
    before_detection_points = before_detected_points_with_id[:, 1:]
    cur_transformed_points = before_detection_points.copy()
    kd_tree = KDTree(data=before_detection_points)
    dd, kn_indices = kd_tree.query(cur_detected_points, k=1)
    kn_indices_first = kn_indices
    # kn_indices_first = kn_indices
    # ii表示query数据集中第i个点的最近点在原始数据集中的所有ii[i]
    # 为什么需要做对极几何验证？
    # 如果采用最近邻匹配，一定会出现错误匹配，例如超出视野范围的点，漏检的点，或者假阳性的点
    # 直接采用最近邻，然后通过阈值过滤没有依据，主要原因是阈值和机器人的移动速度有关系，不能设置单纯的阈值，而对极几何产生的偏差和移动速度无关，因此可以设置阈值来过滤
    # opencv直接提供了RANSAC的算法，用8点法来计算来计算基本矩阵，并且给出了哪些是outlier，哪些不是
    before_matched_points = before_detection_points[kn_indices_first]
    # test_dis = np.linalg.norm(cur_detected_point_centroids - before_matched_points, axis=1)
    F1, mask1 = cv.findFundamentalMat(cur_detected_point_centroids, before_matched_points, method=cv.FM_RANSAC,
                                      ransacReprojThreshold=1)
    cur_id = np.ones_like(kn_indices_first, dtype=np.int32)
    cur_before_id_map = np.ones_like(kn_indices_first, dtype=np.int32) * -1
    min_h_error = 10

    error_fs = []
    for row_index, kn_index in enumerate(kn_indices):
        a_cur_point = cur_detected_point_centroids[row_index]
        a_before_point = before_detection_points[kn_index]
        homo_cur_point = np.array([a_cur_point[0], a_cur_point[1], 1])
        homo_before_point = np.array([a_before_point[0], a_before_point[1], 1])
        error = np.abs(homo_before_point @ F1 @ homo_cur_point.T)
        error_fs.append(error)
    error_fs = np.asarray(error_fs)
    neg_error_fs = -error_fs
    error_fs_aug = np.concatenate((neg_error_fs, error_fs), axis=0)
    std_f = np.std(error_fs_aug)

    bef_idx_match_record = np.zeros((len(before_detected_points_with_id), 3), dtype=np.int32)
    for row_index, kn_index in enumerate(kn_indices):
        # 只考虑最近点
        a_error_f = error_fs[row_index]
        nn_d_error = dd[row_index]
        if a_error_f > std_f:
            cur_id[row_index] = cur_max_id
            cur_max_id = cur_max_id + 1
        else:
            if bef_idx_match_record[kn_index][0] == 0:
                cur_id[row_index] = before_detected_points_with_id[kn_index][0]
                cur_before_id_map[row_index] = kn_index
                bef_idx_match_record[kn_index][0] = 1
                bef_idx_match_record[kn_index][1] = row_index
                bef_idx_match_record[kn_index][2] = nn_d_error
            else:
                if bef_idx_match_record[kn_index][2] < nn_d_error:
                    cur_id[row_index] = cur_max_id
                    cur_max_id = cur_max_id + 1
                else:
                    cur_id[bef_idx_match_record[kn_index][1]] = cur_max_id
                    cur_before_id_map[bef_idx_match_record[kn_index][1]] = -1
                    cur_max_id = cur_max_id + 1
                    cur_id[row_index] = before_detected_points_with_id[kn_index][0]
                    cur_before_id_map[row_index] = kn_index
                    bef_idx_match_record[kn_index][0] = 1
                    bef_idx_match_record[kn_index][1] = row_index
                    bef_idx_match_record[kn_index][2] = nn_d_error


    cur_id = np.expand_dims(cur_id, axis=1)
    cur_detected_points_with_id = np.concatenate((cur_id, cur_detected_points), axis=1)
    return cur_detected_points_with_id, cur_before_id_map, cur_max_id, cur_transformed_points


def tracking_and_id_assignation_using_icp_and_fundamental_mat(before_detected_points_with_id, cur_detected_points,
                                                              cur_max_id):
    # 先用icp获得单应矩阵，对原始的前一个时刻点云进行变换
    # 然后求解本质矩阵
    before_detection_points = before_detected_points_with_id[:, 1:]
    cur_transformed_points = before_detection_points.copy()
    max_it_size = 5  # icp 5 次，应该对循环设置一个停止
    min_h_error_1 = 10
    min_h_error_2 = 15
    min_f_error = 10
    # ICP使前后两帧网点匹配
    size_icp_matched = 0
    first_icp_dd_mean = None
    first_icp_dd_std = None
    last_icp_dd_mean = None
    for cur_it in range(max_it_size):
        kd_tree_icp = KDTree(data=cur_transformed_points)
        dd, kn_indices_first = kd_tree_icp.query(cur_detected_points, k=1)
        if first_icp_dd_mean is None:
            first_icp_dd_mean = np.mean(dd)
            first_icp_dd_std = np.std(dd)
        before_matched_points = cur_transformed_points[kn_indices_first]
        H1, H_mask1 = cv.findHomography(before_matched_points, cur_detected_points, method=cv.RANSAC,
                                        ransacReprojThreshold=5)
        cur_size_icp_matched = np.sum(H_mask1)
        if cur_size_icp_matched == size_icp_matched or cur_it == max_it_size - 1:
            last_icp_dd_mean = np.mean(dd)
            break
        size_icp_matched = cur_size_icp_matched
        for tp_index in range(len(cur_transformed_points)):
            a_transformed_point = cur_transformed_points[tp_index]
            homo_a_transformed_point = np.array([a_transformed_point[0], a_transformed_point[1], 1])
            cur_transformed_points[tp_index] = (H1 @ homo_a_transformed_point.T)[0:2]
    nn_d_error_t = first_icp_dd_mean / 2
    kd_tree_fm = KDTree(data=cur_transformed_points)
    dd, kn_indices_first = kd_tree_fm.query(cur_detected_points, k=1)
    before_matched_points = before_detection_points[kn_indices_first]
    # test_dis = np.linalg.norm(cur_detected_point_centroids - before_matched_points, axis=1)
    F1, mask1 = cv.findFundamentalMat(cur_detected_points, before_matched_points, method=cv.FM_RANSAC,
                                      ransacReprojThreshold=5)
    cur_id = np.ones_like(kn_indices_first, dtype=np.int32)
    cur_before_id_map = np.ones_like(kn_indices_first, dtype=np.int32) * -1
    bef_idx_match_record = np.zeros((len(cur_transformed_points), 3), dtype=np.int32)
    for row_index, kn_index in enumerate(kn_indices_first):
        a_cur_point = cur_detected_points[row_index]
        a_before_point = before_detection_points[kn_index]
        a_before_transformed_point = cur_transformed_points[kn_index]
        homo_cur_point = np.array([a_cur_point[0], a_cur_point[1], 1])
        homo_before_point = np.array([a_before_point[0], a_before_point[1], 1])
        test_1 = homo_before_point @ F1 @ homo_cur_point.T
        error_f = np.abs(homo_before_point @ F1 @ homo_cur_point.T)
        error_h = np.linalg.norm(a_cur_point - a_before_transformed_point)
        if error_h < last_icp_dd_mean:
            if bef_idx_match_record[kn_index][0] == 0:
                cur_id[row_index] = before_detected_points_with_id[kn_index][0]
                cur_before_id_map[row_index] = kn_index
                bef_idx_match_record[kn_index][0] = 1
                bef_idx_match_record[kn_index][1] = row_index
                bef_idx_match_record[kn_index][2] = error_h
            else:
                if bef_idx_match_record[kn_index][2] < error_h:
                    cur_id[row_index] = cur_max_id
                    cur_max_id = cur_max_id + 1
                else:
                    cur_id[bef_idx_match_record[kn_index][1]] = cur_max_id
                    cur_before_id_map[bef_idx_match_record[kn_index][1]] = -1
                    cur_max_id = cur_max_id + 1
                    cur_id[row_index] = before_detected_points_with_id[kn_index][0]
                    cur_before_id_map[row_index] = kn_index
                    bef_idx_match_record[kn_index][0] = 1
                    bef_idx_match_record[kn_index][1] = row_index
                    bef_idx_match_record[kn_index][2] = error_h
                pass
            #print(f'small error_h : error_f is {error_f}')
        else:
            #print(f'large error_h : error_f is {error_f}')
            if error_h > first_icp_dd_mean + first_icp_dd_std:
                cur_id[row_index] = cur_max_id
                cur_max_id = cur_max_id + 1
            else:
                if error_f < 0.5:
                    if bef_idx_match_record[kn_index][0] == 0:
                        cur_id[row_index] = before_detected_points_with_id[kn_index][0]
                        cur_before_id_map[row_index] = kn_index
                        bef_idx_match_record[kn_index][0] = 1
                        bef_idx_match_record[kn_index][1] = row_index
                        bef_idx_match_record[kn_index][2] = error_h
                    else:
                        if bef_idx_match_record[kn_index][2] < error_h:
                            cur_id[row_index] = cur_max_id
                            cur_max_id = cur_max_id + 1
                        else:
                            cur_id[bef_idx_match_record[kn_index][1]] = cur_max_id
                            cur_before_id_map[bef_idx_match_record[kn_index][1]] = -1
                            cur_max_id = cur_max_id + 1
                            cur_id[row_index] = before_detected_points_with_id[kn_index][0]
                            cur_before_id_map[row_index] = kn_index
                            bef_idx_match_record[kn_index][0] = 1
                            bef_idx_match_record[kn_index][1] = row_index
                            bef_idx_match_record[kn_index][2] = error_h
                    # cur_id[row_index] = before_detected_points_with_id[kn_index][0]
                    # cur_before_id_map[row_index] = kn_index
                else:
                    cur_id[row_index] = cur_max_id
                    cur_max_id = cur_max_id + 1

    cur_id = np.expand_dims(cur_id, axis=1)
    cur_detected_points_with_id = np.concatenate((cur_id, cur_detected_points), axis=1)
    return cur_detected_points_with_id, cur_before_id_map, cur_max_id, cur_transformed_points


def tracking_and_id_assignation_using_icp_and_fundamental_mat_adp(before_detected_points_with_id, cur_detected_points,
                                                              cur_max_id):
    # 先用icp获得单应矩阵，对原始的前一个时刻点云进行变换
    # 然后求解本质矩阵
    before_detection_points = before_detected_points_with_id[:, 1:]
    cur_transformed_points = before_detection_points.copy()
    max_it_size = 5  # icp 5 次，应该对循环设置一个停止
    min_h_error_1 = 10
    min_h_error_2 = 15
    min_f_error = 10
    # ICP使前后两帧网点匹配
    size_icp_matched = 0
    first_icp_dd_mean = None
    #first_icp_dd_std = None
    #last_icp_dd_mean = None
    for cur_it in range(max_it_size):
        kd_tree_icp = KDTree(data=cur_transformed_points)
        dd, kn_indices_first = kd_tree_icp.query(cur_detected_points, k=1)
        if first_icp_dd_mean is None:
            first_icp_dd_mean = np.mean(dd)
            #first_icp_dd_std = np.std(dd)
        before_matched_points = cur_transformed_points[kn_indices_first]
        H1, H_mask1 = cv.findHomography(before_matched_points, cur_detected_points, method=cv.RANSAC,
                                        ransacReprojThreshold=5)
        cur_size_icp_matched = np.sum(H_mask1)
        if cur_size_icp_matched == size_icp_matched or cur_it == max_it_size - 1:
            #last_icp_dd_mean = np.mean(dd)
            break
        size_icp_matched = cur_size_icp_matched
        for tp_index in range(len(cur_transformed_points)):
            a_transformed_point = cur_transformed_points[tp_index]
            homo_a_transformed_point = np.array([a_transformed_point[0], a_transformed_point[1], 1])
            cur_transformed_points[tp_index] = (H1 @ homo_a_transformed_point.T)[0:2]
    nn_d_error_t = first_icp_dd_mean / 2
    kd_tree_fm = KDTree(data=cur_transformed_points)
    dd, kn_indices_first = kd_tree_fm.query(cur_detected_points, k=1)
    before_matched_points = before_detection_points[kn_indices_first]
    # test_dis = np.linalg.norm(cur_detected_point_centroids - before_matched_points, axis=1)
    F1, mask1 = cv.findFundamentalMat(cur_detected_points, before_matched_points, method=cv.FM_RANSAC,
                                      ransacReprojThreshold=5)
    cur_id = np.ones_like(kn_indices_first, dtype=np.int32)
    cur_before_id_map = np.ones_like(kn_indices_first, dtype=np.int32) * -1
    bef_idx_match_record = np.zeros((len(cur_transformed_points), 3), dtype=np.int32)
    error_fs = []
    error_hs = []
    for row_index, kn_index in enumerate(kn_indices_first):
        a_cur_point = cur_detected_points[row_index]
        a_before_point = before_detection_points[kn_index]
        a_before_transformed_point = cur_transformed_points[kn_index]
        homo_cur_point = np.array([a_cur_point[0], a_cur_point[1], 1])
        homo_before_point = np.array([a_before_point[0], a_before_point[1], 1])
        test_1 = homo_before_point @ F1 @ homo_cur_point.T
        error_f = np.abs(homo_before_point @ F1 @ homo_cur_point.T)
        error_h = np.linalg.norm(a_cur_point - a_before_transformed_point)
        error_fs.append(error_f)
        error_hs.append(error_h)
    error_fs = np.asarray(error_fs)
    error_hs = np.asarray(error_hs)
    neg_error_fs = -error_fs
    neg_error_hs = -error_hs
    error_fs_aug = np.concatenate((neg_error_fs, error_fs), axis=0)
    error_hs_aug = np.concatenate((neg_error_hs, error_hs), axis=0)
    # std_f = np.std(error_fs_aug)
    # std_h = np.std(error_hs_aug)
    std_f = np.std(error_fs)
    std_h = np.std(error_hs)
    std_rate = np.sqrt(2-np.pi/2)
    for row_index, kn_index in enumerate(kn_indices_first):
        error_f = error_fs[row_index]
        error_h = error_hs[row_index]
        if error_h < std_h/std_rate:
            # 满足强单应矩阵约束
            if bef_idx_match_record[kn_index][0] == 0:
                cur_id[row_index] = before_detected_points_with_id[kn_index][0]
                cur_before_id_map[row_index] = kn_index
                bef_idx_match_record[kn_index][0] = 1
                bef_idx_match_record[kn_index][1] = row_index
                bef_idx_match_record[kn_index][2] = error_h
            else:
                if bef_idx_match_record[kn_index][2] < error_h:
                    cur_id[row_index] = cur_max_id
                    cur_max_id = cur_max_id + 1
                else:
                    cur_id[bef_idx_match_record[kn_index][1]] = cur_max_id
                    cur_before_id_map[bef_idx_match_record[kn_index][1]] = -1
                    cur_max_id = cur_max_id + 1
                    cur_id[row_index] = before_detected_points_with_id[kn_index][0]
                    cur_before_id_map[row_index] = kn_index
                    bef_idx_match_record[kn_index][0] = 1
                    bef_idx_match_record[kn_index][1] = row_index
                    bef_idx_match_record[kn_index][2] = error_h
                pass
            #print(f'small error_h : error_f is {error_f}')
        else:
            #print(f'large error_h : error_f is {error_f}')
            if error_h > std_h/std_rate*1.665:
                # 连弱的最近点约束都不满足
                cur_id[row_index] = cur_max_id
                cur_max_id = cur_max_id + 1
            else:
                #  强单应矩阵约束
                if error_f < std_f/std_rate:
                    if bef_idx_match_record[kn_index][0] == 0:
                        cur_id[row_index] = before_detected_points_with_id[kn_index][0]
                        cur_before_id_map[row_index] = kn_index
                        bef_idx_match_record[kn_index][0] = 1
                        bef_idx_match_record[kn_index][1] = row_index
                        bef_idx_match_record[kn_index][2] = error_h
                    else:
                        if bef_idx_match_record[kn_index][2] < error_h:
                            cur_id[row_index] = cur_max_id
                            cur_max_id = cur_max_id + 1
                        else:
                            cur_id[bef_idx_match_record[kn_index][1]] = cur_max_id
                            cur_before_id_map[bef_idx_match_record[kn_index][1]] = -1
                            cur_max_id = cur_max_id + 1
                            cur_id[row_index] = before_detected_points_with_id[kn_index][0]
                            cur_before_id_map[row_index] = kn_index
                            bef_idx_match_record[kn_index][0] = 1
                            bef_idx_match_record[kn_index][1] = row_index
                            bef_idx_match_record[kn_index][2] = error_h
                    # cur_id[row_index] = before_detected_points_with_id[kn_index][0]
                    # cur_before_id_map[row_index] = kn_index
                else:
                    cur_id[row_index] = cur_max_id
                    cur_max_id = cur_max_id + 1

    cur_id = np.expand_dims(cur_id, axis=1)
    cur_detected_points_with_id = np.concatenate((cur_id, cur_detected_points), axis=1)
    return cur_detected_points_with_id, cur_before_id_map, cur_max_id, cur_transformed_points


class TrainParams:
    def __init__(self):
        # self.img_file_path = Path(
        #     r'H:\code\python\net_detection_and_tracking\data_0.1\wj-original_image_diff_moving_direction_29241030\img_all_size_uniformed')
        self.img_file_path = Path(
            r'H:\augmented_img')
        # self.img_file_path = Path(
        #     r'H:\code\python\net_detection_and_tracking\data_0.1\hard_detection_img_large_point\img_w278_h167')
        self.train_label_file_path = Path(
            r'\augmented_train_r6')
        self.validation_label_file_path = Path(
            r'H:\augmented_validation_r6')
        self.test_label_file_path = Path(
            r'H:\augmented_test_r6')
        self.train_results_display_save_path = Path(
            r'H:\train_results_display')
        self.valid_results_display_save_path = Path(
            r'H:\valid_results_display')
        self.test_results_display_save_path = Path(
            r'H:\est_results_display')
        self.ckp_save_path = Path(r'H:\ckp')
        # 最优模型保存路径
        self.best_model_save_path = Path(r'H:\ckp\best_models')
        # 测试时用的模型路径
        self.ckp_file_path = Path(
            r'H:\test-ccb-epoch=112-val_F1_measure=0.91.ckpt')
        self.epochs = 200
        self.batch_size = 8
        self.lr = 0.0005
        self.momentum = 0.1
        self.weight_decay = 5e-4
        self.factor = 0.75
        self.patience = 3
        self.min_lr = 0
        self.gamma = 0.1
        self.task_type = 'Test'
        self.act_fn = 'leaky_relu'
        self.n_classes = 2
        self.clf_threshold = 0.4
        self.unet_middle_channel_size = 2
        self.k_size_LSLR = 3
        self.net_work_type = 1
        self.error_threshold = 12
        self.use_gray_img = False
        self.bilinear = False
        self.dice_loss_rate = 0.2

        self.input_image_width = 960
        self.input_image_height = 576

        self.num_workers = 4
        self.precision = 32


if __name__ == '__main__':

    # 导入标记文件
    # 导入图片
    # 目标识别
    # 目标跟踪并赋予ID号
    # 计算各标记和各跟踪到的对象的距离
    # 跟新评价
    associated_point_dis_threshold = 100
    accs = []
    accs_names = []
    results_save_path = r'H:\tracking_results'
    label_file_path = Path(r'H:\tracking\label_2')
    ori_img_file_path = Path(r'H:\img_all_size_uniformed')
    net_point_seg_model_path = Path(r'H:\test-ccb-epoch=112-val_F1_measure=0.91.ckpt')

    for test_index in range(1, 17):
        accs_names.append(str(test_index))
        cur_train_params = TrainParams()
        act_fn_config.set_activate_fn(cur_train_params.act_fn)
        point_label_file_path = label_file_path/str(test_index)
        results_save_path_cur_idx = fr'{results_save_path}\test_{test_index}'
        if not os.path.exists(results_save_path_cur_idx):
            # 创建文件夹（支持多级目录创建）
            os.makedirs(results_save_path_cur_idx)
        data_transform = user_def_transform.ComposeOnlyImg([user_def_transform.ToTensorOnlyImg(),
                                                            user_def_transform.NormalizeOnlyImg(
                                                                compute_img_mean_and_variance.img_mean,
                                                                compute_img_mean_and_variance.img_variance)])

        net_point_seg_model = unet_with_seperate_conv.UNetWithSeperateConv(3, 2, cur_train_params)
        ck = torch.load(net_point_seg_model_path)["state_dict"]
        ck = {k.replace("network.", ""): v for k, v in ck.items()}
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net_point_seg_model.load_state_dict(ck)
        net_point_seg_model.to(device).eval()
        #模型预热
        dummy_input = torch.randn(1, 3, 960, 574).to('cuda')
        for index in range(3):
            dummy_pred = net_point_seg_model.forward(dummy_input)
        intput_c = 3
        point_label_files = file_related.get_filenames_of_path(point_label_file_path)
        before_detection_points_with_id = None
        cur_max_id = 0
        before_img_for_display = None
        tracklet_list = []
        acc = mm.MOTAccumulator(auto_id=True)
        accs.append(acc)
        for cur_point_label_file in point_label_files:
            label_file_name = cur_point_label_file.stem
            img_file_name = ori_img_file_path / (label_file_name + '.jpg')
            start_time = time.time()
            cur_img = cv.imread(str(img_file_name))
            img_height, img_width = cur_img.shape[:2]
            labels = file_related.read_json(cur_point_label_file)['labels']
            cur_label_points_with_ids = np.zeros((len(labels), 3))
            for label_index, a_label in enumerate(labels):
                cur_label_points_with_ids[label_index][0] = a_label['id']
                cur_label_points_with_ids[label_index][1] = a_label['points'][0]
                cur_label_points_with_ids[label_index][2] = a_label['points'][1]
            #和实际处理无关的代码放到和面，避免影响运行时间
            #和评价相关的代码放到后面
            transformed_img = data_transform(cur_img).to(device).unsqueeze(0)
            pred = net_point_seg_model.forward(transformed_img)
            end_time = time.time()
            #print(f"神经网络执行时间：{end_time - start_time}秒")
            heat_map_pred = nnFun.softmax(pred, dim=1)[0][1].detach().cpu().numpy()
            ret, binary_pre = cv.threshold(heat_map_pred, cur_train_params.clf_threshold, 255, cv.THRESH_BINARY)
            binary_pre = binary_pre.astype('uint8')
            # (1)连通域提取
            cur_detected_point_size, cur_detected_point_mask, cur_detected_point_stats, cur_detected_point_centroids = cv.connectedComponentsWithStats(
                binary_pre,
                connectivity=8)
            if before_detection_points_with_id is None:
                initial_ids = np.expand_dims(np.arange(cur_detected_point_size), axis=1)
                before_detection_points_with_id = np.concatenate((initial_ids, cur_detected_point_centroids), axis=1)
                cur_max_id = cur_detected_point_size
                before_img_for_display = cur_img.copy()
                cv.imwrite(fr'{results_save_path_cur_idx}/{label_file_name}_binary.jpg', binary_pre)
                cur_label_points = cur_label_points_with_ids[:, 1:]
                dis_matrix = mm.distances.norm2squared_matrix(cur_label_points, cur_detected_point_centroids,
                                                              max_d2=associated_point_dis_threshold)
                acc.update(
                    cur_label_points_with_ids[:, 0],  # Ground truth objects in this frame
                    before_detection_points_with_id[:, 0],  # Detector hypotheses in this frame
                    dis_matrix)
                ## 计算评价指标

                continue
            start_time = time.time()
            # 前后帧网点跟踪
            # cur_detected_points_with_id, cur_before_id_map, cur_max_id, before_transformed_points = (
            #     tracking_and_id_assignation_icp(before_detection_points_with_id, cur_detected_point_centroids, cur_max_id))
            cur_detected_points_with_id, cur_before_id_map, cur_max_id, before_transformed_points = (
                tracking_and_id_assignation_using_icp_and_fundamental_mat_adp(before_detection_points_with_id,
                                                                          cur_detected_point_centroids, cur_max_id))
            # cur_detected_points_with_id, cur_before_id_map, cur_max_id, before_transformed_points = (
            #     tracking_and_id_assignation_using_fundamental_mat(before_detection_points_with_id,
            #                                                               cur_detected_point_centroids, cur_max_id))
            # cur_detected_points_with_id, cur_before_id_map, cur_max_id, before_transformed_points = (
            #     tracking_and_id_assignation_m_icp(before_detection_points_with_id, cur_detected_point_centroids,
            #                                     cur_max_id))
            # 计算评价指标
            cur_label_points = cur_label_points_with_ids[:, 1:]
            dis_matrix = mm.distances.norm2squared_matrix(cur_label_points, cur_detected_point_centroids,
                                                          max_d2=associated_point_dis_threshold)
            acc.update(
                cur_label_points_with_ids[:, 0],  # Ground truth objects in this frame
                cur_detected_points_with_id[:, 0],  # Detector hypotheses in this frame
                dis_matrix)

            # for display
            before_detection_points = before_detection_points_with_id[:, 1:]
            for tracklet in tracklet_list:
                tracklet[1] = False
            new_tracklet_list = []
            for index in range(len(cur_before_id_map)):
                associated_index = cur_before_id_map[index]
                if associated_index == -1:
                    continue
                # 找到上一个索引
                before_index = associated_index
                matched_tracklet_index = False
                test_index = 0
                for tracklet in tracklet_list:
                    if tracklet[1]:
                        continue
                    test_index = test_index + 1
                    if tracklet[0] == before_index:
                        tracklet[0] = index
                        last_last_point = tracklet[2][-1]
                        cur_point = cur_detected_point_centroids[index]
                        last_point1 = before_detection_points[before_index]
                        last_point2 = tracklet[3]
                        # tracklet[2].append(before_detection_points[before_index])
                        # test
                        cur_to_last_dis = np.linalg.norm(cur_point - last_point1)
                        last_to_last_dis1 = np.linalg.norm(last_point1 - last_last_point)
                        last_to_last_dis2 = np.linalg.norm(last_point2 - last_last_point)
                        if last_to_last_dis1 != last_to_last_dis2:
                            pass
                            #print('haha')
                        # print('*********dis**********')
                        # print(f'cur_to_last_dis: {cur_to_last_dis}')
                        # print(f'last_to_last_dis1: {last_to_last_dis1}')
                        # print(f'last_to_last_dis2: {last_to_last_dis2}')
                        # test
                        tracklet[1] = True
                        tracklet[2].append(last_point2)
                        tracklet[3] = cur_detected_point_centroids[index]
                        matched_tracklet_index = True
                if matched_tracklet_index is False:
                    new_tracklet_list.append(
                        [index, True, [before_detection_points[before_index]], cur_detected_point_centroids[index]])
            # 移除未被匹配到的tracklet
            marked_index = []
            for tracklet_index, tracklet in enumerate(tracklet_list):
                if tracklet[1] is False:
                    marked_index.append(tracklet_index)
            for idx in reversed(marked_index):
                del tracklet_list[idx]
            tracklet_list = tracklet_list + new_tracklet_list
            # 仅显示上一个时刻匹配成功的点
            # for idx, a_cur_detected_point in enumerate(cur_detected_point_centroids):
            #     cv.circle(cur_img_display, (int(a_cur_detected_point[0]), int(a_cur_detected_point[1])), 2, (0, 0, 255), thickness=2)
            #     if mask[idx][0] == 1:
            #         a_before_matched_point = before_matched_points[idx]
            #         cv.circle(cur_img_display, (int(a_before_matched_point[0]), int(a_before_matched_point[1])), 2, (255, 0, 0), thickness=2)

            #display
            binary_pre = binary_pre.astype('uint8')
            binary_bgr = cv.cvtColor(binary_pre, cv.COLOR_GRAY2BGR)

            for a_points in before_transformed_points:
                cv.circle(binary_bgr, (int(a_points[0]), int(a_points[1])), 5, (0, 0, 255), thickness=-1)
            cv.imwrite(fr'{results_save_path_cur_idx}/{label_file_name}_binary.jpg', binary_bgr)

            cur_img_display = cur_img.copy()
            for a_tracklet in tracklet_list:
                traj_points = a_tracklet[2]
                cur_point = a_tracklet[3]
                cv.circle(cur_img_display, (int(cur_point[0]), int(cur_point[1])), 2, (255, 0, 0), thickness=2)
                # print(f'trajectory_size: {len(traj_points)}')
                start_point = cur_point
                if len(traj_points) > 0:
                    traj_len = 0
                    for a_point in reversed(traj_points):
                        cv.circle(cur_img_display, (int(a_point[0]), int(a_point[1])), 2, (0, 0, 255), thickness=2)
                        cv.line(cur_img_display, a_point.astype(np.int32), start_point.astype(np.int32), (0, 0, 255),
                                thickness=2)
                        start_point = a_point
                        traj_len = traj_len + 1
                        if traj_len > 3:
                            break
            before_detection_points_with_id = cur_detected_points_with_id
            cv.imwrite(fr'{results_save_path_cur_idx}/{label_file_name}_traj.jpg', cur_img_display)
            #cv.imwrite(str(results_save_path_cur_idx / f'{label_file_name}_e_line.jpg'), before_img_for_display)
            before_img_for_display = cur_img.copy()
        #计算单个文件夹评价分数
        # summary = mh.compute(acc, metrics=mm.metrics.motchallenge_metrics, name=f'test_{test_index}')
        # summary.to_csv(results_save_path_cur_idx / f'{label_file_name}_summary.csv', index=False)
        # print(summary)
        mh = mm.metrics.create()
        acc_count = len(point_label_files)
        acc_parts = [acc]
        acc_names = ['full']
        for acc_idx in range(1, acc_count):
            acc_parts.append(acc.events.loc[0:acc_idx])
            acc_names.append(f'part_{acc_idx}')
        summary = mh.compute_many(
            acc_parts,
            metrics=mot_metrics,
            names=acc_names)
        summary.to_csv(fr'{results_save_path_cur_idx}/{test_index}_summary.csv', index=False)
        #print(summary)
    # 计算所有文件夹评价分数
    mh_all = mm.metrics.create()
    summary = mh_all.compute_many(accs, metrics=mot_metrics, generate_overall=True)
    summary.to_csv(fr'{results_save_path}/all_summary.csv', index=False)
    print(summary)
    pass
