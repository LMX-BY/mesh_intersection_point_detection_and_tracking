import cv2
import pytorch_lightning as pl
from data_loader import net_point_and_lane_data_loader
import torch
import torch.utils.data as data
from torchvision import transforms
import cv2 as cv
from model import unet_with_LRLS_conv
from model import unet_with_DW_LRLS_conv
from model import unet_with_DW_LRLS_short_cut
from model import ori_unet
from model import unet_with_shuffle_v2
from model import unet_with_LRLS_and_shuffle
from model import unet_with_LRLS_and_shuffle_full
from model import unet_with_seperate_conv
from model import act_fn_config
from loss import loss_combined_CE_and_dice
from pytorch_lightning import Trainer
import numpy as np
from pathlib import Path
from utiles import file_related
from utiles import user_def_transform
from model import weight_initialization
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import Callback
from data_preprocess import compute_img_mean_and_variance
from evaluation import net_point_evaluation
import torch.nn.functional as F
from torchvision import transforms
from thop import profile

# bgr
red = [0, 0, 255]
green = [0, 255, 0]
yellow = [0, 255, 255]
purple = [128, 0, 128]
blue = [255, 0, 0]


def overlay_mask_with_transparency(image, mask, color=(0, 255, 0), alpha=0.5):
    """
    将分割掩膜以半透明方式叠加到原始图片上

    参数:
        image: 原始BGR图像
        mask: 二值掩膜 (0和255或0和1)
        color: 掩膜颜色 (B, G, R)，默认为绿色
        alpha: 透明度 (0.0-1.0)，默认为0.5

    返回:
        叠加后的图像
    """
    # 确保图像是3通道
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # 确保掩膜是二值格式
    if mask.max() <= 1:
        mask = (mask * 255).astype(np.uint8)

    # 创建彩色掩膜
    colored_mask = np.zeros_like(image)
    colored_mask[:] = color

    # 创建掩膜区域 (0-1)
    mask_region = mask.astype(float) / 255.0

    # 将彩色掩膜与原始图像混合
    result = image.copy()

    # 对每个通道进行混合
    for c in range(3):
        result[:, :, c] = (image[:, :, c] * (1 - mask_region * alpha) +
                           colored_mask[:, :, c] * mask_region * alpha)

    return result.astype(np.uint8)

class NetPointsDetection(pl.LightningModule):
    def __init__(self, network, loss, train_params):
        super().__init__()
        self.network = network
        self.loss = loss
        self.train_params = train_params
        # 关闭结果图像保存
        self.save_train_display_results = False
        self.save_test_display_results = True
        self.save_valid_display_results = False
        self.train_display_interval = 50
        self.valid_display_interval = 50
        self.test_display_interval = 2
        self.validation_dice_score = 0
        self.test_dice_score = 0
        self.total_validation_sample_number = 0
        self.total_test_sample_number = 0
        self.best_validation_dice_score = 0
        #param
        self.n_classes = train_params.n_classes
        # evaluation
        self.error_threshold = train_params.error_threshold
        self.net_point_evaluator = net_point_evaluation.NetPointDetectionEvaluation()

    def training_step(self, batch, batch_idx):
        self.network.set_phase_train()
        trans_imgs, labels, file_names, ori_imgs = batch
        pred = self.network(trans_imgs)
        # test 构造背景全是1，前景全是0的假预测，看看loss会是多少
        # fake_pred = torch.zeros_like(pred)
        # for a_pred_fake in fake_pred:
        #     a_pred_fake[0].fill_(100)
        #     a_pred_fake[1].fill_(1)
        # test
        # pred_cut = pred[:, :, 20:100, 20:100]
        # headmaps_cut = headmaps[:, :, 20:100, 20:100]
        #loss = self.loss(pred_cut, headmaps_cut)
        labels = labels.to(device=self.device, dtype=torch.long)
        #fake_loss = self.loss(fake_pred, labels)
        loss = self.loss(pred, labels)
        print(f'loss: {loss}')
        self.log("train_loss", loss)
        if self.save_train_display_results and self.current_epoch % self.train_display_interval == 0:
            #self.save_heatmap(pred_cut, file_names)
            #self.save_gray_img(pred[:,1,:,:], file_names)
            #self.save_binary_img(pred, file_names, self.train_params.train_results_display_save_path)
            self.save_results_with_org_img(ori_imgs, pred, file_names,
                                           self.train_params.train_results_display_save_path)
        return loss

        pass

    def validation_step(self, batch, batch_idx):
        if batch_idx == 0:
            self.validation_dice_score = 0
            self.total_validation_sample_number = 0
            self.best_validation_dice_score = 0
            self.net_point_evaluator.reset()
        self.network.set_phase_eval()
        tran_imgs, labels, file_names, ori_imgs = batch
        pred = self.network(tran_imgs)
        pred_for_score_computing = pred.detach().clone()
        labels_for_score_computing = labels.detach().clone()
        labels_for_score_computing = labels_for_score_computing.to(device=self.device, dtype=torch.long)
        # pred_cut = pred[:, :, 20:100, 20:100]
        # headmaps_cut = headmaps[:, :, 20:100, 20:100]
        # loss = self.loss(pred_cut, headmaps_cut)
        assert labels.min() >= 0 and labels.max() <= 1, 'True mask indices should be in [0, 1]'
        dice_score = 0

        for index, a_label in enumerate(labels_for_score_computing):
            a_pred = pred_for_score_computing[index]
            mask_pred = (F.softmax(a_pred, dim=0) > self.train_params.clf_threshold).float()[1, :]
            mask_true = F.one_hot(a_label, self.n_classes).permute(0, 3, 1, 2).float()[0, 1, :]
            dice_score += loss_combined_CE_and_dice.dice_coeff_single_mask(mask_pred, mask_true)
            a_file_name = file_names[index]
            split_file_name = a_file_name.split('_')
            if len(split_file_name) == 8:
                error_rate = float(split_file_name[6][2:]) / 10
            else:
                error_rate = 1
            binary_pre = mask_pred.detach().cpu().numpy().astype('uint8')
            binary_true = mask_true.detach().cpu().numpy().astype('uint8')
            self.net_point_evaluator.evaluate(binary_pre, binary_true, self.error_threshold * error_rate)
            #detection_eva_type_map_list.append(cur_detection_eva_type_map)

        self.validation_dice_score = self.validation_dice_score + dice_score
        cur_batch_size = len(file_names)
        self.total_validation_sample_number = self.total_validation_sample_number + cur_batch_size
        if batch_idx == (self.trainer.num_val_batches[0] - 1):
            avg_dice_score = self.validation_dice_score / self.total_validation_sample_number
            net_points_detection_metrics = self.net_point_evaluator.summary()
            if avg_dice_score > self.best_validation_dice_score:
                self.best_validation_dice_score = avg_dice_score
            self.log("val_dice", avg_dice_score)
            self.log("val_F1_measure", net_points_detection_metrics["F1_measure"])
            print(f'\n*******************')
            print(f'test dice score: {avg_dice_score}')
            print(f'recall: {net_points_detection_metrics["recall"]}')
            print(f'precision: {net_points_detection_metrics["precision"]}')
            print(f'F1_measure: {net_points_detection_metrics["F1_measure"]}')
            print(f'total_test_sample_number is {self.total_test_sample_number}')
            print(f'*******************')

        if self.save_valid_display_results and self.current_epoch % self.valid_display_interval == 0:
            #self.save_heatmap(pred_cut, file_names)
            #self.save_gray_img(pred[:,1,:,:], file_names)
            #self.save_binary_img(pred, file_names, self.train_params.valid_results_display_save_path)
            self.save_results_with_org_img(ori_imgs, pred, file_names,
                                           self.train_params.valid_results_display_save_path)

    def test_step(self, batch, batch_idx):
        if batch_idx == 0:
            self.test_dice_score = 0
            self.total_test_sample_number = 0
            self.net_point_evaluator.reset()
        self.network.set_phase_eval()
        tran_imgs, labels, file_names, ori_imgs = batch
        pred = self.network(tran_imgs)
        pred_for_score_computing = pred.detach().clone()
        labels_for_score_computing = labels.detach().clone()
        labels_for_score_computing = labels_for_score_computing.to(device=self.device, dtype=torch.long)
        # pred_cut = pred[:, :, 20:100, 20:100]
        # headmaps_cut = headmaps[:, :, 20:100, 20:100]
        # loss = self.loss(pred_cut, headmaps_cut)
        assert labels.min() >= 0 and labels.max() <= 1, 'True mask indices should be in [0, 1]'
        dice_score = 0

        # compute dice score for each img
        detection_eva_type_map_list = []
        for index, a_label in enumerate(labels_for_score_computing):
            a_pred = pred_for_score_computing[index]
            mask_pred = (F.softmax(a_pred, dim=0) > self.train_params.clf_threshold).float()[1, :]
            mask_true = F.one_hot(a_label, self.n_classes).permute(0, 3, 1, 2).float()[0, 1, :]
            dice_score += loss_combined_CE_and_dice.dice_coeff_single_mask(mask_pred, mask_true)
            a_file_name = file_names[index]
            split_file_name = a_file_name.split('_')
            if len(split_file_name) == 8:
                error_rate = float(split_file_name[6][2:]) / 10
            else:
                error_rate = 1
            binary_pre = mask_pred.detach().cpu().numpy().astype('uint8')
            binary_true = mask_true.detach().cpu().numpy().astype('uint8')
            cur_detection_eva_type_map = self.net_point_evaluator.evaluate(binary_pre, binary_true,
                                                                           self.error_threshold * error_rate)
            detection_eva_type_map_list.append(cur_detection_eva_type_map)
        # compute network evaluation
        #self.net_point_evaluator.evaluate()

        self.test_dice_score = self.test_dice_score + dice_score
        cur_batch_size = len(file_names)
        self.total_test_sample_number = self.total_test_sample_number + cur_batch_size
        if batch_idx == (self.trainer.num_test_batches[0] - 1):
            avg_dice_score = self.test_dice_score / self.total_test_sample_number
            net_points_detection_metrics = self.net_point_evaluator.summary()
            print(f'\n*******************')
            print(f'test dice score: {avg_dice_score}')
            print(f'recall: {net_points_detection_metrics["recall"]}')
            print(f'precision: {net_points_detection_metrics["precision"]}')
            print(f'F1_measure: {net_points_detection_metrics["F1_measure"]}')
            print(f'total_test_sample_number is {self.total_test_sample_number}')
            print(f'*******************')

        if self.save_test_display_results and self.current_epoch % self.test_display_interval == 0:
            pass
            #self.save_heatmap(pred, file_names, self.train_params.test_results_display_save_path)
            # self.save_gray_img(pred[:,1,:,:], file_names)
            # self.save_binary_img(pred, file_names, self.train_params.test_results_display_save_path)
            # self.save_results_with_detection_eva_type_mask(ori_imgs, labels_for_score_computing,
            #                                                detection_eva_type_map_list, file_names,
            #                                                self.train_params.test_results_display_save_path)
            # self.save_results_with_org_img(ori_imgs, pred, file_names,
            #                                self.train_params.test_results_display_save_path)
            self.save_results_with_org_img_pro(ori_imgs, pred, file_names,
                                           self.train_params.test_results_display_save_path)
            # self.save_results_with_label_mask(labels_for_score_computing, pred, file_names,
            #                                self.train_params.test_results_display_save_path)

    def configure_optimizers(self):
        non_frozen_parameters = [p for p in self.network.parameters() if p.requires_grad]
        optimizer = torch.optim.RMSprop(non_frozen_parameters,
                                        lr=self.train_params.lr, weight_decay=self.train_params.weight_decay,
                                        momentum=self.train_params.momentum, foreach=True)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max",
                                                                  factor=self.train_params.factor,
                                                                  patience=self.train_params.patience,
                                                                  min_lr=self.train_params.min_lr)

        # return optimizer

        # ??"monitor": "train_loss",是什麽作用？
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
            "monitor": "train_loss",
        }

    def save_heatmap(self, heatmap, file_names, path):
        for i in range(len(file_names)):
            file_name = file_names[i]
            save_path = Path(path) / f'{file_name}_heatmap_display_epoch_{self.trainer.current_epoch}.png'
            heatmap_origin = heatmap[i][1].detach().cpu().numpy()
            heatmap_normalized = cv.normalize(heatmap_origin, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX,
                                              dtype=cv.CV_8U)
            heatmap_heatmap = cv.applyColorMap(heatmap_normalized, cv.COLORMAP_HOT)
            # normalized_heatmap = cv.normalize(heatmap_for_display, None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX,
            #                                   dtype=cv.CV_32F)
            cv.imwrite(str(save_path), heatmap_heatmap)

    def save_gray_img(self, heatmap, file_names, path):
        for i in range(len(file_names)):
            file_name = file_names[i]
            save_path = Path(path) / f'{file_name}_output_display_epoch_{self.trainer.current_epoch}.png'
            heatmap_origin = heatmap[i].squeeze(0).detach().cpu().numpy()
            heatmap_normalized = cv.normalize(heatmap_origin, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX,
                                              dtype=cv.CV_8U)
            cv.imwrite(str(save_path), heatmap_normalized)

    def save_binary_img(self, heatmap, file_names, path):
        s_heatmap = F.softmax(heatmap, dim=1)
        for i in range(len(file_names)):
            file_name = file_names[i]
            save_path = Path(path) / f'{file_name}_output_display_epoch_{self.trainer.current_epoch}.png'
            heatmap_origin = s_heatmap[i][1].detach().cpu().numpy()
            ret, t1 = cv.threshold(heatmap_origin, self.train_params.clf_threshold, 255, cv.THRESH_BINARY)
            # heatmap_normalized = cv.normalize(heatmap_origin, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX,
            #                                   dtype=cv.CV_8U)
            cv.imwrite(str(save_path), t1)

    def save_results_with_org_img(self, ori_imgs, pre_labels, file_names, path):
        s_pre_labels = F.softmax(pre_labels, dim=1)
        for i in range(len(file_names)):
            file_name = file_names[i]
            save_path = Path(path) / f'{file_name}_output_display_epoch_{self.trainer.current_epoch}.png'
            a_pre_labels = s_pre_labels[i][1].detach().cpu().numpy()
            a_ori_img = ori_imgs[i]
            ret, t1 = cv.threshold(a_pre_labels, self.train_params.clf_threshold, 255, cv.THRESH_BINARY)
            colored_t1 = np.zeros_like(a_ori_img, np.uint8)
            mask = (t1 == 255)
            colored_t1[mask, 2] = 255
            comb_img = cv2.addWeighted(a_ori_img, self.train_params.clf_threshold, colored_t1, 0.5, 0)
            cv.imwrite(str(save_path), comb_img)
        pass

    def save_results_with_org_img_pro(self, ori_imgs, pre_labels, file_names, path):
        s_pre_labels = F.softmax(pre_labels, dim=1)
        for i in range(len(file_names)):
            file_name = file_names[i]
            save_path = Path(path) / f'{file_name}_output_display_epoch_{self.trainer.current_epoch}.png'
            a_pre_labels = s_pre_labels[i][1].detach().cpu().numpy()
            a_ori_img = ori_imgs[i]
            ret, t1 = cv.threshold(a_pre_labels, self.train_params.clf_threshold, 255, cv.THRESH_BINARY)
            colored_t1 = np.zeros_like(a_ori_img, np.uint8)
            mask = (t1 == 255)
            comb_img = overlay_mask_with_transparency(a_ori_img, mask, (0, 0, 255))
            # colored_t1[mask, 2] = 255
            # comb_img = cv2.addWeighted(a_ori_img, self.train_params.clf_threshold, colored_t1, 0.5, 0)
            cv.imwrite(str(save_path), comb_img)
        pass

    def save_results_with_label_mask(self, label_map, pre_labels, file_names, path):
        s_pre_labels = F.softmax(pre_labels, dim=1)
        img_width = pre_labels.shape[3]
        img_height = pre_labels.shape[2]
        for i in range(len(file_names)):
            file_name = file_names[i]
            save_path = Path(path) / f'{file_name}_output_display_epoch_{self.trainer.current_epoch}_mask.png'
            a_pre_labels = s_pre_labels[i][1].detach().cpu().numpy()
            a_mask = label_map[i][0].detach().cpu().numpy()
            ret, t_pre = cv.threshold(a_pre_labels, 0.5, 255, cv.THRESH_BINARY)
            colored_pre = np.zeros((img_height, img_width, 3), np.uint8)
            colored_mask = np.zeros((img_height, img_width, 3), np.uint8)
            m_pre = (t_pre == 255)
            m_mask = (a_mask == 1)
            colored_pre[m_pre, 2] = 255
            colored_mask[m_mask, :] = [255, 255, 255]
            comb_img = cv2.addWeighted(colored_mask, 0.5, colored_pre, 0.5, 0)
            cv.imwrite(str(save_path), comb_img)

    def save_results_with_detection_eva_type_mask(self, ori_imgs, label_map, det_eva_type_map, file_names, path):
        img_width = label_map.shape[3]
        img_height = label_map.shape[2]
        for i in range(len(file_names)):
            file_name = file_names[i]
            save_path = Path(
                path) / f'{file_name}_output_display_epoch_{self.trainer.current_epoch}_with_det_eva_type.png'
            a_eva_type_map = det_eva_type_map[i]
            a_ori_img = ori_imgs[i]
            a_label_map = label_map[i][0].detach().cpu().numpy()
            colored_eva_type_map = np.zeros((img_height, img_width, 3), np.uint8)
            colored_label_map = np.zeros((img_height, img_width, 3), np.uint8)
            m_tp = (a_eva_type_map == net_point_evaluation.TP_type)
            m_fp = (a_eva_type_map == net_point_evaluation.FP_type)
            m_label = (a_label_map == 1)
            colored_eva_type_map[m_tp, :] = blue
            colored_eva_type_map[m_fp, :] = red
            colored_label_map[m_label, :] = green
            comb_img_1 = cv2.addWeighted(a_ori_img, 0.5, colored_eva_type_map, 0.5, 0)
            comb_img_2 = cv2.addWeighted(a_ori_img, 0.5, colored_label_map, 0.5, 0)
            comb_img_1_2 = cv2.addWeighted(comb_img_1, 0.5, comb_img_2, 0.5, 0)
            cv.imwrite(str(save_path), comb_img_1_2)


class ModelSizeAndFlopsCallback(Callback):
    def __init__(self, param):
        self.use_gray = param.use_gray_img
        pass

    def on_test_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        cur_model = pl_module.network
        params = pl_module.train_params
        if self.use_gray:
            c = 1
        else:
            c = 3
        input = torch.randn(1, c, params.input_image_height, params.input_image_width).to(pl_module.device)
        flops, params = profile(cur_model, inputs=(input,))
        print(f"Model FLOPS: {flops / 1000000000}, Model Params: {params / 1000000}")
        print('test finished !!!!!!!!!!!!!!!!!!!')


class TrainParams:
    def __init__(self):
        # self.img_file_path = Path(
        #     r'H:\code\python\net_detection_and_tracking\data_0.1\wj-original_image_diff_moving_direction_29241030\img_all_size_uniformed')
        self.img_file_path = Path(
            r'H:\augmented_img')
        # self.img_file_path = Path(
        #     r'H:\code\python\net_detection_and_tracking\data_0.1\hard_detection_img_large_point\img_w278_h167')
        self.train_label_file_path = Path(
            r'H:\augmented_train_r6')
        self.validation_label_file_path = Path(
            r'H:\augmented_validation_r6')
        self.test_label_file_path = Path(
            r'H:\augmented_test_r6')
        self.train_results_display_save_path = Path(
            r'H:\train_results_display')
        self.valid_results_display_save_path = Path(
            r'H:\\valid_results_display')
        self.test_results_display_save_path = Path(
            r'H:\test_results_display')
        self.ckp_save_path = Path(r'H:\ckp')
        # 最优模型保存路径
        self.best_model_save_path = Path(r'H:\best_models')
        # 测试时用的模型路径
        self.ckp_file_path = None
        self.epochs = 250
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
        #self.act_fn = 'relu'
        self.n_classes = 2
        self.clf_threshold = 0.4
        self.unet_middle_channel_size = 64
        self.k_size_LSLR = 3
        self.net_work_type = 1
        self.error_threshold = 12
        self.use_gray_img = False
        self.bilinear = False
        self.dice_loss_rate = 0

        self.input_image_width = 960
        self.input_image_height = 576

        self.num_workers = 4
        self.precision = 32


if __name__ == '__main__':
    cur_train_params = TrainParams()
    print(f'\n...................key params................\n')
    print(f'net_work_type is {cur_train_params.net_work_type}')
    print(f'unet_middle_channel_size is {cur_train_params.unet_middle_channel_size}')
    print(f'k_size_LSLR is {cur_train_params.k_size_LSLR}')
    print(f'batch_size is {cur_train_params.batch_size}')
    print(f'dice_loss_rate is {cur_train_params.dice_loss_rate}')
    # dataset
    if cur_train_params.use_gray_img:
        data_transform = {
            "train": user_def_transform.Compose([user_def_transform.Grayscale(),
                                                 user_def_transform.ToTensorWithDataAndTarget(),
                                                 user_def_transform.Normalize(
                                                     compute_img_mean_and_variance.img_gray_mean,
                                                     compute_img_mean_and_variance.img_gray_variance)]),
            "val": user_def_transform.Compose([user_def_transform.Grayscale(),
                                               user_def_transform.ToTensorWithDataAndTarget(),
                                               user_def_transform.Normalize(
                                                   compute_img_mean_and_variance.img_gray_mean,
                                                   compute_img_mean_and_variance.img_gray_variance)])
        }
        intput_c = 1
    else:
        data_transform = {
            "train": user_def_transform.Compose([user_def_transform.ToTensorWithDataAndTarget(),
                                                 user_def_transform.Normalize(
                                                     compute_img_mean_and_variance.img_mean,
                                                     compute_img_mean_and_variance.img_variance)]),
            "val": user_def_transform.Compose([user_def_transform.ToTensorWithDataAndTarget(),
                                               user_def_transform.Normalize(
                                                   compute_img_mean_and_variance.img_mean,
                                                   compute_img_mean_and_variance.img_variance)])
        }
        intput_c = 3

    train_label_files = file_related.get_filenames_of_path(cur_train_params.train_label_file_path)
    valid_label_files = file_related.get_filenames_of_path(cur_train_params.validation_label_file_path)
    test_label_files = file_related.get_filenames_of_path(cur_train_params.test_label_file_path)

    train_dataset = net_point_and_lane_data_loader.NetPointAndLaneDataLoader(cur_train_params.img_file_path,
                                                                             train_label_files, data_transform["train"])
    validation_dataset = net_point_and_lane_data_loader.NetPointAndLaneDataLoader(cur_train_params.img_file_path,
                                                                                  valid_label_files,
                                                                                  data_transform["val"])
    test_dataset = net_point_and_lane_data_loader.NetPointAndLaneDataLoader(cur_train_params.img_file_path,
                                                                            test_label_files, data_transform["val"])

    train_data_loader = data.DataLoader(train_dataset, cur_train_params.batch_size, shuffle=True,
                                        num_workers=cur_train_params.num_workers,
                                        collate_fn=net_point_and_lane_data_loader.detection_collate)
    valid_data_loader = data.DataLoader(validation_dataset, cur_train_params.batch_size, shuffle=False,
                                        num_workers=cur_train_params.num_workers,
                                        collate_fn=net_point_and_lane_data_loader.detection_collate)
    test_data_loader = data.DataLoader(test_dataset, 1, shuffle=False,
                                       num_workers=cur_train_params.num_workers,
                                       collate_fn=net_point_and_lane_data_loader.detection_collate)

    # model
    act_fn_config.set_activate_fn(cur_train_params.act_fn)
    unet = unet_with_seperate_conv.UNetWithSeperateConv(intput_c, 2, cur_train_params)
    loss = loss_combined_CE_and_dice.LossCombinedCEAndDice(cur_train_params.n_classes, cur_train_params.dice_loss_rate)
    unet.apply(weight_initialization.init_kaiming_normal)

    # checkpoint
    # 输出GFLOPS和模型参数大小的回调函数
    best_model_path = file_related.create_folder_with_cur_time_info(cur_train_params.best_model_save_path,
                                                                    'best_model')
    checkpoint_callback = ModelCheckpoint(monitor='val_F1_measure',
                                          dirpath=best_model_path,
                                          filename='test-ccb-{epoch:02d}-{val_F1_measure:.2f}',
                                          save_top_k=5,
                                          mode='max',
                                          save_last=True)
    model_size_and_flops_callback = ModelSizeAndFlopsCallback(cur_train_params)

    # training task
    net_point_detection_task = NetPointsDetection(unet, loss, cur_train_params)
    ssd_key_pts_trainer = Trainer(
        accelerator="gpu",
        precision=cur_train_params.precision,  # try 16 with enable_pl_optimizer=False
        # callbacks=[checkpoint_callback, learningrate_callback, early_stopping_callback],
        callbacks=[checkpoint_callback, model_size_and_flops_callback],
        default_root_dir=str(cur_train_params.ckp_save_path),  # where checkpoints are saved to
        # logger=neptune_logger,
        log_every_n_steps=1,
        num_sanity_val_steps=0,
        max_epochs=cur_train_params.epochs,
    )
    # 训练流程：训练的时候为train，测试的时候为Test
    if cur_train_params.task_type == "Train":
        ssd_key_pts_trainer.fit(
            model=net_point_detection_task, train_dataloaders=train_data_loader
            , val_dataloaders=valid_data_loader
            , ckpt_path=str(cur_train_params.ckp_file_path)
        )
    else:
        ssd_key_pts_trainer.test(model=net_point_detection_task, ckpt_path=cur_train_params.ckp_file_path,
                                 dataloaders=test_data_loader)
    pass
