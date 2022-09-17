import argparse
import json
import time

import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
from datetime import datetime
import torch
import cv2
from skimage import io

"""
confusionMetric  #注意： 此处横着代表预测值，竖着代表真实值
P\L     P    N
P      TP    FP
N      FN    TN
"""


class SegmentationMetric(object):
    def __init__(self, numClass):
        self.numClass = numClass
        self.confusionMatrix = np.zeros((self.numClass,) * 2)

    def pixelAccuracy(self):
        # return all class overall pixel accuracy
        #  PA = acc = (TP + TN) / (TP + TN + FP + TN)
        acc = np.diag(self.confusionMatrix).sum() / self.confusionMatrix.sum()
        return acc

    def classPixelAccuracy(self):
        # return each category pixel accuracy(A more accurate way to call it precision)
        # acc = (TP) / TP + FP
        classAcc = np.diag(self.confusionMatrix) / self.confusionMatrix.sum(axis=1)
        return classAcc  # 返回的是一个列表值，如：[0.90, 0.80, 0.96]，表示类别1 2 3各类别的预测准确率

    def meanPixelAccuracy(self):
        classAcc = self.classPixelAccuracy()
        meanAcc = np.nanmean(classAcc)  # np.nanmean 求平均值，nan表示遇到Nan类型，其值取为0
        return meanAcc  # 返回单个值，如：np.nanmean([0.90, 0.80, 0.96, nan, nan]) = (0.90 + 0.80 + 0.96） / 3 =  0.89

    def meanIntersectionOverUnion(self):
        # Intersection = TP Union = TP + FP + FN
        # IoU = TP / (TP + FP + FN)
        intersection = np.diag(self.confusionMatrix)  # 取对角元素的值，返回列表
        union = np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) - np.diag(
            self.confusionMatrix)  # axis = 1表示混淆矩阵行的值，返回列表； axis = 0表示取混淆矩阵列的值，返回列表
        IoU = intersection / union  # 返回列表，其值为各个类别的IoU
        mIoU = np.nanmean(IoU)  # 求各类别IoU的平均
        return mIoU

    def genConfusionMatrix(self, imgPredict, imgLabel):  # 同FCN中score.py的fast_hist()函数
        # remove classes from unlabeled pixels in gt image and predict
        mask = (imgLabel >= 0) & (imgLabel < self.numClass)
        label = self.numClass * imgLabel[mask] + imgPredict[mask]
        count = np.bincount(label, minlength=self.numClass ** 2)
        confusionMatrix = count.reshape(self.numClass, self.numClass)
        return confusionMatrix

    def Frequency_Weighted_Intersection_over_Union(self):
        # FWIOU =     [(TP+FN)/(TP+FP+TN+FN)] *[TP / (TP + FP + FN)]
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix))
        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def addBatch(self, imgPredict, imgLabel):
        assert imgPredict.shape == imgLabel.shape
        self.confusionMatrix += self.genConfusionMatrix(imgPredict, imgLabel)

    def reset(self):
        self.confusionMatrix = np.zeros((self.numClass, self.numClass))


class Tool:
    def __int__(self):
        super(Tool, self).__int__()

    @staticmethod
    def copy_file_to_dir(src1, src2, dist):
        cp_file = []
        for file in os.listdir(src1):
            # copy file from src2/mask to dist/test_label
            cp_file.append(file.split('.')[0] + '.png')
        for file in os.listdir(src2):
            if file in cp_file:
                p = os.path.join(src2, file)
                shutil.copy(p, dist)
                print('[%s]: %s copy ok!' % (datetime.now(), file))

    @staticmethod
    def json2mask(image_path, json_path, mask_path, category_types: list[str]):
        """
        把json转成mask,二分类
        :param image_path: 原图路径
        :param json_path: json文件路径
        :param mask_path: 生成的label mask路径
        :param category_types: 类别，如果是多分类，一定要有Background这一项且一定要放在index为0的位置
        """
        color = [255 for _ in range(len(category_types))]
        print(len(category_types), ' 类')
        if not os.path.exists(mask_path):
            os.mkdir(mask_path)
            # 将图片标注json文件批量生成训练所需的标签图像png
            for img_path in os.listdir(image_path):
                img_name = img_path.split('.')[0]
                img = cv2.imread(os.path.join(image_path, img_path))
                h, w = img.shape[:2]
                # 创建一个大小和原图相同的空白图像
                mask = np.zeros([h, w, 1], np.uint8)

                with open(json_path + img_name + '.json', encoding='utf-8') as f:
                    label = json.load(f)

                shapes = label['shapes']
                for shape in shapes:
                    category = shape['label']
                    points = shape['points']
                    # 将图像标记填充至空白图像
                    points_array = np.array(points, dtype=np.int32)
                    mask = cv2.fillPoly(mask, [points_array], color[category_types.index(category)])

                # 生成的标注图像必须为png格式
                save_name = mask_path + img_name + '.png'
                cv2.imwrite(save_name, mask)
                print(save_name + '\tfinished')
                # break

    @staticmethod
    def dice_coeff(imgPredict, imgLabel):
        smooth = 1.
        intersection = (imgPredict * imgLabel).sum()
        return (2. * intersection + smooth) / (imgPredict.sum() + imgLabel.sum() + smooth)

    @staticmethod
    def get_mask_color(n):
        color_map = np.zeros([n, 3]).astype(np.uint8)
        print(color_map)
        color_map[0, :] = np.array([0, 0, 0])
        color_map[1, :] = np.array([244, 35, 232])
        color_map[2, :] = np.array([70, 70, 70])
        color_map[3, :] = np.array([102, 102, 156])
        color_map[4, :] = np.array([190, 153, 153])
        color_map[5, :] = np.array([153, 153, 153])

        color_map[6, :] = np.array([250, 170, 30])
        color_map[7, :] = np.array([220, 220, 0])
        color_map[8, :] = np.array([107, 142, 35])
        color_map[9, :] = np.array([152, 251, 152])
        color_map[10, :] = np.array([70, 130, 180])

        color_map[11, :] = np.array([220, 20, 60])
        color_map[12, :] = np.array([119, 11, 32])
        color_map[13, :] = np.array([0, 0, 142])
        color_map[14, :] = np.array([0, 0, 70])
        color_map[15, :] = np.array([0, 60, 100])

        color_map[16, :] = np.array([0, 80, 100])
        color_map[17, :] = np.array([0, 0, 230])
        color_map[18, :] = np.array([255, 0, 0])
        print(color_map)

        return color_map


class Colorize:
    '''
    这里实现了输入一张mask（output）,输出一个三通道的彩图。
    '''

    def __init__(self, n=19):
        self.cmap = Tool.get_mask_color(256)
        self.cmap[n] = self.cmap[-1]
        self.cmap = torch.from_numpy(self.cmap[:n])  # array->tensor

    def __call__(self, gray_image):
        size = gray_image.size()  # 这里就是上文的output
        color_image = torch.ByteTensor(3, size[0], size[1]).fill_(0)

        for label in range(0, len(self.cmap)):
            mask = gray_image == label
            color_image[0][mask] = self.cmap[label][0]
            color_image[1][mask] = self.cmap[label][1]
            color_image[2][mask] = self.cmap[label][2]

        return color_image


def split_raw(big_raw_img, imgsz=640, overlap_size=64):
    # 计算w方向的切割份数和重叠尺寸
    raw_size_hw = big_raw_img.shape[0:2]
    if raw_size_hw[0] < imgsz:
        n_row = 1
        overlap_h = 0
    else:
        for i in range(2, 20, 1):
            dh = (imgsz * i - raw_size_hw[0]) / (i - 1)
            if dh > overlap_size:
                n_row = i
                overlap_h = dh
                break

    if raw_size_hw[1] < imgsz:
        n_col = 1
        overlap_w = 0
    else:
        for j in range(2, 20, 1):
            dw = (imgsz * j - raw_size_hw[1]) / (j - 1)
            if dw > overlap_size:
                n_col = j
                overlap_w = dw
                break

    # 切割，获得每张图片的左上角点和右下角点
    top_list = []
    bottom_list = []
    for idx in range(n_row):
        top = int(round(idx * (imgsz - overlap_h)))
        down = int(round(min(imgsz + idx * (imgsz - overlap_h), raw_size_hw[0])))
        top_list.append(top)
        bottom_list.append(down)

    # 切割，获得每张图片的左上角点和右下角点
    left_list = []
    right_list = []
    for jdx in range(n_col):
        left = int(round(jdx * (imgsz - overlap_w)))
        right = int(round(min(imgsz + jdx * (imgsz - overlap_w), raw_size_hw[1])))
        left_list.append(left)
        right_list.append(right)

    all_images = []
    all_images_lt_rb = []
    real_overlap_size_wh = [overlap_w, overlap_h]
    for idx in range(n_row):
        for jdx in range(n_col):
            sub_img = big_raw_img[top_list[idx]:bottom_list[idx], left_list[jdx]:right_list[jdx]]
            all_images.append(sub_img)
            all_images_lt_rb.append((left_list[jdx], top_list[idx], right_list[jdx], bottom_list[idx]))

    return all_images, all_images_lt_rb


# 切割五通道的,注意f11标注的，可改成单通道的切割
def handle_dataset_resize(img_path, mask_path, save_img_dir, save_mask_dir):
    if os.path.exists(save_img_dir):
        shutil.rmtree(save_img_dir)
    if os.path.exists(save_mask_dir):
        shutil.rmtree(save_mask_dir)
    os.mkdir(save_img_dir)
    os.mkdir(save_mask_dir)

    for tiff in os.listdir(img_path):
        p = img_path + tiff
        # img = io.imread(p)
        # img = np.transpose(img[:5, :, :], (1, 2, 0))
        # img = np.transpose(img, (1, 2, 0))
        img = cv2.imread(p, 0)
        x, y = split_raw(img)

        for i, sub_img in enumerate(x):
            # sub_img = np.transpose(sub_img, (2, 0, 1))
            # print(sub_img.shape)
            # save_sub_img_path = save_img_dir + tiff.split('.')[0] + '_' + str(i) + '.tiff'
            save_sub_img_path = save_img_dir + tiff.split('.')[0] + '_' + str(i) + '.bmp'
            # io.imsave(save_sub_img_path, sub_img)
            cv2.imwrite(save_sub_img_path, sub_img)
            print(f'[{datetime.now()}] ====> {save_sub_img_path} \t save ok')

    for each_mask in os.listdir(mask_path):
        p = mask_path + each_mask
        mask = cv2.imread(p, 0)
        x, y = split_raw(mask)

        for i, sub_mask in enumerate(x):
            save_sub_mask_path = save_mask_dir + each_mask.split('.')[0] + '_' + str(i) + '.png'
            cv2.imwrite(save_sub_mask_path, sub_mask)
            print(f'[{datetime.now()}]====> {save_sub_mask_path}\t save ok')


def split_img_and_mask():
    raw_img_path = './dataset_dm_zm/'
    raw_mask_path = r'D:\py_program\testAll\data_handle_all\segment_handle_data\data\mask_mut/'
    new_img_path = './data/img/'
    new_mask_path = './data/mask/'
    handle_dataset_resize(raw_img_path, raw_mask_path, new_img_path, new_mask_path)
    print('split finished')


def resize_test_img_and_mask(
        test_img_path='./test_img/',
        true_label_path='./test_label/',
        after_resize_test_img_path='./test_resize_img/',
        after_resize_test_mask_path='./test_resize_mask/'
):
    # test_img_path = './test_img/',
    # true_label_path = './test_label/',
    # after_resize_test_img_path = './test_resize_img/',
    # after_resize_test_mask_path = './test_resize_mask/'
    if os.path.exists(after_resize_test_img_path):
        shutil.rmtree(after_resize_test_img_path)
    if os.path.exists(after_resize_test_mask_path):
        shutil.rmtree(after_resize_test_mask_path)
    os.mkdir(after_resize_test_mask_path)
    os.mkdir(after_resize_test_img_path)
    handle_dataset_resize(test_img_path, true_label_path, after_resize_test_img_path, after_resize_test_mask_path)
    print('resize test dataset ok')


def split_train_and_test():
    import os
    import shutil
    import random

    train_path = r'D:\files\data/train_data/'
    train_labels = r'D:\files\data/train_mask/'
    test_path = r'D:\files\data/test_data/'
    test_labels = r'D:\files\data/test_mask/'

    names = os.listdir(train_path)
    num = int(len(names) * 0.5)
    new_names = random.sample(names, num)
    # print(new_names)
    for name in new_names:
        shutil.move(os.path.join(train_path, name), os.path.join(test_path, name))
        shutil.move(os.path.join(train_labels, name[:-5] + '.tiff'), os.path.join(test_labels, name[:-5] + '.tiff'))


def clear_test_res():
    after_resize_test_img_path = './test_resize_img/'
    after_resize_test_mask_path = './test_resize_mask/'
    if os.path.exists(after_resize_test_img_path):
        shutil.rmtree(after_resize_test_img_path)
    if os.path.exists(after_resize_test_mask_path):
        shutil.rmtree(after_resize_test_mask_path)
    os.mkdir(after_resize_test_img_path)
    os.mkdir(after_resize_test_mask_path)


def move_file_from_dir1_to_dir2(dir1, dir2, dir3):
    '''
    :param dir1:全部都有
    :param dir2: 只有一部分
    :param dir3: dir1 - dir2
    '''
    d1, d2 = set(), handle_CAN_train_data(dir2)
    for file in os.listdir(dir1):
        d1.add(file)
    print(f'全部有 {len(d1)}张')
    d3 = d1 - d2
    for d in d3:
        shutil.move(dir1 + d, dir3 + d)
    print('移动完毕')


def handle_CAN_train_data(train_path_resized):
    train_set = set()
    for file in os.listdir(train_path_resized):
        # raw_file = file.split('.')[0][:-2] + '.bmp'
        raw_file = file.split('.')[0][:-2] + '.png'
        train_set.add(raw_file)
    print(f'train有 {len(train_set)} 张')
    return train_set


def paste_evaluation(img, m_iou, save_path):
    text = str(m_iou)
    left_down_location = (0, img.shape[0] - 25)
    # img = io.imread(img_path)
    # img = np.transpose(img[:5, :, :], (1, 2, 0))
    cv2.putText(img, text, left_down_location,
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1.5, color=(255, 255, 255), thickness=3,
                lineType=None, bottomLeftOrigin=None)
    cv2.imwrite(save_path, img)


def test_show_diff_pred_raw():
    raw_path = r'D:\files\data\test_mask'
    pred_path = r'D:\files\data\save_img'
    raw_list, pred_list = [], []
    for raw in os.listdir(raw_path):
        img_path = os.path.join(raw_path, raw)
        img = cv2.imread(img_path, 0)
        raw_list.append(img)
    for pred in os.listdir(pred_path):
        img_path = os.path.join(pred_path, pred)
        img = cv2.imread(img_path)
        # img = io.imread(img_path)
        # img = np.transpose(img[:5, :, :], (1, 2, 0))
        pred_list.append(img)

    for i in range(len(raw_list)):
        plt.figure(figsize=(18, 9))
        # plt.figure()
        plt.subplot(1, 2, 1)
        img = plt.imshow(raw_list[i])
        img.set_cmap('gray')

        plt.subplot(1, 2, 2)
        plt.imshow(pred_list[i])
        plt.show()
        # time.sleep(2)


def early_stop(avg_loss: list, num):
    if len(avg_loss) < num:
        return
    last_num_loss = avg_loss[len(avg_loss) - num:]
    last_avg_loss = sum(last_num_loss) / len(last_num_loss)


def get_parse():
    train_data = r'D:\py_program\testAll\segement\src\data\img/'
    train_label = r'D:\py_program\testAll\segement\src\data\mask/'
    test_data = r'D:\files\data\test_data/'
    test_label = r'D:\files\data\test_mask/'

    img_save_path = r'D:\files\data\save_img/'
    mask_save_path = r'D:\files\data\save_mask/'

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=300, help='model training epochs')
    parser.add_argument('--weight', type=str,
                        default='./params/UnetAddLayers_zdm_ep300_BCE_640x640_selfResize_best.pth',
                        help='weights path')
    parser.add_argument('--weight-last', type=str,
                        default='./params/UnetAddLayers_zdm_ep300_BCE_640x640_selfResize_last.pth',
                        help='last epoch weights path')
    parser.add_argument('--train-data', type=str, default=train_data, help='train data path')
    parser.add_argument('--train-label', type=str, default=train_label, help='train label path')
    parser.add_argument('--test-data', type=str, default=test_data, help='test data path')
    parser.add_argument('--test-label', type=str, default=test_label, help='test label path')
    parser.add_argument('--img_save_path', type=str, default=img_save_path, help='to save pred img')
    parser.add_argument('--mask_save_path', type=str, default=mask_save_path, help='to save pred mask')

    parser.add_argument('--train_loss_curve_save_path', type=str, default='./train_loss_pic/',
                        help='train loss curve save path')
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoint/',
                        help='checkpoint params save path')
    parser.add_argument('--go_on_epoch', type=int, default=100, help='checkpoint params epoch')
    parser.add_argument('--go_on_param', type=str, default='_ep100.pth',
                        help='checkpoint go on params')

    parser.add_argument('--img-size', nargs='+', type=int, default=(640, 640), help='image size')
    parser.add_argument('--batch-size', type=int, default=4, help='batch size')
    parser.add_argument('--input_channel', type=int, default=5, help='model input channels')
    parser.add_argument('--output_channel', type=int, default=1, help='model output channels')

    parser.add_argument('--threshold', type=float, default=0.5, help='object confidence threshold')
    parser.add_argument('--workers', type=int, default=4, help='maximum number of dataloader workers')
    parser.add_argument('--class_number', type=int, default=2, help='segement label class number')

    opt = parser.parse_args()
    # print(opt)
    return opt


if __name__ == '__main__':
    args = get_parse()
    print(args.weight.split('/')[-1].split('.')[0])
