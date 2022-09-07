import shutil

import torch
from torchvision import transforms
import numpy as np
import os
import cv2
from PIL import Image
from skimage import io
import math
from medpy import metric as mc

from models import unet_all_conv
from utils import SegmentationMetric, Tool, split_raw, get_parse

args = get_parse()
# weight_path = './params/Unet2_zdm_ep400_BCE_640x640_selfResize_all_conv.pth'
# last_weight_path = './checkpoint_path/Unet1_zdm_ep400_BCELogic_640x640_AdamW_cheakpoint/unet_ep40.pth'
# #
# img_save_path = './save_img/Unet2_zdm_ep400_BCE_640x640_selfResize_all_conv/'
# mask_save_path = './test_res/'
# test_img_path = './test_img/'
# true_label_path = './test_label/'
#
img_save_path = r'D:\files\data\save_img/'
mask_save_path = r'D:\files\data\save_mask/'
test_img_path = r'D:\files\data\test_data/'
true_label_path = r'D:\files\data\test_mask/'

model = unet_all_conv.Unet(5, 1).cuda()
model.load_state_dict(torch.load(args.weight))
img_resize = (640, 640)
threshold = 0.5
transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Resize(img_resize),
    # transforms.RandomResizedCrop(img_resize)
])


def five_channel_test():
    if os.path.exists(img_save_path):
        shutil.rmtree(img_save_path)
    if os.path.exists(mask_save_path):
        shutil.rmtree(mask_save_path)

    os.makedirs(img_save_path)
    os.makedirs(mask_save_path)

    m_dice = 0
    m_pa = 0
    m_cpa = 0
    m_mpa = 0
    m_mIoU = 0
    i = 1
    num = len(os.listdir(test_img_path))
    for raw_file in os.listdir(test_img_path):
        p = test_img_path + raw_file
        img = io.imread(p)
        img = np.transpose(img, (1, 2, 0))
        all_sub_img, start_location = split_raw(img)
        pred_img = np.zeros(img.shape[:2])
        for idx, box in enumerate(start_location):
            x1, y1, x2, y2 = box
            img = np.array(all_sub_img[idx] / 255., np.float32)
            test_img = transform(img)[None]
            test_img = test_img.cuda()
            out = model(test_img)
            # print(out.shape)
            out = torch.reshape(out.cpu(), img_resize)
            out = out.cpu().detach().numpy()
            pred_img[y1:y2, x1:x2] = out
        pred_img = np.where(pred_img > threshold, 1, 0)
        # 把图片贴合回去展示
        # draw in mask
        save_img = Image.open(test_img_path + raw_file)
        # save_img = save_img.resize(img_resize)
        mask_img_to_save = np.array(pred_img * 255, np.uint8)
        mask_img = cv2.cvtColor(mask_img_to_save, cv2.COLOR_GRAY2RGBA)

        h, w, c = mask_img.shape
        mask_img[:, :, 3] = 60
        mask = Image.fromarray(np.uint8(mask_img))
        b, g, r, a = mask.split()
        save_img.paste(mask, (0, 0, w, h), mask=a)
        save_img = np.array(save_img)
        # print(save_img)
        cv2.imwrite(os.path.join(img_save_path, 'test_img_' + raw_file), save_img)
        cv2.imwrite(os.path.join(mask_save_path, 'test_mask_' + raw_file.split('.')[0] + '.png'), mask_img_to_save)
        print(raw_file + '  save res ok')
        # 模型评估
        metric = SegmentationMetric(2)
        imgPredict = np.array(pred_img.reshape(-1), np.uint8)
        imgLabel = cv2.imread(true_label_path + raw_file.split('.')[0] + '.png', 0)
        # imgLabel = cv2.resize(imgLabel, img_resize)
        imgLabel = np.array((imgLabel / 255).reshape(-1), np.int8)

        metric.addBatch(imgPredict, imgLabel)
        pa = metric.pixelAccuracy()
        cpa = metric.classPixelAccuracy()
        mpa = metric.meanPixelAccuracy()
        mIoU = metric.meanIntersectionOverUnion()
        dice = mc.binary.dc(imgPredict, imgLabel)

        print('\n\n', '**==**==' * 50)
        print(f'第{i}张测试图片')
        print('m_dice:', dice)
        print('像素准确率 PA is : %f' % pa)
        print('类别像素准确率 CPA is :', cpa)
        print('类别平均像素准确率 MPA is : %f' % mpa)
        print('mIoU is : %f' % mIoU, end='\n\n')

        m_pa += pa
        m_cpa += cpa
        m_mpa += mpa
        m_mIoU += mIoU
        m_dice += dice
        i += 1

    print('\n\n', '**==**==' * 50)
    print('m_dice:', m_dice / num)
    print('all 像素准确率 AVG_PA is : %f' % (m_pa / num))
    print('all 类别像素准确率 AVG_CPA is :', m_cpa / num)
    print('all 类别平均像素准确率 AVG_MPA is : %f' % (m_mpa / num))
    print('AVG_mIoU is : %f' % (m_mIoU / num), end='\n\n')


def mv_test_mask_label():
    if os.path.exists(true_label_path):
        shutil.rmtree(true_label_path)
    os.mkdir(true_label_path)

    t = Tool()
    src1 = test_img_path
    src2 = r'D:\py_program\testAll\data_handle_all\segment_handle_data\data\mask/'
    dist = true_label_path
    t.copy_file_to_dir(src1, src2, dist)


if __name__ == '__main__':
    # mv_test_mask_label()
    # model_test()
    five_channel_test()

# python train.py --data VOC.yaml --weights can_yolov5n.pt --img 640
# /train/exp/weights/best.pt
# runs/train/exp3/weights/best.pt
# runs/detect/exp4