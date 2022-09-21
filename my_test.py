import shutil

import torch
from torchvision import transforms
import numpy as np
import os
import cv2
from PIL import Image
from skimage import io
from medpy import metric as mc

from models import UNetAddLayers
from utils import SegmentationMetric, Tool, split_raw, get_parse, paste_evaluation

args = get_parse()

model = UNetAddLayers.Unet(5, 1).cuda()
model.load_state_dict(torch.load(args.weight))
transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Resize(args.img_size),
])


def five_channel_test():
    if os.path.exists(args.img_save_path):
        shutil.rmtree(args.img_save_path)
    if os.path.exists(args.mask_save_path):
        shutil.rmtree(args.mask_save_path)

    os.mkdir(args.img_save_path)
    os.mkdir(args.mask_save_path)

    m_dice = 0
    m_pa = 0
    m_cpa = 0
    m_mpa = 0
    m_mIoU = 0
    i = 1
    num = len(os.listdir(args.test_data))
    for raw_file in os.listdir(args.test_data):
        p = args.test_data + raw_file
        img = io.imread(p)
        img = np.transpose(img, (1, 2, 0))
        all_sub_img, start_location = split_raw(img, overlap_size=64)
        pred_img = np.zeros(img.shape[:2])
        for idx, box in enumerate(start_location):
            x1, y1, x2, y2 = box
            img = np.array(all_sub_img[idx] / 255., np.float32)
            test_img = transform(img)[None]
            test_img = test_img.cuda()
            out = model(test_img)
            # print(out.shape)
            out = torch.reshape(out.cpu(), args.img_size)
            out = out.cpu().detach().numpy()
            # pred_img[y1:y2, x1:x2] = out
            pred_img[y1:y2, x1:x2] = cv2.bitwise_or(pred_img[y1:y2, x1:x2], out)
        pred_img = np.where(pred_img > args.threshold, 1, 0)
        # 把图片贴合回去展示
        # draw in mask
        save_img = Image.open(args.test_data + raw_file)
        # save_img = save_img.resize(args.img_size)
        mask_img_to_save = np.array(pred_img * 255, np.uint8)
        mask_img = cv2.cvtColor(mask_img_to_save, cv2.COLOR_GRAY2RGBA)

        h, w, c = mask_img.shape
        mask_img[:, :, 3] = 50
        mask = Image.fromarray(np.uint8(mask_img))
        b, g, r, a = mask.split()
        save_img.paste(mask, (0, 0, w, h), mask=a)
        save_img = np.array(save_img)
        # print(save_img)

        # cv2.imwrite(os.path.join(args.img_save_path, 'test_img_' + raw_file), save_img)
        # cv2.imwrite(os.path.join(args.mask_save_path, 'test_mask_' + raw_file.split('.')[0] + '.png'), mask_img_to_save)
        # print(raw_file + '  save res ok')
        # 模型评估
        metric = SegmentationMetric(args.class_number)
        imgPredict = np.array(pred_img.reshape(-1), np.uint8)
        imgLabel = cv2.imread(args.test_label + raw_file.split('.')[0] + '.png', 0)
        # imgLabel = cv2.resize(imgLabel, args.img_size)
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

        save_path = os.path.join(args.img_save_path, 'test_img_' + raw_file)
        paste_evaluation(save_img, mIoU, save_path)
        # cv2.imwrite(os.path.join(args.img_save_path, 'test_img_' + raw_file), save_img)
        cv2.imwrite(os.path.join(args.mask_save_path, 'test_mask_' + raw_file.split('.')[0] + '.png'), mask_img_to_save)
        print(raw_file + '  save res ok')

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


def model_test():
    # resize_test_img_and_mask()

    if os.path.exists(args.img_save_path):
        shutil.rmtree(args.img_save_path)
    if os.path.exists(args.mask_save_path):
        shutil.rmtree(args.mask_save_path)

    os.mkdir(args.img_save_path)
    os.mkdir(args.mask_save_path)

    # model = UNet.Unet(5, 1)
    # model = r2_unet.R2U_Net(5, 1)
    # model = r2_unet.R2U_Net(5, 1).cuda()
    # model = UNet.Unet(5, 1).cuda()
    # model.load_state_dict(torch.load(args.weight))

    # checkpoints = torch.load(last_weight_path)
    # model.load_state_dict(checkpoints['model_state_dict'])

    # print(model)
    all_pa = []
    all_mpa = []
    all_miou = []
    all_dice = []
    # all_dice2 = []

    model.eval()
    for file in os.listdir(args.test_data):
        img_file = os.path.join(args.test_data, file)
        img = io.imread(img_file)
        img = np.transpose(img[:5, :, :], (1, 2, 0))
        img = np.array(img / 255., np.float32)
        # img = transform(img).unsqueeze(0)
        img = transform(img)[None]
        img = img.cuda()

        out = model(img)
        # print(out)
        out = torch.reshape(out.cpu(), args.img_size)
        # out = (out).detach().numpy()
        out = out.cpu().detach().numpy()
        out = np.where(out > 0.5, 1, 0)
        # print(out)

        # draw in mask
        save_img = Image.open(args.test_data + file)
        save_img = save_img.resize(args.img_size)
        mask_img = np.array(out * 255, np.uint8)
        # print(mask_img)
        mask_img = cv2.cvtColor(mask_img, cv2.COLOR_GRAY2RGBA)

        h, w, c = mask_img.shape
        mask_img[:, :, 3] = 60
        mask = Image.fromarray(np.uint8(mask_img))
        b, g, r, a = mask.split()
        save_img.paste(mask, (0, 0, w, h), mask=a)
        save_img = np.array(save_img)
        # print(save_img)
        cv2.imwrite(os.path.join(args.img_save_path, 'test_img_' + file), save_img)
        cv2.imwrite(os.path.join(args.mask_save_path, 'test_mask_' + file.split('.')[0] + '.png'), mask_img)
        print(file + '  save res ok')

        # 评价指标
        metric = SegmentationMetric(2)

        imgPredict = np.array(out.reshape(-1), np.uint8)
        imgLabel = cv2.imread(args.test_label + file.split('.')[0] + '.png', 0)
        imgLabel = cv2.resize(imgLabel, args.img_size)
        imgLabel = np.array((imgLabel / 255).reshape(-1), np.int8)

        # print(imgPredict.shape)
        # print(mask_img.shape)
        metric.addBatch(imgPredict, imgLabel)
        ConfusionMatrix = metric.genConfusionMatrix(imgPredict, imgLabel)
        pa = metric.pixelAccuracy()
        cpa = metric.classPixelAccuracy()
        mpa = metric.meanPixelAccuracy()
        mIoU = metric.meanIntersectionOverUnion()
        dice = mc.binary.dc(imgPredict, imgLabel)
        # dice2 = Tool.dice_coeff(imgPredict, imgLabel)

        all_pa.append(pa)
        all_mpa.append(mpa)
        all_miou.append(mIoU)
        all_dice.append(dice)
        # all_dice2.append(dice2)

        print('\n\n=====================', file, ' 的评价指标： ')
        print('ConfusionMatrix is:')
        print(ConfusionMatrix)
        print('pa is : %f' % pa)
        print('cpa is :')
        print(cpa)
        print('mpa is : %f' % mpa)
        print('mIOu is : %f' % mIoU)
        print('dice is : %f' % dice)
        # print('dice2 is : %f' % dice2)
        # break
    print('**==**==' * 50)
    print('avg pa: ', sum(all_pa) / len(all_pa))
    print('avg mpa: ', sum(all_mpa) / len(all_mpa))
    print('avg miou: ', sum(all_miou) / len(all_miou))
    print('avg dice: ', sum(all_dice) / len(all_dice))
    # print('avg dice2: ', sum(all_dice2) / len(all_dice2))


def mv_test_mask_label():
    if os.path.exists(args.test_label):
        shutil.rmtree(args.test_label)
    os.mkdir(args.test_label)

    t = Tool()
    src1 = args.test_data
    src2 = r'D:\py_program\testAll\data_handle_all\segment_handle_data\data\mask/'
    dist = args.test_label
    t.copy_file_to_dir(src1, src2, dist)


if __name__ == '__main__':
    # mv_test_mask_label()
    # model_test()
    five_channel_test()
