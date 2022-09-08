import torch
from torch import nn
from torch.utils.data import DataLoader
from torch import optim
from matplotlib import pyplot as plt
# import numpy as np
from datetime import datetime
# from pathlib import Path
# from torch.cuda.amp import GradScaler, autocast
import os
import gc

from data_handle import MyData
from models import UNetAddLayers
from utils import get_parse

gc.collect()
torch.cuda.empty_cache()
args = get_parse()

cur_model_name = args.weight.split('/')[-1].split('.')[0]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_dataset = MyData(args.train_label, args.train_data, img_size=(640, 640))
train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
# scaler = GradScaler()
go_on_weight = args.checkpoint_path + args.go_on_param
save_checkpoint_path = args.checkpoint_path + cur_model_name + '/'
checkpoints = torch.load(go_on_weight)
model = UNetAddLayers.Unet(5, 1).to(device)
model.load_state_dict(checkpoints['model_state_dict'])
# print(model)

loss_func = nn.BCELoss()
optimizer = optim.Adam(model.parameters())
optimizer.load_state_dict(checkpoints['optimizer_state_dict'])


# loss_func = nn.BCEWithLogitsLoss()
# optimizer = optim.AdamW(model.parameters())


def draw_loss(loss_list, epochs):
    x = [_ for _ in range(len(loss_list))]
    y = loss_list
    plt.figure()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    # plt.scatter(x, y)
    plt.legend(loc='best', labels=['loss'])
    plt.title('unet train loss curve')
    plt.plot(x, y, label='unet train loss curve')
    plt.savefig(os.path.join(args.train_loss_curve_save_path + cur_model_name, '_epoch_' + str(epochs) + '.png'))
    print(str(epochs) + ' loss save ok')
    plt.close()


def mkdir_path():
    # f = Path(__file__).resolve()
    # root = f.parents[0]
    if not os.path.exists(save_checkpoint_path):
        os.makedirs(save_checkpoint_path)
    if not os.path.exists(args.train_loss_curve_save_path + cur_model_name):
        os.makedirs(args.train_loss_curve_save_path + cur_model_name)


def train():
    model.train()
    best_loss = 0.0001
    train_loss_list = []
    for i in range(args.go_on_epoch, 1 + args.epochs):
        epoch_loss = 0
        step = 0
        for data, label in train_dataloader:
            optimizer.zero_grad()
            data, label = data.to(device), label.to(device)
            # with autocast():
            output = model(data)
            loss = loss_func(output, label)

            # scaler.scale(loss).backward()  # 将张量乘以比例因子，反向传播
            # scaler.step(optimizer)  # 将优化器的梯度张量除以比例因子。
            # scaler.update()  # 更新比例因子

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            step += 1
            if step % 5 == 0:
                print('[%s]: Epoch %d ======> global step %d/%d ===========> train_loss: %.6f ' % (
                    datetime.now(), i, step, len(train_dataloader), loss.item()))
            if best_loss > loss.item():
                torch.save(model.state_dict(), args.weight)
                best_loss = loss.item()
                print('cur best loss: ', best_loss, '\tsave ok')

        # 每20次epoch存一次断点
        if i % 20 == 0:
            checkpoint = {
                'epochs': i,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }
            torch.save(checkpoint, save_checkpoint_path + '_ep' + str(i) + '.pth')
            print('ep_', i, '_pth : =================== checkpoint save ok')

        avg_loss = epoch_loss / len(train_dataloader)
        print("Epoch: %d  ===========> Avg loss: %.4f " % (i, avg_loss))
        train_loss_list.append(avg_loss)
        if i % 5 == 0:
            draw_loss(train_loss_list, i)
        if i == args.epochs:
            torch.save(model.state_dict(), args.weight_last)
            print('last model save ok!')


if __name__ == '__main__':
    mkdir_path()
    train()
