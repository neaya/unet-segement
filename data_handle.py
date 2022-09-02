import numpy as np
from torch.utils.data import Dataset
import os
from torchvision import transforms
from torch.utils.data import DataLoader
import cv2
from skimage import io


class MyData(Dataset):
    def __init__(self, mask_path, img_path, img_size=(512, 512)):
        super(MyData, self).__init__()
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            # transforms.RandomResizedCrop(img_size),
            # transforms.Resize(img_size),
            # transforms.RandomHorizontalFlip()
        ])
        self.data_path = []
        self.mask_path = []
        for file in os.listdir(img_path):
            self.data_path.append(os.path.join(img_path, file))
        for file in os.listdir(mask_path):
            self.mask_path.append(os.path.join(mask_path, file))

    def __getitem__(self, index):
        img = io.imread(self.data_path[index])
        img = np.transpose(img[:5, :, :], (1, 2, 0))
        label = cv2.imread(self.mask_path[index], 0)
        data, label = np.array(img / 255, np.float32), np.array(label / 255, np.float32)
        data, label = self.transforms(img), self.transforms(label)

        return data, label

    def __len__(self):
        return len(self.data_path)


if __name__ == '__main__':
    train_img_path = r'D:\work\work_data\ZM\zm_jm\images/'
    train_label_path = r'D:\py_program\testAll\data_handle_all\segment_handle_data\data\mask_mut/'
    train_dataset = MyData(train_label_path, train_img_path, img_size=(640, 640))
    train_loader = DataLoader(train_dataset,
                              batch_size=2,
                              shuffle=True,
                              num_workers=2)
    for data, label in train_loader:
        print(data.shape, label.shape)
        break
