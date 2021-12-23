"""
@project:
@author:
@file:
@time:
@ide:
"""



# custom dataset class for albumentations library
import os
import random
import torch
from PIL import Image
from torchvision.datasets.folder import *
from matplotlib import pyplot as plt
import cv2
import numpy as np
import torch.utils.data as Data
import albumentations as A
from albumentations.pytorch import ToTensorV2

class MyFolder(DatasetFolder):
    def __init__(self, root: str, transform=None, target_transform=None, loader=default_loader, is_valid_file=None):
        super(MyFolder, self).__init__(root, loader, IMG_EXTENSIONS if is_valid_file is None else None,
                                          transform=transform,
                                          target_transform=target_transform,
                                          is_valid_file=is_valid_file)
        self.imgs = self.samples

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        # sample = self.loader(path)
        sample = cv2.imread(path)
        sample = cv2.cvtColor(sample, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            sample = self.transform(image=sample)["image"]
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target


train_transform = A.Compose(
    [
        # A.SmallestMaxSize(max_size=160),
        # A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
        A.Resize(height=320, width=320),
        # A.RandomCrop(height=128, width=128),
		# A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
		# A.RandomBrightnessContrast(p=0.5),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2(),
    ]
)
data_dir_train = "I:/data//train"
train_dataset = MyFolder(data_dir_train, transform=train_transform)
train_loader = Data.DataLoader(
        dataset=train_dataset,
        num_workers=0,
        batch_size=9,
        shuffle=True
    )


if __name__ == '__main__':
    #获取一个batch的图像，然后进行可视化
    #获取一个batch
    for step, (b_x, b_y) in enumerate(train_loader):
        if step > 0:
            break
    #可视化一个batch的图像
    batch_x = b_x.squeeze().numpy()
    batch_y = b_y.numpy()
    class_label = train_dataset.classes
    # class_label[0] = "T-shirt"
    plt.figure(figsize=(224, 224))
    for ii in np.arange(len(batch_y)):
        plt.subplot(3, 3, ii+1)
        plt.imshow(np.transpose(batch_x[ii].astype(np.uint8), (1, 2, 0))) #, cmap = plt.cm.gray
        plt.title(class_label[batch_y[ii]], size=9)
        plt.axis("off")             #不显示坐标轴
        plt.subplots_adjust(wspace=0.005)
        plt.show()