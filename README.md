## 构建自己的图像分类网络

-------
#### 1、项目介绍

构建自己的网络模型，实现参数较小的模型用于图像的分类，如图像的二分类。可用于嵌入式部署，模型较小，结构简单。

------

#### 2、环境

* pytorch 1.7+
* python 3.6+
* opencv-python == 3.4

-------

#### 3、数据准备

本文的数据结构为：

* classfication
  * data
    * train(训练集)
      * 0(为自己类别的名称)
      * 1(为自己类别的名称)
    * val(验证集)
      * 0(为自己类别的名称)
      * 1(为自己类别的名称)

* utils
  * dataset(数据增强)

* weight(模型保存)
* save_video
* mynet.py(网络1)
* shufflenetV2.py(网络2)
* train.py
* predict.py
* video.py(视频检测)

--------------

#### 4、模型参数与结构

mynet.py:

MyNet(
  (conv1): Conv2d(3, 16, kernel_size=(5, 5), stride=(1, 1))
  (pool1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (conv2): Conv2d(16, 32, kernel_size=(5, 5), stride=(1, 1))
  (pool2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (fc1): Flatten(start_dim=1, end_dim=-1)
  (fc2): Linear(in_features=26912, out_features=2, bias=True)

)

     Layer (type)     Output Shape         Param #
     ================================================
    Conv2d-1       [-1, 16, 124, 124]      1,216
    MaxPool2d-2    [-1, 16, 62, 62]        0
    Conv2d-3       [-1, 32, 58, 58]        12,832
    MaxPool2d-4    [-1, 32, 29, 29]        0
    Linear-5       [-1, 2]                 53,826
    ===================================================

Total params: 67,874
Trainable params: 67,874
Non-trainable params: 0
Input size (MB): 0.19
Forward/backward pass size (MB): 3.37
Params size (MB): 0.26
Estimated Total Size (MB): 3.82

----------------------------------

#### 5、关于项目

项目的地址：

[github]()

CSDN：

[CSDN博客]()



