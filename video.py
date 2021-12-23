import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

# from model_v2 import MobileNetV2
# from mynet import LeNet #模型1

from shufflenetV2_1_2 import ShuffleNetV2 #模型2

import cv2
import time
import numpy as np

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [
         transforms.ToPILImage(),
         transforms.Resize((128, 128)),
         # transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # load image
    # img_path = "G:/pycharm_pytorch171/data/img/test/img00012.jpg"
    # assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    # img = Image.open(img_path)
    # plt.imshow(img)
    # # [N, C, H, W]
    # img = data_transform(img)
    # # expand batch dimension
    # img = torch.unsqueeze(img, dim=0)
    #
    # # read class_indict
    # json_path = './class_indices.json'
    # assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)
    #
    # json_file = open(json_path, "r")
    # class_indict = json.load(json_file)

    #-----------------------------------------------------------------
    # 调用摄像头
    capture = cv2.VideoCapture("I:/data/normal1.avi")  # capture=cv2.VideoCapture("1.mp4")
    # fps = 0.0

    fps = capture.get(cv2.CAP_PROP_FPS)
    # 获取cap视频流的每帧大小
    size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))


    # 定义编码格式mpge-4
    fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', '2')
    # 定义视频文件输入对象
    outVideo = cv2.VideoWriter('./save_video/saveDir_Z90_shufflenetv2_10_normal.avi', fourcc, fps, size)

    while (True):
        t1 = time.time()
        ref, frame = capture.read()
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # frame = Image.fromarray(np.uint8(frame))  # 将array格式转化为Image格式

        # plt.imshow(frame)
        # frame = Image.fromarray(np.uint8(frame))
        frame_t = data_transform(frame)
        frame_t = torch.unsqueeze(frame_t, 0)

        # read class_indict
        json_path = './class_indices.json'
        assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

        json_file = open(json_path, "r")
        class_indict = json.load(json_file)

        # create model
        # model = LeNet().to(device) #模型1
        model = ShuffleNetV2([4, 8, 4], [6,12,24,48,96]).to(device) #模型2
        # load model weights
        model_weight_path = "./weight/shufflenetv2_Z90_10.pth"
        model.load_state_dict(torch.load(model_weight_path, map_location=device))
        model.eval()

        with torch.no_grad():
            # predict class
            output = torch.squeeze(model(frame_t.to(device))).cpu()
            predict = torch.softmax(output, dim=0)
            predict_cla = torch.argmax(predict).numpy()

        print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
                                                     predict[predict_cla].numpy())
        print(print_res)

        # output = model_ft(frame_t)
        # confidence, preds = torch.max(output, 1)
        # preds_name = class_names[int(preds)]

        fps = (fps + (1. / (time.time() - t1))) / 2
        # print('The predition is :{}, confidence is {:.4f}'.format(preds_name, confidence.item()))
        print("fps= %.2f" % (fps))

        # frame = cv2.cvtColor(np.array(frame_t), cv2.COLOR_RGB2BGR)



        # cv2.rectangle(frame, (10, 240), (715, 475), (0, 0, 255), 2)
        if (print_res.split(" ")[1] == "abnormal"):
            img = cv2.putText(frame, print_res, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
            outVideo.write(img)
            # cv2.imshow('2.jpg', img)
            # cv2.imwrite('wrong.jpg', img)
        elif (print_res.split(" ")[1] == "normal"):
            img = cv2.putText(frame, print_res, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
            outVideo.write(img)
            # cv2.imshow('2.jpg', img)
            # cv2.imwrite('right.jpg', img)

        # frame = cv2.putText(frame, 'The predition is :{}, confidence is {:.4f}'.format(preds_name, confidence.item()),
        #                     (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("video", img)
        # cv2.imwrite("test", img)
        c = cv2.waitKey(1) & 0xff
        if c == 27:
            capture.release()
            break

    #-----------------------------------------------------------------


    # # create model
    # model = mobilenet_v3_large(num_classes=2).to(device)
    # # load model weights
    # model_weight_path = "./weight/MobileNetV3_large110.pth"
    # model.load_state_dict(torch.load(model_weight_path, map_location=device))
    # model.eval()
    # with torch.no_grad():
    #     # predict class
    #     output = torch.squeeze(model(img.to(device))).cpu()
    #     predict = torch.softmax(output, dim=0)
    #     predict_cla = torch.argmax(predict).numpy()
    #
    # print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
    #                                              predict[predict_cla].numpy())
    # print(print_res)
    # # print(print_res.split(" ")[1] == )
    #
    # image = cv2.imread(img_path)
    # print(image.shape)  # 高480，宽720
    # # xmin:, xmax:, ymin:, ymax:
    # # cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
    # cv2.rectangle(image, (10, 240), (715, 475), (0, 0, 255), 2)
    # if (print_res.split(" ")[1] == "wrong"):
    #     img = cv2.putText(image, print_res, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
    #     cv2.imshow('2.jpg', img)
    #     cv2.imwrite('wrong.jpg', img)
    # else:
    #     img = cv2.putText(image, print_res, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
    #     cv2.imshow('2.jpg', img)
    #     cv2.imwrite('right.jpg', img)
    # # cv2.imshow('2.jpg', img)
    # # cv2.imwrite('right.jpg', img)
    # # cv2.imwrite('wrong.jpg', img)
    # cv2.waitKey(0)

    # plt.title(print_res)
    # print(print_res)
    # plt.show()


if __name__ == '__main__':
    main()
