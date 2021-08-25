import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from mynet import MyNet

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

   

    #-----------------------------------------------------------------
    # 调用摄像头
    capture = cv2.VideoCapture("I:/data/层数圈数识别/剪辑/正常/normal1.avi")  # capture=cv2.VideoCapture("1.mp4")
    # fps = 0.0

    fps = capture.get(cv2.CAP_PROP_FPS)
    # 获取cap视频流的每帧大小
    size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))


    # 定义编码格式mpge-4
    fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', '2')
    # 定义视频文件输入对象
    outVideo = cv2.VideoWriter('saveDir3.avi', fourcc, fps, size)

    while (True):
        t1 = time.time()
        ref, frame = capture.read()
        

        frame_t = data_transform(frame)
        frame_t = torch.unsqueeze(frame_t, 0)

        # read class_indict
        json_path = './class_indices.json'
        assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

        json_file = open(json_path, "r")
        class_indict = json.load(json_file)

        # create model
        model = LeNet().to(device)
        # load model weights
        model_weight_path = "./weight/mynet_20.pth"
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

      
        fps = (fps + (1. / (time.time() - t1))) / 2
        # print('The predition is :{}, confidence is {:.4f}'.format(preds_name, confidence.item()))
        # print("fps= %.2f" % (fps))

        if (print_res.split(" ")[1] == "0"):
            img = cv2.putText(frame, print_res, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
            outVideo.write(img)
           
        else:
            img = cv2.putText(frame, print_res, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
            outVideo.write(img)
            

        cv2.imshow("video", img)
        # cv2.imwrite("test", img)
        c = cv2.waitKey(30) & 0xff
        if c == 27:
            capture.release()
            break

if __name__ == '__main__':
    main()
