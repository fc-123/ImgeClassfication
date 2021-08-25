from torchsummary import summary
import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F


class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Flatten()
        self.fc2 = nn.Linear(32*29*29, 2)

        

    def forward(self, x):
        x = F.relu(self.conv1(x))     #(3,128,128)-->(16,124,124)
        x = self.pool1(x)             #(124,124)  -->(62,62)
        x = F.relu(self.conv2(x))     #(62,62)    -->(58,58)
        x = self.pool2(x)             #(58,58)    -->(29,29)
        x = x.view((x.shape[0], -1))                              
        x = self.fc2(x)              
        return x


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = MyNet().to(device)
    print(net)
    summary(net, (3, 128, 128))
    