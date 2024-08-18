import torch
from torch import nn
from torchsummary import summary


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # 1、卷积核: 5 * 5 * 6 填充2行2列默认值为0
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2)
        # 激活函数
        self.sig = nn.Sigmoid()
        # 2、持化层 2 * 2 步幅2
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        # 3、卷积核
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        # 4、持化层 2 * 2 步幅2
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        # 展平
        self.flatten = nn.Flatten()
        # 5、全链接层
        self.fc1 = nn.Linear(in_features= 16 * 5 * 5, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=80)
        self.fc3 = nn.Linear(in_features=80, out_features=10)

    def forward(self, x):
        """
        前向传播
        :param x: 输入
        :return:
        """
        x = self.sig(self.conv1(x))
        x = self.pool1(x)
        x = self.sig(self.conv2(x))
        x = self.pool1(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = LeNet().to(device)
    summary(model, (1, 28, 28))
