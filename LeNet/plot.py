from torchvision.datasets import FashionMNIST
from torchvision import transforms  # 处理数据集
from torch.utils import data
import matplotlib.pyplot as plt

train_data = FashionMNIST(
    root='data',
    train=True,
    download=True,
    transform=transforms.Compose([transforms.Resize(224), transforms.ToTensor()])
)

data_loader = data.DataLoader(
    train_data,
    batch_size=64,  # 将训练集数据分批，每批64张
    shuffle=True  # 打乱顺序
)

# b_x.shape torch.Size([64, 1, 224, 224]), 64张图，1通道，224*224
for step, (b_x, b_y) in enumerate(data_loader):
    if step > 0:
        break

batch_x = b_x.squeeze().numpy()  # 将四维张量移长度为1的维度，并转成numpy数组
batch_y = b_y.numpy()
class_label = train_data.classes
print(class_label)
if __name__ == '__main__':
    # 可视化一个Batch的图像
    plt.figure(figsize=(13, 5))  # 窗口大小
    for i in range(len(batch_y)):
        plt.subplot(4, 16, i + 1)  # 4行16列
        plt.imshow(batch_x[i], cmap='gray')
        plt.title(class_label[batch_y[i]], size=10)
        plt.axis('off')
        plt.subplots_adjust(wspace=0.05)
    plt.show()
