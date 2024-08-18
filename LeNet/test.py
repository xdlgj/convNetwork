import torch
from torchvision.datasets import FashionMNIST
from torchvision import transforms

from model import LeNet


def test_data():
    data = FashionMNIST(
        root='./data',
        train=False,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(size=28)
        ])
    )
    dataloader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=True)
    return dataloader


def test_model(model, data):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    correct_rate = 0
    for img_data, label in data:
        img_data = img_data.to(device)
        # 开启测试模式
        model.eval()
        # 每个分类的占比
        outputs = model(img_data)
        pre_label = torch.argmax(outputs, dim=1)
        print(f"预测值：{pre_label} --- 实际值：{label}")
        correct_rate += torch.sum(pre_label == label)
    print(f"测试的准确率：{correct_rate / len(data)}")


if __name__ == '__main__':
    data = test_data()
    model = LeNet()
    # 加载模型
    model.load_state_dict(torch.load('model.pth', weights_only=False))
    test_model(model, data)