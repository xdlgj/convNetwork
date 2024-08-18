import copy
import time

import torch
from torchvision.datasets import FashionMNIST
from torchvision import transforms
from torch import utils
import pandas as pd
import matplotlib.pyplot as plt

from LeNet.model import LeNet


def train_val_data_process():
    """
    拆分训练集和验证集
    :return:
    """
    data = FashionMNIST(
        root='./data',
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(size=28)
        ])
    )
    train_data, val_data = utils.data.random_split(data, [round(len(data) * 0.8), len(data) - round(len(data) * 0.8)])
    train_dataloader = utils.data.DataLoader(train_data, batch_size=128, shuffle=True, num_workers=4)
    eval_dataloader = utils.data.DataLoader(train_data, batch_size=128, shuffle=True, num_workers=4)
    return train_dataloader, eval_dataloader

def train_model(model, train_dataloader, eval_dataloader, epochs=10, lr=0.001):
    """
    训练模型
    :param model:
    :param train_dataloader:
    :param val_dataloader:
    :param epochs:
    :param lr:
    :return:
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 优化器，用于反向传播、梯度下降, model.parameters(): 模型初始化的参数
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # 损失函数，在分类模型中，一般使用交叉熵损失函数
    criterion = torch.nn.CrossEntropyLoss()
    # 将模型放到设备中
    model = model.to(device)
    # 复制当前模型的参数
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    train_loss_all = []
    train_correct_all = []
    train_num = train_dataloader.batch_size * len(train_dataloader)
    eval_loss_all = []
    eval_correct_all = []
    eval_num = eval_dataloader.batch_size * len(eval_dataloader)
    start_time = time.time()
    for epoch in range(epochs):  # 训练的次数
        print(f'Epoch {epoch + 1}/{epochs}')
        print('-' * 10)
        # 训练集的损失值
        train_loss = 0.0
        # 训练集的准确率
        train_correct = 0

        #################
        #    训练集      #
        #################
        for batch_idx, (inputs, labels) in enumerate(train_dataloader):  # 获取每一个批次的数据
            inputs = inputs.to(device)
            labels = labels.to(device)
            # 开启训练模式
            model.train()
            # 正向传播, 输出128个样本对应的10个类别的概率
            outputs = model(inputs)
            # 获取每一张图片的最大概率对应的类别下标
            pre_labels = torch.argmax(outputs, dim=1)
            # 计算损失值
            loss = criterion(outputs, labels)
            # 将梯度初始化为0
            optimizer.zero_grad()
            # 反向传播
            loss.backward()
            # 根据网络反向传播的梯度信息来更新网络的参数，以起到降低loss函数计算值的作用
            optimizer.step()
            # 对损失值进行累加
            train_loss += loss.item() * labels.size(0)
            # 计算准确率
            train_correct += torch.sum(pre_labels == labels).item()
        train_loss_all.append(train_loss / train_num)
        train_correct_all.append(train_correct / train_num)

        #################
        #    验证集      #
        #################
        # 验证集的损失值
        eval_loss = 0.0
        # 验证集的准确率
        eval_correct = 0
        for batch_idx, (inputs, labels) in enumerate(eval_dataloader):  # 获取每一个批次的数据
            inputs = inputs.to(device)
            labels = labels.to(device)
            # 开启训练模式
            model.train()
            # 正向传播, 输出128个样本对应的10个类别的概率
            outputs = model(inputs)
            # 获取每一张图片的最大概率对应的类别下标
            pre_labels = torch.argmax(outputs, dim=1)
            # 计算损失值
            loss = criterion(outputs, labels)
            # 对损失值进行累加
            eval_loss += loss.item() * labels.size(0)
            # 计算准确率
            eval_correct += torch.sum(pre_labels == labels).item()

        eval_loss_all.append(eval_loss / eval_num)
        eval_correct_all.append(eval_correct / eval_num)
        print(f'train_loss: {train_loss_all[-1]}, train_correct: {train_correct_all[-1]}, ')
        print(f'eval_loss: {eval_loss_all[-1]}, eval_correct: {eval_correct_all[-1]}, ')
        # 获取最优模型
        if eval_correct_all[-1] > best_acc:
            best_acc = eval_correct_all[-1]
            best_model_wts = copy.deepcopy(model.state_dict())
        print(f"耗时：{time.time() - start_time}")

    # 保存最优的模型
    torch.save(best_model_wts, 'model.pth')
    df = pd.DataFrame(data={
        "epoch": range(1, epochs + 1),
        'train_loss': train_loss_all,
        'train_correct': train_correct_all,
        'eval_loss': eval_loss_all,
        'eval_correct': eval_correct_all
    })
    return df


def show_data(df):
    """
    展示训练结果
    :param df:
    :return:
    """
    df.plot(x='epoch', y=['train_loss', 'eval_loss'])
    df.plot(x='epoch', y=['train_correct', 'eval_correct'])
    plt.show()


if __name__ == '__main__':
    train_dataloader, eval_dataloader = train_val_data_process()
    model = LeNet()
    df = train_model(model, train_dataloader, eval_dataloader, epochs=10, lr=0.001)
    show_data(df)
