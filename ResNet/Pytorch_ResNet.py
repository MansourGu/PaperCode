from AlexNet.DataLoader_Pytorch_Cifar_10 import  Cifar10_DataSet
from torch import nn
from torch.optim import Adam

from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import numpy as np
#########超参数设置#########
device = 'cuda'
lr = 1e-3 # 学习率
batch_size = 256
num_class = 10
weight_decay = 5e-4
momentum = 0.9
num_epoch = 100
Train_file_path = 'E:/Code/DataSet/cifar-10-batches-py/data_batch_{}'
Test_file_path = 'E:/Code/DataSet/cifar-10-batches-py/test_batch'
#########################

class DownSample(nn.Module):
    def __init__(self, input_channel, output_channel, dilation=2):
        super(DownSample, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_channel, out_channels=output_channel, kernel_size=1, stride=dilation, dilation=dilation)
        self.BN1 = nn.BatchNorm2d(output_channel)

    def forward(self, x):
        x = self.conv1(x)
        x = self.BN1(x)
        return x

class Basic_Block(nn.Module):
    def __init__(self, input_channel, output_channel, DownSample = None, dilation=2):
        super(Basic_Block, self).__init__()
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.DownSample = DownSample
        if DownSample is not None:
            stride = 2
            pad = 1
        else:
            stride = 1
            pad = 1

        self.Conv1 = nn.Conv2d(input_channel, output_channel, kernel_size=3, stride=stride, padding=pad)
        self.BN1 = nn.BatchNorm2d(output_channel)
        self.Conv2 = nn.Conv2d(output_channel, output_channel, kernel_size=3, stride=1, padding=1)
        self.BN2 = nn.BatchNorm2d(output_channel)
        self.Relu = nn.ReLU()

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #     elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)

    def forward(self, x):
        identity = x
        x = self.Conv1(x)
        x = self.BN1(x)
        x = self.Relu(x)
        x = self.Conv2(x)
        x = self.BN2(x)

        if self.DownSample is not None:
            identity = self.DownSample(identity)
        x = identity + x
        x = self.Relu(x)

        return x

class Bottleneck(nn.Module):
    def __init__(self, input_channel, output_channel, DownSample = None, dilation=2):
        super(Bottleneck, self).__init__()
        self.DownSample = DownSample
        if DownSample is None:
            stride = 1
            pad = 0
        elif DownSample is not None and dilation == 1:
            stride = 1
            pad = 0
        else:
            stride = 2
            pad = 0
        self.input_channel = input_channel
        self.output_channel = output_channel
        first_channel = input_channel
        if output_channel / first_channel == 4 and dilation == 2:
            self.Conv1 = nn.Conv2d(self.output_channel, input_channel, kernel_size=1, stride=stride, padding=pad)
        elif output_channel / first_channel == 4 and dilation == 1:
            self.Conv1 = nn.Conv2d(self.input_channel, input_channel, kernel_size=1, stride=stride, padding=pad)
        elif output_channel / first_channel == 2:
            self.input_channel = int(self.input_channel / 2)
            self.Conv1 = nn.Conv2d(first_channel, self.input_channel, kernel_size=1, stride=stride, padding=pad)

        self.ReLu = nn.ReLU()



        self.BN1 = nn.BatchNorm2d(self.input_channel)
        self.Conv2 = nn.Conv2d(self.input_channel, self.input_channel, kernel_size=3, stride=1, padding=1)
        self.BN2 = nn.BatchNorm2d(self.input_channel)
        self.Conv3 = nn.Conv2d(self.input_channel, self.output_channel, kernel_size=1, stride=1, padding=0)
        self.BN3 = nn.BatchNorm2d(self.output_channel)

    def forward(self, x):
        identity = x
        x = self.Conv1(x)
        x = self.BN1(x)
        x = self.ReLu(x)
        x = self.Conv2(x)
        x = self.BN2(x)
        x = self.Conv3(x)
        x = self.BN3(x)

        if self.DownSample is not None:
            identity = self.DownSample(identity)
        x += identity
        x = self.ReLu(x)

        return x

class ResNet_ImageNet(nn.Module):
    def __init__(self, Block, block_list):
        super(ResNet_ImageNet, self).__init__()
        self.block_list = block_list
        self.Conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.BN1 = nn.BatchNorm2d(64)
        self.Relu = nn.ReLU()
        self.MaxPool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        if Block == Bottleneck:
            self.Layer1 = self.make_layer(self.block_list[0], Block, 64, 64, 256, is_DownSample=True, dilation=1)
            self.Layer2 = self.make_layer(self.block_list[1], Block, 256, 128, 512, is_DownSample=True)
            self.Layer3 = self.make_layer(self.block_list[2], Block, 512, 256, 1024, is_DownSample=True)
            self.Layer4 = self.make_layer(self.block_list[3], Block, 1024, 512, 2048, is_DownSample=True)
            self.Linear = nn.Linear(2048, 1000)
        else:
            self.Layer1 = self.make_layer(self.block_list[0], Block, 64, 64, 64, is_DownSample=False)
            self.Layer2 = self.make_layer(self.block_list[1], Block, 64, 128, 128, is_DownSample=True)
            self.Layer3 = self.make_layer(self.block_list[2], Block, 128, 256, 256, is_DownSample=True)
            self.Layer4 = self.make_layer(self.block_list[3], Block, 256, 512, 512, is_DownSample=True)
            self.Linear = nn.Linear(512, 1000)
        self.AvgPool = nn.AdaptiveAvgPool2d((1, 1))
        self.Flatten = nn.Flatten()
    def make_layer(self, num, Block, input_channel, mid_channel, output_channel, is_DownSample = True, dilation=2):
        layer = []
        if is_DownSample:
            DownSample_layer = DownSample(input_channel, output_channel, dilation)
        else:
            DownSample_layer = None
        layer.append(Block(input_channel, output_channel, DownSample_layer, dilation))

        for i in range(1, num):
            layer.append(Block(mid_channel, output_channel))

        return nn.Sequential(*layer)

    def forward(self, x):

        x = self.Conv1(x)
        print(x.shape)
        x = self.BN1(x)
        x = self.Relu(x)
        x = self.MaxPool1(x)
        x = self.Layer1(x)
        print(x.shape)
        x = self.Layer2(x)
        print(x.shape)
        x = self.Layer3(x)
        print(x.shape)
        x = self.Layer4(x)
        print(x.shape)
        x = self.AvgPool(x)
        print(x.shape)
        x = self.Flatten(x)
        print(x.shape)
        x = self.Linear(x)
        print(x.shape)
        return x

class ResNet_Cifar(nn.Module):
    def __init__(self, n):
        super(ResNet_Cifar, self).__init__()
        self.n = int((n-2) / 6)
        self.Conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.BN1 = nn.BatchNorm2d(16)
        self.Relu = nn.ReLU()
        self.Layer1 = self.make_layer(self.n * 6 + 2, 16, 16, is_DownSaple=False)
        self.Layer2 = self.make_layer(self.n * 2, 16, 32, is_DownSaple=True)
        self.Layer3 = self.make_layer(self.n * 2, 32, 64, is_DownSaple=True)
        self.AvgPool = nn.AdaptiveAvgPool2d((1, 1))
        self.Flatten = nn.Flatten()
        self.Linear = nn.Linear(64, 10)
    def make_layer(self, num, input_channel, output_channel, is_DownSaple=True, dilation=2):
        layers = []
        if is_DownSaple:
            DownSample_layer = DownSample(input_channel, output_channel, dilation)
        else:
            DownSample_layer = None
        layers.append(Basic_Block(input_channel, output_channel, DownSample_layer, dilation))
        for i in range(1, num):
            layers.append(Basic_Block(output_channel, output_channel, None, dilation))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.Conv1(x)
        x = self.BN1(x)
        x = self.Relu(x)

        x = self.Layer1(x)

        x = self.Layer2(x)

        x = self.Layer3(x)

        x = self.AvgPool(x)

        x = self.Flatten(x)

        x = self.Linear(x)

        return x

train_dataset = Cifar10_DataSet(file_path=Train_file_path, train=True)
test_dataset = Cifar10_DataSet(file_path=Test_file_path, train=False)

train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
test_dataloader = DataLoader(test_dataset, shuffle=True, batch_size=batch_size)

Net = ResNet_Cifar(50).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = Adam(Net.parameters(), lr=lr)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=10, gamma=0.1)

for epoch in range(num_epoch):
    Net.train()  # 开始训练
    Correct = 0.
    Total_Loss = 0
    for i, data in tqdm(enumerate(train_dataloader)):
        img, label = data
        label = label.type(torch.LongTensor) ##将label从int 转化为长向量
        label = label.to(device)
        #print(img.shape)
        img = img.to(device)
        output = Net(img)
        loss = criterion(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        Net.eval()
        for i, data in enumerate(test_dataloader):
            img, label = data
            label = label.type(torch.LongTensor)  ##将label从int 转化为长向量
            img, label = img.to(device), label.to(device)
            output = Net(img)
            loss = criterion(output, label).to(device)
            _, preds = torch.max(output, dim=1)
            Correct += torch.sum(preds == label).item()
            Total_Loss += loss
        lr_scheduler.step()
        print('Epoch{}: Test_AverageLoss:{}, Test_Acc:{}%'.format(epoch + 1, Total_Loss / 10000, Correct / 100))