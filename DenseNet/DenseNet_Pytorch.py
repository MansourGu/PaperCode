from AlexNet.DataLoader_Pytorch_Cifar_10 import  Cifar10_DataSet
from torch import nn
from torch.optim import Adam
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import numpy as np
#########超参数设置#########
device = 'cuda'
lr = 1e-3 # 学习率
batch_size = 128
num_class = 10
weight_decay = 5e-4
momentum = 0.9
num_epoch = 100
Train_file_path = 'E:/Code/DataSet/cifar-10-batches-py/data_batch_{}'
Test_file_path = 'E:/Code/DataSet/cifar-10-batches-py/test_batch'
#########################

class Dense_Layer(nn.Module):
    def __init__(self, input_channel, growth_rate, bn_size, drop_rate):
        super(Dense_Layer, self).__init__()
        self.BN1 = nn.BatchNorm2d(input_channel)
        self.Relu = nn.ReLU()
        self.Conv1 = nn.Conv2d(input_channel, growth_rate * bn_size, kernel_size=1, stride=1)
        self.BN2 = nn.BatchNorm2d(bn_size * growth_rate)
        self.Conv2 = nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1)

        self.drop_rate = drop_rate

    def forward(self, input):
        if isinstance(input, torch.Tensor):
            previous_feature = [input]
        else:
            previous_feature = input
        x = torch.cat(previous_feature, 1) # 将之前特征图按照通道维度进行拼接

        x = self.BN1(x)
        x = self.Relu(x)
        x = self.Conv1(x)  # BottleNeck output

        x = self.BN2(x)
        x = self.Relu(x)
        x = self.Conv2(x) # new feature
        if self.drop_rate > 0:
            x = nn.functional.dropout(x, self.drop_rate, training=self.training)

        return x

class DenseBlock(nn.ModuleDict):
    def __init__(self, num_layers, input_channel, growth_rate, bn_size, drop_rate):
        super(DenseBlock, self).__init__()

        for i in range(num_layers):
            dense_layer = Dense_Layer(input_channel + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('dense_layer_{}'.format(i+1), dense_layer)

    def forward(self, x):
        features = [x]
        for name, layer in self.items():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)

class Transition_layer(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(Transition_layer, self).__init__()
        self.BN1 = nn.BatchNorm2d(input_channel)
        self.Relu = nn.ReLU()
        self.Conv1 = nn.Conv2d(input_channel, output_channel, kernel_size=1, stride=1)
        self.AvgPool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):

        x = self.BN1(x)
        x = self.Relu(x)
        x = self.Conv1(x)
        x = self.AvgPool(x)

        return x

class DenseNet(nn.Module):
    def __init__(self, growth_rate=12, block_config=(6, 12, 24, 16), initial_channel=64, bn_size=4, drop_rate=0, num_class=1000):
        super(DenseNet, self).__init__()
        self.Conv1 = nn.Conv2d(3, initial_channel, kernel_size=7, stride=2, padding=3)
        self.BN1 = nn.BatchNorm2d(initial_channel)
        self.Relu = nn.ReLU()
        self.MaxPool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        channels = initial_channel
        self.layers = []
        for i, num in enumerate(block_config):
            layer = self.make_layer(num, channels, bn_size, growth_rate, drop_rate)
            self.layers.append(layer)
            channels = channels + growth_rate * num
            if i != len(block_config) - 1:
                self.layers.append(Transition_layer(channels, channels//2))
                channels //= 2
        self.layers.append(nn.BatchNorm2d(channels))
        self.layers = nn.Sequential(*self.layers)

        self.Global_Avg = nn.AdaptiveAvgPool2d((1, 1))
        self.Flatten = nn.Flatten()
        self.Linear = nn.Linear(channels, num_class)

    def make_layer(self, num_layer, input_channel, bn_size, growth_rate, dropout_rate):
        layers = nn.Sequential(DenseBlock(num_layer, input_channel, growth_rate, bn_size, dropout_rate))
        return layers

    def forward(self, x):
        x = self.Conv1(x)
        x = self.BN1(x)
        x = self.Relu(x)
        x = self.MaxPool(x)

        x = self.layers(x)

        x = self.Relu(x)

        x = self.Global_Avg(x)

        x = self.Flatten(x)

        x = self.Linear(x)
        return x

class DenseNet_Cifar(nn.Module):
    def __init__(self, growth_rate=12, block_config=(6, 12, 24, 16), initial_channel=64, bn_size=4, drop_rate=0, num_class=10):
        super(DenseNet_Cifar, self).__init__()
        self.Conv1 = nn.Conv2d(3, initial_channel, kernel_size=3, stride=1, padding=1)
        self.BN1 = nn.BatchNorm2d(initial_channel)
        self.Relu = nn.ReLU()
        self.MaxPool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        channels = initial_channel
        self.layers = []
        for i, num in enumerate(block_config):
            layer = self.make_layer(num, channels, bn_size, growth_rate, drop_rate)
            self.layers.append(layer)
            channels = channels + growth_rate * num
            if i != len(block_config) - 1:
                self.layers.append(Transition_layer(channels, channels//2))
                channels //= 2
        self.layers.append(nn.BatchNorm2d(channels))
        self.layers = nn.Sequential(*self.layers)

        self.Global_Avg = nn.AdaptiveAvgPool2d((1, 1))
        self.Flatten = nn.Flatten()
        self.Linear = nn.Linear(channels, num_class)

    def make_layer(self, num_layer, input_channel, bn_size, growth_rate, dropout_rate):
        layers = nn.Sequential(DenseBlock(num_layer, input_channel, growth_rate, bn_size, dropout_rate))
        return layers

    def forward(self, x):
        x = self.Conv1(x)
        x = self.BN1(x)
        x = self.Relu(x)
        #x = self.MaxPool(x)

        x = self.layers(x)

        x = self.Relu(x)

        x = self.Global_Avg(x)

        x = self.Flatten(x)

        x = self.Linear(x)
        return x

################数据集处理##################
transform = transforms.Compose([transforms.Resize((56, 56))])
train_dataset = Cifar10_DataSet(Train_file_path, train=True, transform=transform)
test_dataset = Cifar10_DataSet(Test_file_path, train=False, transform=transform)

Train_DataLoader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=0)
Test_DataLoader = DataLoader(test_dataset, shuffle=True, batch_size=batch_size, num_workers=0)
###########################################

Net = DenseNet_Cifar().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = Adam(Net.parameters(), lr=lr)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=4, gamma=0.1)

for epoch in range(num_epoch):
    Net.train()  # 开始训练
    Correct = 0.
    Total_Loss = 0
    for i, data in tqdm(enumerate(Train_DataLoader)):
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
        for i, data in enumerate(Test_DataLoader):
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