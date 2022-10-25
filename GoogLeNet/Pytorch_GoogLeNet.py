from AlexNet.DataLoader_Pytorch_Cifar_10 import  Cifar10_DataSet
from torch import nn
from torch.optim import Adam
from torchvision import transforms
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
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


class Inception_v1(nn.Module):
    def __init__(self, input_channel, ch1x1, ch3x3_reduce, ch3x3, ch5x5_reduce, ch5x5, pool_proj):
        super(Inception_v1, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels=input_channel, out_channels=ch1x1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True))
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels=input_channel, out_channels=ch3x3_reduce, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True),
            nn.Conv2d(in_channels=ch3x3_reduce, out_channels=ch3x3, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True))
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels=input_channel, out_channels=ch5x5_reduce, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True),
            nn.Conv2d(in_channels=ch5x5_reduce, out_channels=ch5x5, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True))
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True),
            nn.Conv2d(in_channels=input_channel, out_channels=pool_proj, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True))

    def _forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 =self.branch4(x)
        output = [branch1, branch2, branch3, branch4]
        return output

    def forward(self, x):
        result = self._forward(x)
        return torch.cat(result, 1)

class Aux_Logits(nn.Module):
    def __init__(self, input_channel, num_class):
        super(Aux_Logits, self).__init__()
        self.Conv1 = nn.Conv2d(in_channels=input_channel, out_channels=128, kernel_size=1, stride=1, padding=0)
        self.ReLu = nn.ReLU()
        self.Flatten = nn.Flatten()
        self.Linear1 = nn.Linear(in_features=2048, out_features=1024)
        self.Linear2 = nn.Linear(in_features=1024, out_features=num_class)
        self.Dropout = nn.Dropout(0.7)

    def forward(self, x):
        x = F.adaptive_avg_pool2d(x, (4, 4))
        x = self.Conv1(x)
        x = self.ReLu(x)
        x = self.Flatten(x)
        x = self.Linear1(x)
        x = self.ReLu(x)
        x = self.Dropout(x)
        x = self.Linear2(x)

        return x

class GoogLeNet(nn.Module):
    def __init__(self, num_class):
        super(GoogLeNet, self).__init__()
        self.num_class = num_class
        self.Relu = nn.ReLU()


        # (batch, 3, 224, 224)
        self.Conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3)
        # (batch, 64, 112, 112)
        self.Maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        # (batch, 64, 56, 56)
        self.Conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0)
        # (batch, 64, 56, 56)
        self.Conv3 = nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, stride=1, padding=1)
        # (batch, 192, 56, 56)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        # (batch, 192, 28, 28)

        self.Inception3a = Inception_v1(192, 64, 96, 128, 16, 32, 32)
        # (batch, 256, 28, 28)
        self.Inception3b = Inception_v1(256, 128, 128, 192, 32, 96, 64)
        # (batch, 480, 28, 28)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        # (batch, 480, 14, 14)

        self.Inception4a = Inception_v1(480, 192, 96, 208, 16, 48, 64)
        # (batch, 512, 14, 14)
        self.Inception4b = Inception_v1(512, 160, 112, 224, 24, 64, 64)
        # (batch, 512, 14, 14)
        self.Inception4c = Inception_v1(512, 128, 128, 256, 24, 64, 64)
        # (batch, 512, 14, 14)
        self.Inception4d = Inception_v1(512, 112, 144, 288, 32, 64, 64)
        # (batch, 528, 14, 14)
        self.Inception4e = Inception_v1(528, 256, 160, 320, 32, 128, 128)
        # (batch, 832, 14, 14)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        # (batch, 832, 7, 7)

        self.aux1 = Aux_Logits(512, num_class)
        self.aux2 = Aux_Logits(528, num_class)

        self.Inception5a = Inception_v1(832, 256, 160, 320, 32, 128, 128)
        # (batch, 832, 7, 7)
        self.Inception5b = Inception_v1(832, 384, 192, 384, 48, 128, 128)
        # (batch, 1024, 7, 7)

        self.Avgpool1 = nn.AdaptiveAvgPool2d((1, 1))
        # (batch, 1024, 1, 1)

        self.Flatten = nn.Flatten()
        # (batch, 1024)

        self.Dropout = nn.Dropout(0.2)

        self.Linear1 = nn.Linear(in_features=1024, out_features=num_class)

    def forward(self, x):
        x = self.Conv1(x)
        x = self.Relu(x)
        x = self.Maxpool1(x)

        x = self.Conv2(x)
        x = self.Relu(x)
        x = self.Conv3(x)
        x = self.Relu(x)
        x = self.Maxpool2(x)

        x = self.Inception3a(x)
        x = self.Inception3b(x)
        x = self.Maxpool3(x)

        x = self.Inception4a(x)
        if self.training:
            aux1 = self.aux1(x)
        else:
            aux1 = None

        x = self.Inception4b(x)
        x = self.Inception4c(x)
        x = self.Inception4d(x)

        if self.training:
            aux2 = self.aux2(x)
        else:
            aux2 = None

        x = self.Inception4e(x)
        x = self.Maxpool4(x)

        x = self.Inception5a(x)
        x = self.Inception5b(x)

        x = self.Avgpool1(x)

        x = self.Flatten(x)
        x = self.Dropout(x)
        x = self.Linear1(x)

        return x, aux2, aux1


class MyNet(nn.Module):
    def __init__(self, num_class):
        super(MyNet, self).__init__()
        self.num_class = num_class
        self.Relu = nn.ReLU()

        # (batch, 3, 56, 56)
        self.Conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        # (batch, 64, 56, 56)
        #self.Maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        # (batch, 64, 56, 56)
        self.Conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0)
        # (batch, 64, 56, 56)
        self.Conv3 = nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, stride=1, padding=1)
        # (batch, 192, 56, 56)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        # (batch, 192, 28, 28)

        self.Inception3a = Inception_v1(192, 64, 96, 128, 16, 32, 32)
        # (batch, 256, 28, 28)
        self.Inception3b = Inception_v1(256, 128, 128, 192, 32, 96, 64)
        # (batch, 480, 28, 28)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        # (batch, 480, 14, 14)

        self.Inception4a = Inception_v1(480, 192, 96, 208, 16, 48, 64)
        # (batch, 512, 14, 14)
        self.Inception4b = Inception_v1(512, 160, 112, 224, 24, 64, 64)
        # (batch, 512, 14, 14)
        self.Inception4c = Inception_v1(512, 128, 128, 256, 24, 64, 64)
        # (batch, 512, 14, 14)
        self.Inception4d = Inception_v1(512, 112, 144, 288, 32, 64, 64)
        # (batch, 528, 14, 14)
        self.Inception4e = Inception_v1(528, 256, 160, 320, 32, 128, 128)
        # (batch, 832, 14, 14)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        # (batch, 832, 7, 7)

        self.aux1 = Aux_Logits(512, num_class)
        self.aux2 = Aux_Logits(528, num_class)

        self.Inception5a = Inception_v1(832, 256, 160, 320, 32, 128, 128)
        # (batch, 832, 7, 7)
        self.Inception5b = Inception_v1(832, 384, 192, 384, 48, 128, 128)
        # (batch, 1024, 7, 7)

        self.Avgpool1 = nn.AdaptiveAvgPool2d((1, 1))
        # (batch, 1024, 1, 1)

        self.Flatten = nn.Flatten()
        # (batch, 1024)

        self.Dropout = nn.Dropout(0.2)

        self.Linear1 = nn.Linear(in_features=1024, out_features=num_class)

    def forward(self, x):
        x = self.Conv1(x)
        x = self.Relu(x)
        #x = self.Maxpool1(x)

        x = self.Conv2(x)
        x = self.Relu(x)
        x = self.Conv3(x)
        x = self.Relu(x)
        x = self.Maxpool2(x)

        x = self.Inception3a(x)
        x = self.Inception3b(x)
        x = self.Maxpool3(x)

        x = self.Inception4a(x)
        if self.training:
            aux1 = self.aux1(x)
        else:
            aux1 = None

        x = self.Inception4b(x)
        x = self.Inception4c(x)
        x = self.Inception4d(x)

        if self.training:
            aux2 = self.aux2(x)
        else:
            aux2 = None

        x = self.Inception4e(x)
        x = self.Maxpool4(x)

        x = self.Inception5a(x)
        x = self.Inception5b(x)

        x = self.Avgpool1(x)

        x = self.Flatten(x)
        x = self.Dropout(x)
        x = self.Linear1(x)

        return x, aux2, aux1


################数据集处理##################
transform = transforms.Compose([transforms.Resize((56, 56))])
train_dataset = Cifar10_DataSet(Train_file_path, train=True, transform=transform)
test_dataset = Cifar10_DataSet(Test_file_path, train=False, transform=transform)

Train_DataLoader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=0)
Test_DataLoader = DataLoader(test_dataset, shuffle=True, batch_size=batch_size, num_workers=0)
###########################################

Net = MyNet(num_class).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = Adam(Net.parameters(), lr=lr)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=30, gamma=0.1)

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
        output1, output2, output3 = Net(img)
        loss0 = criterion(output1, label)
        loss1 = criterion(output2, label)
        loss2 = criterion(output3, label)
        loss = loss0 + 0.3 * loss1 + 0.3 * loss2
        #print(output.shape)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        Net.eval()
        for i, data in enumerate(Test_DataLoader):
            img, label = data
            label = label.type(torch.LongTensor)  ##将label从int 转化为长向量
            img, label = img.to(device), label.to(device)
            output, _, _ = Net(img)
            loss = criterion(output, label).to(device)
            _, preds = torch.max(output, dim=1)
            Correct += torch.sum(preds == label).item()
            Total_Loss += loss
        lr_scheduler.step()
        print('Epoch{}: Test_AverageLoss:{}, Test_Acc:{}%'.format(epoch + 1, Total_Loss / 10000, Correct / 100))

