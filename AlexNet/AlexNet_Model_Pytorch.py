import torch
import torchvision.datasets
from torch import nn
from torch.optim import SGD
from torchvision import transforms
from DataLoader_Pytorch_Cifar_10 import Cifar10_DataSet
from torch.utils.data import DataLoader
from tqdm import tqdm

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
##############################

class AlexNet(nn.Module):###原始AlexNet####以适应ImageNet数据集 224 x 224
    def __init__(self, num_class):
        super(AlexNet, self).__init__()
        # input_size = (batch, 3, 227, 227)
        self.num_class = num_class
        self.Conv1 = nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0) # (batch, 96, 55, 55)
        self.Maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0) # (batch, 96, 27, 27)
        self.Conv2 = nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2) # (batch, 256, 27, 27)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0) # (batch, 256, 13, 13)
        self.Conv3 = nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1) # (batch, 384, 13, 13)
        self.Conv4 = nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1) # (batch, 384, 13, 13)
        self.Conv5 = nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1) # (batch, 256, 13, 13)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0) # (batch, 256, 6, 6)
        self.Linear1 = nn.Linear(256 * 6 * 6, 4096)
        self.Relu = nn.ReLU()
        self.Linear2 = nn.Linear(4096, 4096)
        self.Linear3 = nn.Linear(4096, self.num_class)
        self.SoftMax = nn.Softmax()
        self.LRN = nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2)
        self.DropOut = nn.Dropout(p=0.5)


    def forward(self, x):

        ####Conv Block####
        x = self.Conv1(x)
        #x = self.Relu(x)
        x = self.LRN(x)
        x = self.Maxpool1(x)
        x = self.Conv2(x)
        x = self.Relu(x)
        x = self.LRN(x)
        x = self.Maxpool2(x)
        x = self.Conv3(x)
        x = self.Relu(x)
        x = self.Conv4(x)
        x = self.Relu(x)
        x = self.Conv5(x)
        x = self.Relu(x)
        x = self.Maxpool3(x)
        ####################

        ######Linear Layer###
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1, 1).squeeze()
        x = self.DropOut(x)
        x = self.Linear1(x)
        x = self.Relu(x)
        x = self.DropOut(x)
        x = self.Linear2(x)
        x = self.Relu(x)
        x = self.Linear3(x)
        return x

class MyNet(nn.Module): # 从AlexNet修改以适应CIFAR-10数据集
    def __init__(self, num_class):
        super(MyNet, self).__init__()
        # input_size = (batch, 3, 32, 32)
        self.num_class = num_class
        self.Conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)  # (batch, 16, 32, 32)
        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # (batch, 16, 16, 16)
        self.Conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)  # (batch, 32, 16, 16)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # (batch, 32, 8, 8)
        self.Conv3 = nn.Conv2d(32, 128, kernel_size=3, stride=1, padding=1)  # (batch, 128, 8, 8)
        self.Conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)  # (batch, 128, 8, 8)
        self.Conv5 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)  # (batch, 64, 8, 8)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # (batch, 64, 4, 4)
        self.Linear1 = nn.Linear(64 * 4 * 4, 2048)
        self.Relu = nn.ReLU()
        self.Linear2 = nn.Linear(2048, 512)
        self.Linear3 = nn.Linear(512, self.num_class)
        self.LRN = nn.LocalResponseNorm(size=3, alpha=1e-4, beta=0.75, k=2)
        self.DropOut = nn.Dropout(p=0.5)

    def forward(self, x):
        ####Conv Block####
        x = self.Conv1(x)
        x = self.Relu(x)
        x = self.LRN(x)
        x = self.Maxpool1(x)
        x = self.Conv2(x)
        x = self.Relu(x)
        x = self.LRN(x)
        x = self.Maxpool2(x)
        x = self.Conv3(x)
        x = self.Relu(x)
        x = self.Conv4(x)
        x = self.Relu(x)
        x = self.Conv5(x)
        x = self.Relu(x)
        x = self.Maxpool3(x)
        ####################

        ######Linear Layer###
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1, 1).squeeze()
        x = self.DropOut(x)
        x = self.Linear1(x)
        x = self.Relu(x)
        x = self.DropOut(x)
        x = self.Linear2(x)
        x = self.Relu(x)
        x = self.Linear3(x)
        return x


#############初始化网络权重#############
def weight_init(layer): # 初始化权重
    if isinstance(layer, nn.Conv2d):
        layer.weight.data.normal_(0, 0.01)
        if layer.in_channels == 3:
            layer.bias.data.zero_()
        else:
            layer.bias.data.fill_(1)
    elif isinstance(layer, nn.Linear):
        layer.weight.data.normal_(0, 0.01)
        layer.bias.data.fill_(1)


Net = MyNet(10)
Net.apply(weight_init)
Net.to(device)
#########################################


#####数据集预处理#########
Train_Dataset = Cifar10_DataSet(Train_file_path, train=True, transform=transforms.Compose([transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]))
Train_DataLoader = DataLoader(Train_Dataset, batch_size, shuffle=True)
Test_Dataset = Cifar10_DataSet(Test_file_path, train=False, transform=transforms.Compose([transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]))
Test_DataLoader = DataLoader(Test_Dataset, batch_size, shuffle=True)
#########################


########训练器设置#########
criterion = nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.Adam(params=Net.parameters(), lr=lr)
#optimizer = torch.optim.SGD(params=Net.parameters(),momentum=momentum, lr=lr, weight_decay=weight_decay)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1) # 学习率衰减 30轮变为原来的1/10
###########################

##########训练##########

for epoch in range(num_epoch):
    Net.train()  # 开始训练
    Correct = 0.
    Total_Loss = 0
    for i, data in tqdm(enumerate(Train_DataLoader)):
        img, label = data
        label = label.type(torch.LongTensor) ##将label从int 转化为长向量
        label = label.to(device)

        img = img.to(device)
        output = Net(img)
        loss = criterion(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    Net.eval() # 在测试集训练
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
    print('Epoch{}: Test_AverageLoss:{}, Test_Acc:{}%'.format(epoch+1, Total_Loss/10000, Correct/100))




