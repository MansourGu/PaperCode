import torch
from AlexNet.DataLoader_Pytorch_Cifar_10 import Cifar10_DataSet
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm

################超参数#####################
Train_file_path = 'E:/Code/DataSet/cifar-10-batches-py/data_batch_{}'
Test_file_path = 'E:/Code/DataSet/cifar-10-batches-py/test_batch'
batch_size = 128
lr = 1e-4
device = 'cuda'
num_class = 10
num_epoch = 20
#########################################################


##########网络设计##############################
class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()

        self.Conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1) # (batch, 64, 224, 224)
        self.Conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1) # (batch, 64, 224, 224)
        self.MaxPool1 = nn.MaxPool2d(kernel_size=2, stride=2) # (batch, 64, 112, 112)

        self.Conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1) # (batch,128, 112, 112)
        self.Conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1) # (batch, 128, 112, 112)
        self.MaxPool2 = nn.MaxPool2d(kernel_size=2, stride=2) # (batch, 128, 56, 56)

        self.Conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1) # (batch,256, 56, 56)
        self.Conv6 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1) # (batch, 256, 56, 56)
        self.Conv7 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0) # (batch, 256, 56, 56)
        self.MaxPool3 = nn.MaxPool2d(kernel_size=2, stride=2) # (batch, 256, 28, 28)

        self.Conv8 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)  # (batch,512, 28, 28)
        self.Conv9 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)  # (batch, 512, 28, 28)
        self.Conv10 = nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0)  # (batch, 512, 28, 28)
        self.MaxPool4 = nn.MaxPool2d(kernel_size=2, stride=2)  # (batch, 512, 14, 14)

        self.Conv11 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)  # (batch,512, 14, 14)
        self.Conv12 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)  # (batch,512, 14, 14)
        self.Conv13 = nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0)  # (batch,512, 14, 14)
        self.MaxPool5 = nn.MaxPool2d(kernel_size=2, stride=2)  # (batch, 512, 7, 7)

        self.Linear1 = nn.Linear(7 * 7 * 512, 4096)
        self.Linear2 = nn.Linear(4096, 4096)
        self.Linear3 = nn.Linear(4096, 1000)

        self.Relu = nn.ReLU()

        self.Dropout = nn.Dropout(0.5)

        self.Flatten = nn.Flatten()

    def forward(self, x):
        x = self.Conv1(x)
        x = self.Relu(x)
        x = self.Conv2(x)
        x = self.Relu(x)
        x = self.MaxPool1(x)

        x = self.Conv3(x)
        x = self.Relu(x)
        x = self.Conv4(x)
        x = self.Relu(x)
        x = self.MaxPool2(x)

        x = self.Conv5(x)
        x = self.Relu(x)
        x = self.Conv6(x)
        x = self.Relu(x)
        x = self.Conv7(x)
        x = self.Relu(x)
        x = self.MaxPool3(x)

        x = self.Conv8(x)
        x = self.Relu(x)
        x = self.Conv9(x)
        x = self.Relu(x)
        x = self.Conv10(x)
        x = self.Relu(x)
        x = self.MaxPool4(x)

        x = self.Conv11(x)
        x = self.Relu(x)
        x = self.Conv12(x)
        x = self.Relu(x)
        x = self.Conv13(x)
        x = self.Relu(x)
        x = self.MaxPool5(x)

        x = self.Flatten(x)
        x = self.Linear1(x)
        x = self.Relu(x)
        x = self.Dropout(x)

        x = self.Linear2(x)
        x = self.Relu(x)
        x = self.Dropout(x)

        x = self.Linear3(x)

        return x

class MyNet(nn.Module):
    def __init__(self, num_class):
        super(MyNet, self).__init__()
        self.num_class = num_class
        self.Conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)  # (batch, 16, 64, 64)
        self.Conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)  # (batch, 16, 64, 64)
        # self.MaxPool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # (batch, 64, 112, 112)

        self.Conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)  # (batch, 32, 64, 64)
        self.Conv4 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)  # (batch, 32, 64, 64)
        # self.MaxPool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # (batch, 128, 56, 56)

        self.Conv5 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)  # (batch, 64, 64, 64)
        self.Conv6 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)  # (batch, 64, 64, 64)
        self.Conv7 = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0)  # (batch, 64, 64, 64)
        self.MaxPool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # (batch, 64, 32, 32)

        self.Conv8 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)  # (batch, 128, 32, 32)
        self.Conv9 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)  # (batch, 128, 32, 32)
        self.Conv10 = nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0)  # (batch, 128, 32, 32)
        self.MaxPool4 = nn.MaxPool2d(kernel_size=2, stride=2)  # (batch, 512, 16, 16)

        self.Conv11 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)  # (batch, 128, 16, 16)
        self.Conv12 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)  # (batch, 128, 16, 16)
        self.Conv13 = nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0)  # (batch, 128, 16, 16)
        self.MaxPool5 = nn.MaxPool2d(kernel_size=2, stride=2)  # (batch, 128, 8, 8)

        self.Linear1 = nn.Linear(8 * 8 * 128, 4096)
        self.Linear2 = nn.Linear(4096, 1024)
        self.Linear3 = nn.Linear(1024, self.num_class)

        self.Relu = nn.ReLU(inplace=True)

        self.Dropout = nn.Dropout(0.5)

        self.Flatten = nn.Flatten()
    def forward(self, x):
        x = self.Conv1(x)
        x = self.Relu(x)
        x = self.Conv2(x)
        x = self.Relu(x)
       # x = self.MaxPool1(x)

        x = self.Conv3(x)
        x = self.Relu(x)
        x = self.Conv4(x)
        x = self.Relu(x)
       # x = self.MaxPool2(x)

        x = self.Conv5(x)
        x = self.Relu(x)
        x = self.Conv6(x)
        x = self.Relu(x)
        x = self.Conv7(x)
        x = self.Relu(x)
        x = self.MaxPool3(x)

        x = self.Conv8(x)
        x = self.Relu(x)
        x = self.Conv9(x)
        x = self.Relu(x)
        x = self.Conv10(x)
        x = self.Relu(x)
        x = self.MaxPool4(x)

        x = self.Conv11(x)
        x = self.Relu(x)
        x = self.Conv12(x)
        x = self.Relu(x)
        x = self.Conv13(x)
        x = self.Relu(x)
        x = self.MaxPool5(x)

        x = self.Flatten(x)
        x = self.Linear1(x)
        x = self.Relu(x)
  #      x = self.Dropout(x)

        x = self.Linear2(x)
        x = self.Relu(x)
    #    x = self.Dropout(x)

        x = self.Linear3(x)

        return x

#######################################

################数据集处理##################
transform = transforms.Compose([transforms.Resize((64, 64))])
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
        output = Net(img)
        #print(output.shape)
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

