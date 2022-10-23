from AlexNet.DataLoader_Pytorch_Cifar_10 import Load_Cifar_10_Train_Dataset, Load_Cifar_10_Test_Dataset
import torch
from torch import nn

Train_file_path = 'E:/Code/DataSet/cifar-10-batches-py/data_batch_{}'
Test_file_path = 'E:/Code/DataSet/cifar-10-batches-py/test_batch'
train_dataset = Load_Cifar_10_Train_Dataset(Train_file_path)
test_dataset =Load_Cifar_10_Test_Dataset(Test_file_path)

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


        x = self.Linear1(x)
        x = self.Relu(x)
        x = self.Dropout(x)

        x = self.Linear2(x)
        x = self.Relu(x)
        x = self.Dropout(x)

        x = self.Linear3(x)
