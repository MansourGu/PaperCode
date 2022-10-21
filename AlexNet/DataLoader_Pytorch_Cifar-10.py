import numpy as np
from torch.utils.data import DataLoader, Dataset
file_path = 'E:/Code/DataSet/cifar-10-batches-py/data_batch_{}'
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def Load_Cifar_10_Batch(batch_path):
    batch_dict = unpickle(batch_path)
    batch_data = batch_dict[b'data']
    batch_label = batch_dict[b'labels']
    batch_data = batch_data.reshape(10000, 3, 32, 32)
    return np.array(batch_data), np.array(batch_label)

def Load_Cifar_10_Dataset(file_path):
    Data_Set, Data_Label = Load_Cifar_10_Batch(file_path.format(1))
    for i in range(2, 6):
        batch_path = file_path.format(i)
        batch_data, batch_label = Load_Cifar_10_Batch(batch_path)
        Data_Set = np.vstack((Data_Set, batch_data))
        Data_Label = np.hstack((Data_Label, batch_label))
    return Data_Set, Data_Label

class Cifar10_DataSet(Dataset):
    def __init__(self, file_path, transform=None):
        self.DataSet, self.Data_Label = Load_Cifar_10_Dataset(file_path)
        self.transform = transform

    def __getitem__(self, item):
        img = self.DataSet[item]
        label = self.Data_Label[item]
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return self.DataSet.shape[0]
