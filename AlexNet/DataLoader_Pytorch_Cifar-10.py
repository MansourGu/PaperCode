import numpy as np
import cv2 as cv
file_path = 'E:/Code/DataSet/cifar-10-batches-py/data_batch_1'
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def Load_Cifar_10_Batch(file_path):
    batch_dict = unpickle(file_path)


dict = unpickle(file_path)
pictures = dict[b'data']
pic = pictures.reshape(10000, 3, 32, 32)[0]
print(pic.shape)
pic = np.transpose(pic, (1, 2, 0))

cv.imshow('修改后', pic)
cv.waitKey(0)