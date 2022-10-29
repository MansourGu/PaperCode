import tensorflow as tf
import numpy as np
from tensorflow.python.keras.layers import Conv2D, MaxPool2D, ReLU, Dropout, Dense, Flatten, AvgPool2D
from tensorflow.python.keras import Sequential
import tensorflow_addons as tfa
from tqdm import tqdm
###########数据集路径#############
Train_file_path = 'E:/Code/DataSet/cifar-10-batches-py/data_batch_{}'
Test_file_path = 'E:/Code/DataSet/cifar-10-batches-py/test_batch'
################################

########超参数设置############
num_class = 10
num_epoch = 20
lr = 1e-4
batch_size = 256
############################

########数据集读取##############
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def Load_Cifar_10_Batch(batch_path):
    batch_dict = unpickle(batch_path)
    batch_data = batch_dict[b'data'].astype(np.float)
    batch_data /= 255.
    batch_label = batch_dict[b'labels']
    batch_data = batch_data.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1) #TensorFlow Conv2D 输入是 （batch，W，H，C)
    return np.array(batch_data), np.array(batch_label)

def Load_Cifar_10_Train_Dataset(file_path):
    Data_Set, Data_Label = Load_Cifar_10_Batch(file_path.format(1))
    for i in range(2, 6):
        batch_path = file_path.format(i)
        batch_data, batch_label = Load_Cifar_10_Batch(batch_path)
        Data_Set = np.vstack((Data_Set, batch_data))
        Data_Label = np.hstack((Data_Label, batch_label))
    return Data_Set, Data_Label

def Load_Cifar_10_Test_Dataset(file_path):
    Dataset, Data_Label = Load_Cifar_10_Batch(file_path)
    return Dataset, Data_Label



train_image, train_label = Load_Cifar_10_Train_Dataset(Train_file_path)
test_image, test_label = Load_Cifar_10_Test_Dataset(Test_file_path)
train_dataset = tf.data.Dataset.from_tensor_slices((train_image, train_label)).shuffle(1000).batch(batch_size)
test_dataset = tf.data.Dataset.from_tensor_slices((test_image, test_label)).shuffle(1000).batch(batch_size)
#######################################

class DownSample(tf.keras.Model):
    def __init__(self, output_channel, stride=2):
        super(DownSample, self).__init__()
        self.Conv1 = Conv2D(filters=output_channel, kernel_size=1, padding='same', strides=stride)
        self.BN1 = tf.keras.layers.BatchNormalization()

    def call(self, x):
        x = self.Conv1(x)
        x = self.BN1(x)
        return x

class Basic_Block(tf.keras.Model):
    def __init__(self, mid_channel, output_channel, DownSample=None, stride=2):
        super(Basic_Block, self).__init__()
        self.mid_channel = mid_channel
        self.output_channel = output_channel
        self.DownSample = DownSample

        self.Conv1 = Conv2D(mid_channel, kernel_size=3, padding='same', strides=stride)
        self.BN1 = tf.keras.layers.BatchNormalization()
        self.Conv2 = Conv2D(output_channel, kernel_size=3, strides=1, padding='same')
        self.BN2 = tf.keras.layers.BatchNormalization()
        self.Flatten = Flatten()
        self.Relu = ReLU()

    def call(self, x):
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

class Bottle_neck(tf.keras.Model):
    def __init__(self, mid_channel, output_channel, DownSample=None, stride=2):
        super(Bottle_neck, self).__init__()

        self.mid_channel = mid_channel
        self.output_channel = output_channel
        self.DownSample = DownSample
        self.Conv1 = Conv2D(mid_channel, kernel_size=1, strides=stride, padding='same')
        self.BN1 = tf.keras.layers.BatchNormalization()
        self.Conv2 = Conv2D(mid_channel, kernel_size=3, padding='same', strides=1)
        self.BN2 = tf.keras.layers.BatchNormalization()
        self.Conv3 = Conv2D(output_channel, kernel_size=1, strides=1, padding='same')
        self.BN3 = tf.keras.layers.BatchNormalization()
        self.Relu = ReLU()

    def call(self, x):
        identity = x
        x = self.Conv1(x)
        x = self.BN1(x)
        x = self.Relu(x)

        x = self.Conv2(x)

        x = self.BN2(x)
        x = self.Relu(x)
        x = self.Conv3(x)
        x = self.BN3(x)

        if self.DownSample is not None:
            identity = self.DownSample(identity)

        x += identity
        x = self.Relu(x)
        return x

class ResNet_ImageNet(tf.keras.Model):
    def __init__(self, Block, block_list):
        super(ResNet_ImageNet, self).__init__()
        self.block_list = block_list
        self.Conv1 = Conv2D(64, kernel_size=7, strides=2, padding='same')
        self.BN1 = tf.keras.layers.BatchNormalization()
        self.Relu = ReLU()
        self.MaxPool1 = MaxPool2D(pool_size=3, strides=2, padding='same')

        if Block == Bottle_neck:
            self.Layer1 = self.make_layer(self.block_list[0], Block, 64, 256, stride=1)
            self.Layer2 = self.make_layer(self.block_list[1], Block, 128, 512)
            self.Layer3 = self.make_layer(self.block_list[2], Block, 256, 1024)
            self.Layer4 = self.make_layer(self.block_list[3], Block, 512, 2048)
        else:
            self.Layer1 = self.make_layer(self.block_list[0], Block, 64, 64, is_DownSample=False, stride=1)
            self.Layer2 = self.make_layer(self.block_list[1], Block, 128, 128)
            self.Layer3 = self.make_layer(self.block_list[2], Block, 256, 256)
            self.Layer4 = self.make_layer(self.block_list[3], Block, 512, 512)
        self.Linear = Dense(1000)
        self.AvgPool = tfa.layers.AdaptiveAveragePooling2D(output_size=(1, 1))
        self.Flatten = Flatten()

    def make_layer(self, num, Block, mid_channel, output_channel, is_DownSample = True, stride=2):
        layers = []
        if is_DownSample:
            DownSample_layer = DownSample(output_channel, stride=stride)
        else:
            DownSample_layer = None
        layers.append(Block(mid_channel, output_channel, DownSample_layer, stride=stride))
        for i in range(num):
            layers.append(Block(mid_channel, output_channel, stride=1))

        return Sequential(layers)

    def call(self, x):
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

class ResNet_Cifar(tf.keras.Model):
    def __init__(self, n):
        super(ResNet_Cifar, self).__init__()
        self.n = int((n - 2) / 6)
        self.Conv1 = Conv2D(16, kernel_size=3, strides=1, padding='same')
        self.BN1 = tf.keras.layers.BatchNormalization()
        self.Relu = ReLU()
        self.Layer1 = self.make_layer(self.n * 2 + 1, 16, 16, is_DownSample=False, stride=1)
        self.Layer2 = self.make_layer(self.n * 2, 32, 32, is_DownSample=True, stride=2)
        self.Layer3 = self.make_layer(self.n * 2, 64, 64, is_DownSample=True, stride=2)
        self.AvgPool = tfa.layers.AdaptiveAveragePooling2D((1, 1))
        self.Flatten = Flatten()
        self.Linear = Dense(10)
        self.BN0 = tf.keras.layers.BatchNormalization()

    def make_layer(self, num, mid_channel, output_channel, is_DownSample, stride=2):
        layers = Sequential()
        if is_DownSample:
            DownSample_layer = DownSample(output_channel, stride=stride)
        else:
            DownSample_layer = None
        layers.add(Basic_Block(mid_channel, output_channel, DownSample_layer, stride=stride))
        for i in range(1, num):
            layers.add(Basic_Block(mid_channel, output_channel, stride=1))

        return layers

    def call(self, x):
        x = self.BN0(x)

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


#######设置损失函数与优化器##########
loss_func = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)  # 损失函数
optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
#################################

##################设置网络参数追踪###################
train_loss = tf.keras.metrics.Mean(name="train_loss")
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="train_acc")
test_loss = tf.keras.metrics.Mean(name="test_loss")
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="test_acc")
###################################################

###################训练与测试###########################
Net = ResNet_Cifar(50)

def train_batch(imgs, label):
    with tf.GradientTape() as tape:
        preds = Net(imgs, training=True)
        loss = loss_func(label, preds)
    gradients = tape.gradient(loss, Net.trainable_variables)
    optimizer.apply_gradients(zip(gradients, Net.trainable_variables))

    train_loss(loss)
    train_accuracy(label, preds)

def predict_test_batch(imgs, label):
    preds = Net(imgs, training=False)
    loss = loss_func(label, preds)

    test_loss(loss)
    test_accuracy(label, preds)


for epoch in range(num_epoch):
    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()

    for imgs, label in tqdm(train_dataset):
        train_batch(imgs, label)

    for imgs, label in tqdm(test_dataset):
        predict_test_batch(imgs, label)

    print(
        f'Epoch {epoch + 1}, '
        f'Loss: {train_loss.result()}, '
        f'Accuracy: {train_accuracy.result() * 100}, '
        f'Test Loss: {test_loss.result()}, '
        f'Test Accuracy: {test_accuracy.result() * 100}'
    )
############################################################

