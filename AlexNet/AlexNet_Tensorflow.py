import tensorflow as tf
import numpy as np
from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Conv2D, MaxPool2D, ReLU, Dropout, Dense, Flatten
from keras.optimizers import Adam
from tensorflow.python.ops.nn import LRN

###########数据集路径#############
Train_file_path = 'E:/Code/DataSet/cifar-10-batches-py/data_batch_{}'
Test_file_path = 'E:/Code/DataSet/cifar-10-batches-py/test_batch'
################################

########超参数设置############
num_class = 10
num_epoch = 20
lr = 0.0001
batch_size = 128
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
########################################



##############网络设计################
class AlexNet(tf.keras.Model):

    def LRN_OP(self, x):
        return LRN(input=x, depth_radius=3, bias=2, alpha=1e-4, beta=0.75)


    def __init__(self, num_class):
        super(AlexNet, self).__init__()
        ####输入尺寸 (batch, 227, 227, 3)
        self.num_class = num_class
        self.Conv1 = Conv2D(filters=96, kernel_size=11, strides=4, padding='valid', activation='relu',
                            kernel_initializer=tf.random_normal_initializer(0, 0.01),
                            bias_initializer=tf.constant_initializer(0)) # (batch, 55, 55, 96)
        self.MaxPool1 = MaxPool2D(pool_size=(3, 3), strides=2, padding='valid') # (batch, 27, 27, 96)
        self.Conv2 = Conv2D(filters=256, kernel_size=5, strides=1, padding='same', activation='relu',
                            kernel_initializer=tf.random_normal_initializer(0, 0.01),
                            bias_initializer=tf.constant_initializer(1)) # (batch, 27, 27, 256)
        self.MaxPool2 = MaxPool2D(pool_size=(3, 3), strides=2, padding='valid') # (batch, 13, 13, 256)
        self.Conv3 = Conv2D(filters=384, kernel_size=3, strides=1, padding='same', activation='relu',
                            kernel_initializer=tf.random_normal_initializer(0, 0.01),
                            bias_initializer=tf.constant_initializer(1)) # (batch, 13, 13, 384)
        self.Conv4 = Conv2D(filters=384, kernel_size=3, strides=1, padding='same', activation='relu',
                            kernel_initializer=tf.random_normal_initializer(0, 0.01),
                            bias_initializer=tf.constant_initializer(1)) # (batch, 13, 13, 384)
        self.Conv5 = Conv2D(filters=256, kernel_size=3, strides=1, padding='same', activation='relu',
                            kernel_initializer=tf.random_normal_initializer(0, 0.01),
                            bias_initializer=tf.constant_initializer(1)) # (batch, 13, 13, 256)
        self.MaxPool3 = MaxPool2D(pool_size=(2, 2), strides=2, padding='valid') # (batch, 6, 6, 256)
        self.Flatten = Flatten()
        self.Linear1 = Dense(4096, 'relu',
                            kernel_initializer=tf.random_normal_initializer(0, 0.01),
                            bias_initializer=tf.constant_initializer(1)) # (batch, 4096)
        self.Linear2 = Dense(4096, 'relu',
                            kernel_initializer=tf.random_normal_initializer(0, 0.01),
                            bias_initializer=tf.constant_initializer(1)) # (batch, 4096)
        self.Linear3 = Dense(self.num_class,
                            kernel_initializer=tf.random_normal_initializer(0, 0.01),
                            bias_initializer=tf.constant_initializer(1)) # (batch, num_class)
        self.Dropout = Dropout(0.5)


    def call(self, x):
        x = self.Conv1(x)
        x = self.LRN_OP(x)
        x = self.MaxPool1(x)
        x = self.Conv2(x)
        x = self.LRN_OP(x)
        x = self.MaxPool2(x)
        x = self.Conv3(x)
        x = self.Conv4(x)
        x = self.Conv5(x)
        x = self.MaxPool3(x)
        x = self.Flatten(x)
        x = self.Linear1(x)
        x = self.Linear2(x)
        x = self.Linear3(x)
        return x


class MyNet(tf.keras.Model):

    def LRN_OP(self, x):
        return LRN(input=x, depth_radius=3, bias=2, alpha=1e-4, beta=0.75)


    def __init__(self, num_class):
        super(MyNet, self).__init__()
        ####输入尺寸 (batch, 32, 32, 3)
        self.num_class = num_class
        self.Conv1 = Conv2D(filters=16, kernel_size=3, strides=1, padding='same', activation='relu',
                            kernel_initializer=tf.random_normal_initializer(0, 0.01),
                            bias_initializer=tf.constant_initializer(0)) # (batch, 32, 32, 16)
        self.MaxPool1 = MaxPool2D(pool_size=(2, 2), strides=2, padding='valid') # (batch, 16, 16, 16)
        self.Conv2 = Conv2D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu',
                            kernel_initializer=tf.random_normal_initializer(0, 0.01),
                            bias_initializer=tf.constant_initializer(1)) # (batch, 16, 16, 32)
        self.MaxPool2 = MaxPool2D(pool_size=(2, 2), strides=2, padding='valid') # (batch, 8, 8, 32)
        self.Conv3 = Conv2D(filters=128, kernel_size=3, strides=1, padding='same', activation='relu',
                            kernel_initializer=tf.random_normal_initializer(0, 0.01),
                            bias_initializer=tf.constant_initializer(1)) # (batch, 8, 8, 128)
        self.Conv4 = Conv2D(filters=128, kernel_size=3, strides=1, padding='same', activation='relu',
                            kernel_initializer=tf.random_normal_initializer(0, 0.01),
                            bias_initializer=tf.constant_initializer(1)) # (batch, 8, 8, 128)
        self.Conv5 = Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu',
                            kernel_initializer=tf.random_normal_initializer(0, 0.01),
                            bias_initializer=tf.constant_initializer(1)) # (batch, 8, 8, 64)
        self.MaxPool3 = MaxPool2D(pool_size=(2, 2), strides=2, padding='valid') # (batch, 4, 4, 64)
        self.Flatten = Flatten()
        self.Linear1 = Dense(2048, 'relu',
                            kernel_initializer=tf.random_normal_initializer(0, 0.01),
                            bias_initializer=tf.constant_initializer(1)) # (batch, 2048)
        self.Linear2 = Dense(512, 'relu',
                            kernel_initializer=tf.random_normal_initializer(0, 0.01),
                            bias_initializer=tf.constant_initializer(1)) # (batch, 512)
        self.Linear3 = Dense(self.num_class,
                            kernel_initializer=tf.random_normal_initializer(0, 0.01),
                            bias_initializer=tf.constant_initializer(1)) # (batch, num_class)
        self.Dropout = Dropout(0.5)
        self.BN1 = tf.keras.layers.BatchNormalization()
        self.BN2 = tf.keras.layers.BatchNormalization()

    def call(self, x):
        x = self.BN1(x)
        x = self.Conv1(x)
        #x = self.LRN_OP(x)
        x = self.MaxPool1(x)
        x = self.BN2(x)
        x = self.Conv2(x)
        #x = self.LRN_OP(x)
        x = self.MaxPool2(x)
        x = self.Conv3(x)
        x = self.Conv4(x)
        x = self.Conv5(x)
        x = self.MaxPool3(x)
        x = self.Flatten(x)
        x = self.Linear1(x)
        x = self.Linear2(x)
        x = self.Linear3(x)
        return x
##################################################

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
Net = MyNet(num_class)

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

    for imgs, label in train_dataset:
        train_batch(imgs, label)

    for imgs, label in test_dataset:
        predict_test_batch(imgs, label)

    print(
        f'Epoch {epoch + 1}, '
        f'Loss: {train_loss.result()}, '
        f'Accuracy: {train_accuracy.result() * 100}, '
        f'Test Loss: {test_loss.result()}, '
        f'Test Accuracy: {test_accuracy.result() * 100}'
    )
############################################################


