import keras
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
batch_size = 96
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

def Image_Resize(img, label):
    img = tf.image.resize(img, [56, 56])
    return img, label

train_image, train_label = Load_Cifar_10_Train_Dataset(Train_file_path)
train_image, train_label = Image_Resize(train_image, train_label)
test_image, test_label = Load_Cifar_10_Test_Dataset(Test_file_path)
test_image, test_label = Image_Resize(test_image, test_label)

train_dataset = tf.data.Dataset.from_tensor_slices((train_image, train_label)).shuffle(1000).batch(batch_size)
test_dataset = tf.data.Dataset.from_tensor_slices((test_image, test_label)).shuffle(1000).batch(batch_size)
#######################################

class Dense_Layer(tf.keras.Model):
    def __init__(self, growth_rate, bn_size, drop_rate):
        super(Dense_Layer, self).__init__()
        self.BN1 = tf.keras.layers.BatchNormalization()
        self.Relu = ReLU()
        self.Conv1 = Conv2D(filters=growth_rate * bn_size, kernel_size=1, strides=1, padding='same')
        self.BN2 = tf.keras.layers.BatchNormalization()
        self.Conv2 = Conv2D(growth_rate, kernel_size=3, strides=1, padding='same')

        self.drop_rate = drop_rate
        self.dropout = Dropout(drop_rate)

    def call(self, input):
        if isinstance(input, tf.Tensor):
            previous_feature = [input]
        else:
            previous_feature = input
        x = tf.concat(previous_feature, axis=-1)

        x = self.BN1(x)
        x = self.Relu(x)
        x = self.Conv1(x)  # BottleNeck output

        x = self.BN2(x)
        x = self.Relu(x)
        x = self.Conv2(x)  # new feature
        if self.drop_rate > 0:
            x = self.dropout(x)

        return x

class DenseBlock(tf.keras.Sequential):
    def __init__(self, num_layers, growth_rate, bn_size, drop_rate):
        super(DenseBlock, self).__init__()
        for i in range(num_layers):
            self.add(Dense_Layer(growth_rate, bn_size, drop_rate))

    def call(self, x):
        features = [x]
        for layer in self.layers:
            new_feature = layer(features)
            features.append(new_feature)
        return tf.concat(features, -1)

class Transition_layer(tf.keras.Model):
    def __init__(self, output_channel):
        super(Transition_layer, self).__init__()
        self.BN1 = tf.keras.layers.BatchNormalization()
        self.Relu = ReLU()
        self.Conv1 = Conv2D(filters=output_channel, kernel_size=1, strides=1, padding='same')
        self.AvgPool = AvgPool2D(pool_size=2, strides=2)

    def call(self, x):
        x = self.BN1(x)
        x = self.Relu(x)
        x = self.Conv1(x)
        x = self.AvgPool(x)

        return x

class DenseNet_ImageNet(tf.keras.Model):
    def __init__(self, growth_rate=12, block_config=(6, 12, 24, 16), initial_channel=64, bn_size=4, drop_rate=0, num_class=1000):
        super(DenseNet_ImageNet, self).__init__()
        self.Conv1 = Conv2D(filters=initial_channel, kernel_size=7, strides=2, padding='same')
        self.BN1 = tf.keras.layers.BatchNormalization()
        self.ReLu = ReLU()
        self.MaxPool = MaxPool2D(pool_size=3, strides=2, padding='same')

        channels = initial_channel
        self.blocks = []
        for i, num in enumerate(block_config):
            layer = DenseBlock(num, growth_rate, bn_size, drop_rate)
            self.blocks.append(layer)
            channels += growth_rate * num
            if i != len(block_config) - 1:
                self.blocks.append(Transition_layer(channels // 2))
                channels //= 2
        self.blocks.append(tf.keras.layers.BatchNormalization())
        self.blocks = Sequential(self.blocks)
        self.Global_Avg = tfa.layers.AdaptiveAveragePooling2D((1, 1))
        self.flatten = Flatten()
        self.Linear = Dense(num_class)

    def call(self, x):
        x = self.Conv1(x)
        print(x.shape)
        x = self.BN1(x)
        x = self.ReLu(x)
        x = self.MaxPool(x)
        print(x.shape)
        x = self.blocks(x)
        print(x.shape)
        x = self.ReLu(x)

        x = self.Global_Avg(x)
        print(x.shape)
        x = self.flatten(x)
        print(x.shape)
        x = self.Linear(x)
        print(x.shape)
        return x

class DenseNet_Cifar(tf.keras.Model):
    def __init__(self, growth_rate=12, block_config=(6, 12, 24, 16), initial_channel=64, bn_size=4, drop_rate=0, num_class=10):
        super(DenseNet_Cifar, self).__init__()
        self.Conv1 = Conv2D(filters=initial_channel, kernel_size=3, strides=1, padding='same')
        self.BN1 = tf.keras.layers.BatchNormalization()
        self.ReLu = ReLU()
        #self.MaxPool = MaxPool2D(pool_size=3, strides=2, padding='same')

        channels = initial_channel
        self.blocks = []
        for i, num in enumerate(block_config):
            layer = DenseBlock(num, growth_rate, bn_size, drop_rate)
            self.blocks.append(layer)
            channels += growth_rate * num
            if i != len(block_config) - 1:
                self.blocks.append(Transition_layer(channels // 2))
                channels //= 2
        self.blocks.append(tf.keras.layers.BatchNormalization())
        self.blocks = Sequential(self.blocks)
        self.Global_Avg = tfa.layers.AdaptiveAveragePooling2D((1, 1))
        self.flatten = Flatten()
        self.Linear = Dense(num_class)

    def call(self, x):
        x = self.Conv1(x)

        x = self.BN1(x)
        x = self.ReLu(x)
        #x = self.MaxPool(x)

        x = self.blocks(x)

        x = self.ReLu(x)

        x = self.Global_Avg(x)

        x = self.flatten(x)

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
Net = DenseNet_Cifar()

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

