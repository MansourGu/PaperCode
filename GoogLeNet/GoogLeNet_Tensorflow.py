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



#############网络设计################

class Inception_v1(tf.keras.Model):
    def __init__(self, ch1x1, ch3x3_reduced, ch3x3, ch5x5_reduced, ch5x5, pool_proj):
        super(Inception_v1, self).__init__()

        self.branch1 = Sequential([Conv2D(filters=ch1x1, kernel_size=1, strides=1, padding='same', activation='relu'),])

        self.branch2 = Sequential([Conv2D(filters=ch3x3_reduced, kernel_size=1, strides=1, padding='same', activation='relu'),
                                   Conv2D(filters=ch3x3, kernel_size=3, strides=1, padding='same', activation='relu')])

        self.branch3 = Sequential([Conv2D(filters=ch5x5_reduced, kernel_size=1, strides=1, padding='same', activation='relu'),
                                   Conv2D(filters=ch5x5, kernel_size=3, strides=1, padding='same', activation='relu')])

        self.branch4 = Sequential([MaxPool2D(pool_size=3, strides=1, padding='same'),
                                   Conv2D(filters=pool_proj, kernel_size=1, strides=1, padding='same', activation='relu')])

    def _call(self, x):

        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        output = [branch1, branch2, branch3, branch4]
        return output

    def call(self, x):

        result = self._call(x)
        return tf.concat(result, -1)

class Aux_Logits(tf.keras.Model):
    def __init__(self, num_class):
        super(Aux_Logits, self).__init__()

        self.Conv1 = Conv2D(filters=num_class, kernel_size=1, strides=1, padding='same', activation='relu')
        self.Flatten = Flatten()
        self.Linear1 = Dense(2048, activation='relu')
        self.Linear2 = Dense(num_class, activation='relu')
        self.Dropout = Dropout(0.7)
        self.AvgPool = AvgPool2D(pool_size=5, strides=3, padding='valid')


    def call(self, x):
        x = self.AvgPool(x)
        x = self.Conv1(x)
        x = self.Flatten(x)
        x = self.Linear1(x)
        x = self.Dropout(x)
        x = self.Linear2(x)

        return x



class GoogLeNet(tf.keras.Model):

    def __init__(self, num_class):
        super(GoogLeNet, self).__init__()

        self.num_class = num_class
        #输入尺寸 (batch, 224, 224, 3)
        self.Conv1 = Conv2D(filters=64, kernel_size=7, strides=2, dilation_rate=2, activation='relu')
        # (batch, 112, 112, 64)
        self.MaxPool1 = MaxPool2D(pool_size=3, strides=2, padding='same')
        # (batch, 56, 56, 64)
        self.Conv2 = Conv2D(filters=64, kernel_size=1, strides=1, padding='same', activation='relu')
        # (batch, 56, 56, 64)
        self.Conv3 = Conv2D(filters=192, kernel_size=1, strides=1, padding='same', activation='relu')
        # (batch, 56, 56, 192)
        self.MaxPool2 = MaxPool2D(pool_size=3, strides=2, padding='same')
        # (batch, 28, 28, 192)

        self.Inception3a = Inception_v1(64, 96, 128, 16, 32, 32)
        # (batch, 28, 28, 256)
        self.Inception3b = Inception_v1(128, 128, 192, 32, 96, 64)
        # (batch, 28, 28, 480)
        self.MaxPool3 = MaxPool2D(pool_size=3, strides=2, padding='same')
        # (batch, 14, 14, 480)

        self.Inception4a = Inception_v1(192, 96, 208, 16, 48, 64)
        # (batch, 14, 14, 512)
        self.Inception4b = Inception_v1(160, 112, 224, 24, 64, 64)
        # (batch, 14, 14, 512)
        self.Inception4c = Inception_v1(128, 128, 256, 24, 64, 64)
        # (batch, 14, 14, 512)
        self.Inception4d = Inception_v1(112, 144, 288, 32, 64, 64)
        # (batch, 14, 14, 528)
        self.Inception4e = Inception_v1(256, 160, 320, 32, 128, 128)
        # (batch, 14, 14, 832)
        self.MaxPool4 = MaxPool2D(pool_size=3, strides=2, padding='same')

        self.aux1 = Aux_Logits(num_class)
        self.aux2 = Aux_Logits(num_class)

        self.Inception5a = Inception_v1(256, 160, 320, 32, 128, 128)
        # (batch, 7, 7, 832)
        self.Inception5b = Inception_v1(384, 192, 384, 48, 128, 128)
        # (batch, 7, 7, 1024)

        self.AvgPool1 = tfa.layers.AdaptiveAveragePooling2D(output_size=(1, 1), data_format='channels_last')
        # (batch, 1, 1, 1024)

        self.Flatten = Flatten()

        self.Linear1 = Dense(num_class)

    def call(self, x, training=True):

        x = self.Conv1(x)
        x = self.MaxPool1(x)

        x = self.Conv2(x)
        x = self.Conv3(x)
        x = self.MaxPool2(x)

        x = self.Inception3a(x)
        x = self.Inception3b(x)
        x = self.MaxPool3(x)

        x = self.Inception4a(x)
        if training:
            aux1 = self.aux1(x)
        else:
            aux1 = None

        x = self.Inception4b(x)
        x = self.Inception4c(x)
        x = self.Inception4d(x)

        if training:
            aux2 = self.aux2(x)
        else:
            aux2 = None

        x = self.Inception4e(x)
        x = self.MaxPool4(x)

        x = self.Inception5a(x)
        x = self.Inception5b(x)

        x = self.Avgpool1(x)

        x = self.Flatten(x)
        x = self.Dropout(x)
        x = self.Linear1(x)

        return x, aux2, aux1


class MyNet(tf.keras.Model):
    def __init__(self, num_class):
        super(MyNet, self).__init__()
        self.num_class = num_class
        # 输入尺寸 (batch, 56, 56, 3)
        self.Conv1 = Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu')
        # (batch, 56, 56, 64)
        #self.MaxPool1 = MaxPool2D(pool_size=3, strides=2, padding='same')
        # (batch, 56, 56, 64)
        self.Conv2 = Conv2D(filters=64, kernel_size=1, strides=1, padding='same', activation='relu')
        # (batch, 56, 56, 64)
        self.Conv3 = Conv2D(filters=192, kernel_size=1, strides=1, padding='same', activation='relu')
        # (batch, 56, 56, 192)
        self.MaxPool2 = MaxPool2D(pool_size=3, strides=2, padding='same')
        # (batch, 28, 28, 192)

        self.Inception3a = Inception_v1(64, 96, 128, 16, 32, 32)
        # (batch, 28, 28, 256)
        self.Inception3b = Inception_v1(128, 128, 192, 32, 96, 64)
        # (batch, 28, 28, 480)
        self.MaxPool3 = MaxPool2D(pool_size=3, strides=2, padding='same')
        # (batch, 14, 14, 480)

        self.Inception4a = Inception_v1(192, 96, 208, 16, 48, 64)
        # (batch, 14, 14, 512)
        self.Inception4b = Inception_v1(160, 112, 224, 24, 64, 64)
        # (batch, 14, 14, 512)
        self.Inception4c = Inception_v1(128, 128, 256, 24, 64, 64)
        # (batch, 14, 14, 512)
        self.Inception4d = Inception_v1(112, 144, 288, 32, 64, 64)
        # (batch, 14, 14, 528)
        self.Inception4e = Inception_v1(256, 160, 320, 32, 128, 128)
        # (batch, 14, 14, 832)
        self.MaxPool4 = MaxPool2D(pool_size=3, strides=2, padding='same')

        self.aux1 = Aux_Logits(num_class)
        self.aux2 = Aux_Logits(num_class)

        self.Inception5a = Inception_v1(256, 160, 320, 32, 128, 128)
        # (batch, 7, 7, 832)
        self.Inception5b = Inception_v1(384, 192, 384, 48, 128, 128)
        # (batch, 7, 7, 1024)

        self.AvgPool1 = tfa.layers.AdaptiveAveragePooling2D(output_size=(1, 1), data_format='channels_last')
        # (batch, 1, 1, 1024)

        self.Flatten = Flatten()

        self.Dropout = Dropout(0.2)

        self.Linear1 = Dense(num_class)

    def call(self, x, training=True):

        x = self.Conv1(x)
        #x = self.Maxpool1(x)

        x = self.Conv2(x)
        x = self.Conv3(x)
        x = self.MaxPool2(x)

        x = self.Inception3a(x)
        x = self.Inception3b(x)
        x = self.MaxPool3(x)

        x = self.Inception4a(x)
        if training:
            aux1 = self.aux1(x)
        else:
            aux1 = None

        x = self.Inception4b(x)
        x = self.Inception4c(x)
        x = self.Inception4d(x)

        if training:
            aux2 = self.aux2(x)
        else:
            aux2 = None

        x = self.Inception4e(x)
        x = self.MaxPool4(x)

        x = self.Inception5a(x)
        x = self.Inception5b(x)

        x = self.AvgPool1(x)

        x = self.Flatten(x)
        x = self.Dropout(x)
        x = self.Linear1(x)

        return x, aux2, aux1


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
        preds0, preds1, preds2 = Net(imgs, training=True)
        loss0 = loss_func(label, preds0)
        loss1 = loss_func(label, preds1)
        loss2 = loss_func(label, preds2)
        loss = loss0 + 0.3 * loss1 + 0.3 * loss2
    gradients = tape.gradient(loss, Net.trainable_variables)
    optimizer.apply_gradients(zip(gradients, Net.trainable_variables))

    train_loss(loss)
    train_accuracy(label, preds0)

def predict_test_batch(imgs, label):
    preds, _, _ = Net(imgs, training=False)
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



