import tensorflow as tf
try:
    import tensorflow.python.keras.layers as layers
except:
    import tensorflow.keras.layers as layers
try:
    import tensorflow.python.keras.models as models
except:
    import tensorflow.keras.models as models
import tensorflow.keras as keras
import tensorflow.nn as nn



class Baseline(keras.Model):
    def __init__(self):
        super(Baseline, self).__init__()
        self.conv = layers.Conv2D(filters=6, kernel_size=(5, 5), padding='same')
        self.bn = layers.BatchNormalization()
        self.relu = layers.ReLU()
        self.maxpool = layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='same')
        self.dropou1 = layers.Dropout(0.2)
        self.flatten = layers.Flatten()
        self.dence1 = layers.Dense(128, activation='relu')
        self.dropout2 = layers.Dropout(0.2)
        self.dence2 = layers.Dense(10, activation='softmax')

    def call(self, x):
        x = self.dropou1(self.maxpool(self.relu(self.bn(self.conv(x)))))
        x = self.flatten(x)
        y = self.dence2(self.dropout2(self.dence1(x)))
        return y


model = Baseline()
inputs = tf.Variable(tf.constant(range(1, 25), shape=[2, 3, 4, 1], dtype=tf.float32))
outputs = model(inputs)


# inputs = tf.Variable(inputs)

