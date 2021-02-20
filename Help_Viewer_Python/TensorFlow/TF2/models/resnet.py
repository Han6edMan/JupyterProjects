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



class LambdaLayer(keras.Model):
    def __init__(self, lamda):
        super(LambdaLayer, self).__init__()
        self.lamda = lamda

    def call(self, x):
        return self.lamda(x)


class ResNetBlock(keras.Model):
    expansion = 1
    def __init__(self, nfilters, strides=1, option="B", **kwargs):
        super(ResNetBlock, self).__init__(**kwargs)
        self.conv1 = conv3x3(nfilters, strides=strides)
        self.conv2 = conv3x3(nfilters)
        self.bn1 = layers.BatchNormalization()
        self.bn2 = layers.BatchNormalization()
        self.relu = layers.ReLU()
        if strides != 1:
            if option == "A":
                self.residual = LambdaLayer(lambda x: tf.pad(
                    x[:, :, ::2, ::2],
                    [[0, 0], [nfilters//4, nfilters//4], [0, 0], [0, 0]],
                    mode="CONSTANT", constant_values=0
                ))
            elif option == "B":
                self.residual = keras.Sequential()
                self.residual.add(
                    conv3x3(nfilters),
                    layers.BatchNormalization()
                )


    def call(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)) + self.residual(x))
        return out


class ResNet(keras.Model):
        def __init__(self, block, nfilters, nblocks, **kwargs):
        super(RefineNet, self).__init__(**kwargs)
        self.conv1 = layers.Conv2D(64, 7, 2, "same", "channels_first", use_bias=False)
        self.bn1 = layers.BatchNormalization()
        self.relu1 = layers.ReLU()
        self.resnet_block1 = self._make_resnet_block(block, nfilters[0], nblocks[0], stride=1)
        self.resnet_block2 = self._make_resnet_block(block, nfilters[1], nblocks[1], stride=2)
        self.resnet_block3 = self._make_resnet_block(block, nfilters[2], nblocks[2], stride=2)

    def _make_resnet_block(self, basicblock, nfilters, nblocks, stride=1):
        strides = [stride] + [1] * (nblocks - 1)
        blocks = keras.Sequential()
        for stride in strides:
            blocks.add(basicblock(nfilters, stride))
        return blocks
    
    def call(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        b1 = self.resnet_block1(x)
        b2 = self.resnet_block2(b1)
        b3 = self.resnet_block3(b2)