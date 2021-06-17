import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend
from customLayer import HASeparator
from params import parameters


class ResNet50():
    def __init__(self, classes, margin, scale):
        self.classes = classes
        self.margin = margin
        self.scale = scale

    def identity_block(self, input_tensor, kernel_size, filters, stage, block):

        filters1, filters2, filters3 = filters

        bn_axis = 3

        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        x = layers.Conv2D(filters1, (1, 1),
                          kernel_regularizer=tf.keras.regularizers.l2(
                              parameters["weight_decay"]),
                          name=conv_name_base + '2a')(input_tensor)
        x = layers.BatchNormalization(axis=bn_axis,
                                      name=bn_name_base + '2a')(x)
        x = layers.Activation('relu')(x)

        x = layers.Conv2D(filters2,
                          kernel_size,
                          padding='same',
                          kernel_regularizer=tf.keras.regularizers.l2(
                              parameters["weight_decay"]),
                          name=conv_name_base + '2b')(x)
        x = layers.BatchNormalization(axis=bn_axis,
                                      name=bn_name_base + '2b')(x)
        x = layers.Activation('relu')(x)

        x = layers.Conv2D(filters3, (1, 1),
                          kernel_regularizer=tf.keras.regularizers.l2(
                              parameters["weight_decay"]),
                          name=conv_name_base + '2c')(x)
        x = layers.BatchNormalization(axis=bn_axis,
                                      name=bn_name_base + '2c')(x)

        x = layers.add([x, input_tensor])
        x = layers.Activation('relu')(x)
        return x

    def conv_block(self,
                   input_tensor,
                   kernel_size,
                   filters,
                   stage,
                   block,
                   strides=(2, 2)):

        filters1, filters2, filters3 = filters

        bn_axis = 3

        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        x = layers.Conv2D(filters1, (1, 1),
                          strides=strides,
                          kernel_regularizer=tf.keras.regularizers.l2(
                              parameters["weight_decay"]),
                          name=conv_name_base + '2a')(input_tensor)
        x = layers.BatchNormalization(axis=bn_axis,
                                      name=bn_name_base + '2a')(x)
        x = layers.Activation('relu')(x)

        x = layers.Conv2D(filters2,
                          kernel_size,
                          padding='same',
                          kernel_regularizer=tf.keras.regularizers.l2(
                              parameters["weight_decay"]),
                          name=conv_name_base + '2b')(x)
        x = layers.BatchNormalization(axis=bn_axis,
                                      name=bn_name_base + '2b')(x)
        x = layers.Activation('relu')(x)

        x = layers.Conv2D(filters3, (1, 1),
                          kernel_regularizer=tf.keras.regularizers.l2(
                              parameters["weight_decay"]),
                          name=conv_name_base + '2c')(x)
        x = layers.BatchNormalization(axis=bn_axis,
                                      name=bn_name_base + '2c')(x)

        shortcut = layers.Conv2D(filters3, (1, 1),
                                 strides=strides,
                                 kernel_regularizer=tf.keras.regularizers.l2(
                                     parameters["weight_decay"]),
                                 name=conv_name_base + '1')(input_tensor)
        shortcut = layers.BatchNormalization(axis=bn_axis,
                                             name=bn_name_base + '1')(shortcut)

        x = layers.add([x, shortcut])
        x = layers.Activation('relu')(x)
        return x

    def model(self):

        labels = layers.Input([])
        inp = layers.Input(shape=parameters["input_shape"])
        bn_axis = 3

        x = layers.Conv2D(64, (3, 3),
                          padding='valid',
                          name='conv1',
                          kernel_regularizer=tf.keras.regularizers.l2(
                              parameters["weight_decay"]))(inp)
        x = layers.BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
        x = layers.Activation('relu')(x)

        x = self.conv_block(x,
                            3, [64, 64, 256],
                            stage=2,
                            block='a',
                            strides=(1, 1))
        x = self.identity_block(x, 3, [64, 64, 256], stage=2, block='b')
        x = self.identity_block(x, 3, [64, 64, 256], stage=2, block='c')

        x = self.conv_block(x, 3, [128, 128, 512], stage=3, block='a')
        x = self.identity_block(x, 3, [128, 128, 512], stage=3, block='b')
        x = self.identity_block(x, 3, [128, 128, 512], stage=3, block='c')
        x = self.identity_block(x, 3, [128, 128, 512], stage=3, block='d')

        x = self.conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
        x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
        x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
        x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
        x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
        x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

        x = self.conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
        x = self.identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
        x = self.identity_block(x, 3, [512, 512, 2048], stage=5, block='c')
        x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
        x = layers.BatchNormalization(axis=-1)(x)
        x = layers.Dropout(parameters["dropout"])(x)
        x = layers.Dense(64)(x)
        x = layers.BatchNormalization(axis=-1)(x)
        logits, pn_loss = HASeparator(self.classes, self.margin,
                                      self.scale)(x, labels)
        logits = tf.expand_dims(logits, 0)
        pn_loss = tf.expand_dims(pn_loss, 0)
        x = tf.keras.layers.Concatenate(axis=0)([logits, pn_loss])
        model = tf.keras.Model([inp, labels], outputs=x)
        model.compile(metrics=[self.accuracy],
                      optimizer=tf.keras.optimizers.SGD(
                          learning_rate=parameters["learning_rate"],
                          momentum=parameters["momentum"],
                          nesterov=parameters["nesterov"]),
                      loss=self.customLoss())

        return model

    def customLoss(self):
        def loss(y, x):
            class_loss = tf.keras.losses.sparse_categorical_crossentropy(
                y, x[0])
            cos_pen = tf.math.reduce_mean(tf.math.reduce_sum(x[1], axis=-1))
            return class_loss + cos_pen

        return loss

    def accuracy(self, labels, logits):
        return tf.keras.metrics.sparse_categorical_accuracy(labels, logits[0])

def scheduler(epoch):
    if epoch < 2:
        return 0.01
    elif epoch < 80:
        return 0.001
    elif epoch < 120:
        return 0.01