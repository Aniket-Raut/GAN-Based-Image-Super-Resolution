from tensorflow.python.keras.layers import Add, BatchNormalization, Conv2D, Dense, Flatten, Input, LeakyReLU, PReLU, Lambda
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.applications.vgg19 import VGG19
import tensorflow as tf


def residual_block(x_in):
    x = Conv2D(64, kernel_size=3, padding='same')(x_in)
    x = BatchNormalization(momentum=0.8)(x)
    x = PReLU(shared_axes=[1, 2])(x)
    x = Conv2D(64, kernel_size=3, padding='same')(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = Add()([x_in, x])
    return x


def Generator():
    inp = Input(shape=(None, None, 3))
    layer = inp/255.0

    layer = Conv2D(64, kernel_size=9, padding='same')(layer)
    layer = temp = PReLU(shared_axes=[1, 2])(layer)

    # ===============================================
    # Residual Blocks
    # ===============================================
    for i in range(16):
        layer = residual_block(layer)

    layer = Conv2D(64, kernel_size=3, padding='same')(layer)
    layer = BatchNormalization()(layer)
    layer = Add()([temp, layer])

    layer = Conv2D(256, kernel_size=3, padding='same')(layer)
    layer = tf.nn.depth_to_space(layer,block_size=2)
    layer = PReLU(shared_axes=[1, 2])(layer)

    layer = Conv2D(256, kernel_size=3, padding='same')(layer)
    layer = tf.nn.depth_to_space(layer, block_size=2)
    layer = PReLU(shared_axes=[1, 2])(layer)


    layer = Conv2D(3, kernel_size=9, padding='same')(layer)
    layer = (layer + 1) * 127.5

    return Model(inp, layer)


generator = Generator


def d_block(x_in, num_filters, strides=1, batchnorm=True, momentum=0.8):
    x = Conv2D(num_filters, kernel_size=3, strides=strides, padding='same')(x_in)
    if batchnorm:
        x = BatchNormalization(momentum=momentum)(x)
    return LeakyReLU(alpha=0.2)(x)


def discriminator(num_filters=64):
    inp = Input(shape=(384,384, 3))
    layer = inp / 127.5 - 1

    layer = d_block(layer, num_filters, batchnorm=False)
    layer = d_block(layer, num_filters, strides=2)

    layer = d_block(layer, num_filters * 2)
    layer = d_block(layer, num_filters * 2, strides=2)

    layer = d_block(layer, num_filters * 4)
    layer = d_block(layer, num_filters * 4, strides=2)

    layer = d_block(layer, num_filters * 8)
    layer = d_block(layer, num_filters * 8, strides=2)

    layer = Flatten()(layer)

    layer = Dense(512)(layer)
    layer = LeakyReLU(alpha=0.2)(layer)
    layer = Dense(1, activation='sigmoid')(layer)

    return Model(inp, layer)


def vgg19():
    vgg = VGG19(input_shape=(None, None, 3), include_top=False)
    return Model(vgg.input, vgg.layers[5].output)
