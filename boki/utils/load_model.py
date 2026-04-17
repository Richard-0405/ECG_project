import numpy as np
from pathlib import Path
import keras
from keras.models import Model
from keras.layers import Input, Conv1D, Activation, Flatten, Dense, BatchNormalization

# path = Path(__file__).parent / "../weight"
path = 'C:/Users/CBIC/Desktop/env01/algo/ecgs/work/Rythm/weight'

def load_model():
# load weight
    conv1_weight = np.load(f'{path}/conv1_weight.npy')
    conv1_bias = np.load(f'{path}/conv1_bias.npy')
    batch1_layer = np.load(f'{path}/batch1_layer.npy')
    conv2_weight = np.load(f'{path}/conv2_weight.npy')
    conv2_bias = np.load(f'{path}/conv2_bias.npy')
    batch2_layer = np.load(f'{path}/batch2_layer.npy')
    conv3_weight = np.load(f'{path}/conv3_weight.npy')
    conv3_bias = np.load(f'{path}/conv3_bias.npy')
    batch3_layer = np.load(f'{path}/batch3_layer.npy')
    conv4_weight = np.load(f'{path}/conv4_weight.npy')
    conv4_bias = np.load(f'{path}/conv4_bias.npy')
    batch4_layer = np.load(f'{path}/batch4_layer.npy')
    conv5_weight = np.load(f'{path}/conv5_weight.npy')
    conv5_bias = np.load(f'{path}/conv5_bias.npy')
    batch5_layer = np.load(f'{path}/batch5_layer.npy')
    conv6_weight = np.load(f'{path}/conv6_weight.npy')
    conv6_bias = np.load(f'{path}/conv6_bias.npy')
    batch6_layer = np.load(f'{path}/batch6_layer.npy')
    dense_weight = np.load(f'{path}/dense_weight.npy')
    dense_bias = np.load(f'{path}/dense_bias.npy')

    input1 = Input(shape=(450, 1))
    x = Conv1D(filters=12, kernel_size=5, strides=4)(input1)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv1D(filters=9, kernel_size=5, strides=2)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv1D(filters=9, kernel_size=5, strides=2)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv1D(filters=9, kernel_size=3, strides=1)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv1D(filters=3, kernel_size=3, strides=1)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv1D(filters=3, kernel_size=3, strides=1)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Flatten()(x)
    x = Dense(4)(x)

    model = Model(input1, x)
    model.layers[1].trainable = False
    model.layers[2].trainable = False
    model.layers[4].trainable = False
    model.layers[5].trainable = False
    model.layers[7].trainable = False
    model.layers[8].trainable = False
    model.layers[10].trainable = False
    model.layers[11].trainable = False
    model.layers[13].trainable = False
    model.layers[14].trainable = False
    model.layers[16].trainable = False
    model.layers[17].trainable = False
    model.layers[20].trainable = True

    model.layers[1].set_weights([conv1_weight, conv1_bias])
    model.layers[2].set_weights(batch1_layer)
    model.layers[4].set_weights([conv2_weight, conv2_bias])
    model.layers[5].set_weights(batch2_layer)
    model.layers[7].set_weights([conv3_weight, conv3_bias])
    model.layers[8].set_weights(batch3_layer)
    model.layers[10].set_weights([conv4_weight, conv4_bias])
    model.layers[11].set_weights(batch4_layer)
    model.layers[13].set_weights([conv5_weight, conv5_bias])
    model.layers[14].set_weights(batch5_layer)
    model.layers[16].set_weights([conv6_weight, conv6_bias])
    model.layers[17].set_weights(batch6_layer)
    model.layers[20].set_weights([dense_weight, dense_bias])

    return model    