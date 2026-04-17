import numpy as np
import math
from pathlib import Path

path = Path(__file__).parent / "../weight"

def inference(test_x):
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
    # Inference
    ecg = test_x[:].reshape(450, 1)  # reshape input data

    conv1 = conv1D(conv1_weight, conv1_bias, ecg, 4)
    batch1 = batchnorm(conv1, batch1_layer)
    relu1 = relu(batch1)

    conv2 = conv1D(conv2_weight, conv2_bias, relu1, 2)
    batch2 = batchnorm(conv2, batch2_layer)
    relu2 = relu(batch2)

    conv3 = conv1D(conv3_weight, conv3_bias, relu2, 2)
    batch3 = batchnorm(conv3, batch3_layer)
    relu3 = relu(batch3)

    conv4 = conv1D(conv4_weight, conv4_bias, relu3, 1)
    batch4 = batchnorm(conv4, batch4_layer)
    relu4 = relu(batch4)

    conv5 = conv1D(conv5_weight, conv5_bias, relu4, 1)
    batch5 = batchnorm(conv5, batch5_layer)
    relu5 = relu(batch5)

    conv6 = conv1D(conv6_weight, conv4_bias, relu5, 1)
    batch6 = batchnorm(conv6, batch6_layer)
    relu6 = relu(batch6)

    flatten1 = flatten(relu6)

    dense1 = dense(flatten1, dense_weight, dense_bias)

    pred = softmax(dense1)

    return pred


def conv1D(weight, bias, input, stride):

    weight_shape = np.shape(weight)
    input_shape = np.shape(input)

    filt_length = weight_shape[0]
    in_channel = weight_shape[1]
    out_channel = weight_shape[2]
    in_length = input_shape[0]

    out_length = int(np.fix((in_length - filt_length) / stride) + 1)
    feature_map = np.zeros((out_length, out_channel))

    for i in range(out_channel):
        for j in range(out_length):
            tmp_channel = 0
            for k in range(in_channel):
                tmp_filt = 0
                for l in range(filt_length):
                    tmp_filt = tmp_filt + input[j*stride+l, k] * weight[l, k, i]

                tmp_channel = tmp_channel + tmp_filt
            tmp_channel = tmp_channel + bias[i] # add bias

            feature_map[j, i] = tmp_channel

    return feature_map

def relu(input):
    input_shape = np.shape(input)
    out_length = input_shape[0]
    out_channel = input_shape[1]

    feature_map = np.zeros((out_length, out_channel))

    for i in range(out_channel):
        for j in range(out_length):
            if (input[j,i] < 0):
                feature_map[j,i] = 0
            else:
                feature_map[j,i] = input[j,i]

    return  feature_map


def maxpool2(input):
    input_shape = np.shape(input)

    out_length = int(np.fix(input_shape[0] / 2))
    in_channel = input_shape[1]
    output = np.zeros((out_length, in_channel))

    for i in range(out_length):
        for j in range(in_channel):
            output[i, j] = max(input[2*i, j], input[2*i+1, j])

    return output


def batchnorm(input, weight):
    input_shape = np.shape(input)
    in_length = input_shape[0]
    in_channel = input_shape[1]

    output = np.zeros((in_length, in_channel))

    for i in range(in_channel):
        gamma = weight[0, i]
        beta = weight[1, i]
        mean = weight[2, i]
        var = weight[3, i]
        
        a = gamma/ ((var+0.001)**0.5)
        b = beta - gamma*mean/((var+0.001)**0.5)
    
        for j in range (in_length):
            output[j,i] = a*input[j,i] + b


    return output


def avgpool(input):
    input_shape = np.shape(input)
    in_length = input_shape[0]
    in_channel = input_shape[1]

    # length
    output = np.zeros((in_channel, 1))
    for i in range(in_channel):
        output[i] = np.mean(input[:,i])

    return output

def flatten(input):
    len = input.shape[0]
    channel = input.shape[1]
    out_size = len * channel
    output = np.zeros((out_size, 1))

    for i in range(len):
      for j in range(channel):
        output[j+i*channel] = input[i][j]

    return output

def dense(input, weight, bias):
    weight_shape = np.shape(weight)
    out_node = weight_shape[1]
    in_node = weight_shape[0]

    output = np.zeros((out_node, 1))
    for i in range(out_node):
        psum = 0
        for j in range(in_node):
            psum = psum + input[j]*weight[j, i]
        output[i] = psum + bias[i]

    return output


def softmax(input):
    max_idx = np.argmax(input)
    return max_idx

