#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 09:52:39 2020
Implemented using TensorFlow 1.0 and TFLearn 0.3.2

Minghang Zhao, Shisheng Zhong, Xuyun Fu, Baoping Tang, Shaojiang Dong, Michael Pecht,
Deep Residual Networks with Adaptively Parametric Rectifier Linear Units for Fault Diagnosis, 
IEEE Transactions on Industrial Electronics, 2020,  DOI: 10.1109/TIE.2020.2972458 
 
@author: Minghang Zhao
"""

from __future__ import division, print_function, absolute_import

import tflearn
import tensorflow as tf
from tflearn.activations import relu
from tflearn.layers.normalization import batch_normalization as bn
from tflearn.layers.core import input_data, fully_connected
from tflearn.layers.conv import conv_2d, global_avg_pool
import tflearn.data_utils as du

# Data loading and preprocessing
import tflearn.datasets.mnist as mnist
X, Y, testX, testY = mnist.load_data(one_hot=True)
X = X.reshape([-1, 28, 28, 1])
testX = testX.reshape([-1, 28, 28, 1])
X, mean = du.featurewise_zero_center(X)
testX = du.featurewise_zero_center(testX, mean)

# An adaptively parametric rectifier linear unit (APReLU)
def aprelu(incoming):
    in_channels = incoming.get_shape().as_list()[-1]
    scales_n = tf.reduce_mean(tf.reduce_mean(tf.minimum(incoming,0),axis=2,keep_dims=True),axis=1,keep_dims=True)
    scales_p = tf.reduce_mean(tf.reduce_mean(tf.maximum(incoming,0),axis=2,keep_dims=True),axis=1,keep_dims=True)
    scales = tf.concat([scales_n, scales_p],axis=3)
    scales = fully_connected(scales, in_channels, activation='linear',regularizer='L2',
                             weight_decay=0.0001,weights_init='variance_scaling')
    scales = relu(bn(scales))
    scales = fully_connected(scales, in_channels, activation='linear',regularizer='L2',
                             weight_decay=0.0001,weights_init='variance_scaling')
    scales = tflearn.activations.sigmoid(bn(scales))
    scales = tf.expand_dims(tf.expand_dims(scales,axis=1),axis=1)
    return tf.maximum(incoming, 0) + tf.multiply(scales, (incoming - tf.abs(incoming))) * 0.5

# A residual building block with APReLUs
def res_block_aprelu(incoming, nb_blocks, out_channels, downsample=False,
                     downsample_strides=2, batch_norm=True,
                     bias=True, weights_init='variance_scaling',
                     bias_init='zeros', regularizer='L2', weight_decay=0.0001,
                     trainable=True, restore=True, reuse=False, scope=None,
                     name="ResidualBlock"):

    resnet = incoming
    in_channels = incoming.get_shape().as_list()[-1]

    # Variable Scope fix for older TF
    try:
        vscope = tf.variable_scope(scope, default_name=name, values=[incoming],
                                   reuse=reuse)
    except Exception:
        vscope = tf.variable_op_scope([incoming], scope, name, reuse=reuse)

    with vscope as scope:
        name = scope.name #TODO

        for i in range(nb_blocks):

            identity = resnet

            if not downsample:
                downsample_strides = 1

            if batch_norm:
                resnet = bn(resnet)
            resnet = aprelu(resnet)
            resnet = conv_2d(resnet, out_channels, 3,
                             downsample_strides, 'same', 'linear',
                             bias, weights_init, bias_init,
                             regularizer, weight_decay, trainable,
                             restore)

            if batch_norm:
                resnet = bn(resnet)
            resnet = aprelu(resnet)
            resnet = conv_2d(resnet, out_channels, 3, 1, 'same',
                             'linear', bias, weights_init,
                             bias_init, regularizer, weight_decay,
                             trainable, restore)

            # Downsampling
            if downsample_strides > 1:
                identity = tflearn.avg_pool_2d(identity, 1, downsample_strides)

            # Projection to new dimension
            if in_channels != out_channels:
                if (out_channels - in_channels) % 2 == 0:
                    ch = (out_channels - in_channels)//2
                    identity = tf.pad(identity,
                                      [[0, 0], [0, 0], [0, 0], [ch, ch]])
                else:
                    ch = (out_channels - in_channels)//2
                    identity = tf.pad(identity,
                                      [[0, 0], [0, 0], [0, 0], [ch, ch+1]])
                in_channels = out_channels

            resnet = resnet + identity

    return resnet

# Real-time data preprocessing
img_prep = tflearn.ImagePreprocessing()
img_prep.add_featurewise_zero_center(per_channel=True)

# Building Residual Network
net = input_data(shape=[None, 28, 28, 1],
                 data_preprocessing=img_prep)
net = conv_2d(net, 16, 3, regularizer='L2', weight_decay=0.0001)
net = res_block_aprelu(net, 1, 16)
net = res_block_aprelu(net, 1, 32, downsample=True)
net = res_block_aprelu(net, 1, 32)
net = aprelu(bn(net))
net = global_avg_pool(net)
# Regression
net = fully_connected(net, 10, activation='softmax')
mom = tflearn.Momentum(0.1, lr_decay=0.1, decay_step=2000, staircase=True)
net = tflearn.regression(net, optimizer=mom, loss='categorical_crossentropy')
# Training
model = tflearn.DNN(net, checkpoint_path='model_resnet_mnist',
                    max_checkpoints=10, tensorboard_verbose=0,
                    clip_gradients=0.)

model.fit(X, Y, n_epoch=10, snapshot_epoch=False, 
          snapshot_step=500, show_metric=True, batch_size=100, 
          shuffle=True, run_id='resnet_mnist')

training_acc = model.evaluate(X, Y)[0]
validation_acc = model.evaluate(testX, testY)[0]
