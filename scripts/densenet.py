#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 21 10:28:11 2020

@author: eileen
"""

import math
import tensorflow as tf
from tensorflow import nn
import tensorflow.keras as F

class Bottleneck(F.Model):
    def __init__(self, in_planes, growth_rate):
        super(Bottleneck, self).__init__()
        self.bn1 = F.layers.BatchNormalization(axis=1)
        self.conv1 = F.layers.Conv2D(4*growth_rate, kernel_size=1, 
                                     use_bias=False, data_format='channels_first')
        self.bn2 = F.layers.BatchNormalization(axis=1)
        self.conv2 = F.layers.Conv2D(growth_rate, kernel_size=3, padding='same', 
                                     use_bias=False, data_format='channels_first')

    
    def call(self, x):       
        out = self.conv1(nn.relu(self.bn1(x)))
        out = self.conv2(nn.relu(self.bn2(out)))
        out = tf.concat([out,x], 1)
        return out


class Transition(F.Model):
    def __init__(self, in_planes, out_planes):
        super(Transition, self).__init__()
        self.bn = F.layers.BatchNormalization(axis=1)
        self.conv = F.layers.Conv2D(out_planes, kernel_size=1, 
                                    use_bias=False, data_format='channels_first')
        self.avgpool = F.layers.AveragePooling2D(2, data_format='channels_first')
        
  
    def call(self, x):       
        out = self.conv(nn.relu(self.bn(x)))
        out = self.avgpool(out)
        return out

class DenseNet(F.Model):
    def __init__(self, block, nblocks, growth_rate=12, reduction=0.5, num_classes=10):
        super(DenseNet, self).__init__()
        self.growth_rate = growth_rate

        num_planes = 2*growth_rate
        self.conv1 = F.layers.Conv2D(num_planes, kernel_size=3, padding='same', 
                                     use_bias=False, data_format='channels_first')

        self.dense1 = self._make_dense_layers(block, num_planes, nblocks[0])
        num_planes += nblocks[0]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans1 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense2 = self._make_dense_layers(block, num_planes, nblocks[1])
        num_planes += nblocks[1]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans2 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense3 = self._make_dense_layers(block, num_planes, nblocks[2])
        num_planes += nblocks[2]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans3 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense4 = self._make_dense_layers(block, num_planes, nblocks[3])
        num_planes += nblocks[3]*growth_rate

        self.bn = F.layers.BatchNormalization(axis=1)
        self.linear = F.layers.Dense(num_classes)
        self.avgpool = F.layers.AveragePooling2D(4, data_format='channels_first')
        self.flatten = F.layers.Flatten(data_format='channels_first')
        
    def _make_dense_layers(self, block, in_planes, nblock):
        layers = []
        for i in range(nblock):
            layers.append(block(in_planes, self.growth_rate))
            in_planes += self.growth_rate
        return F.Sequential(layers)
    
    def call(self, x):
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.trans3(self.dense3(out))
        out = self.dense4(out)
        out = self.avgpool(out)
        out = self.flatten(out)
        out = self.linear(out)
        return out


def DenseNet121():
    return DenseNet(Bottleneck, [6,12,24,16], growth_rate=32)

def DenseNet169():
    return DenseNet(Bottleneck, [6,12,32,32], growth_rate=32)

def DenseNet201():
    return DenseNet(Bottleneck, [6,12,48,32], growth_rate=32)

def DenseNet161():
    return DenseNet(Bottleneck, [6,12,36,24], growth_rate=48)

def densenet_cifar():
    return DenseNet(Bottleneck, [6,12,24,16], growth_rate=12)

def test(): 
    net = densenet_cifar()
    y = net(tf.random.normal((1, 3, 32, 32)))
    print(y.shape)
