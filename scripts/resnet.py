#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 18 13:20:36 2020

@author: eileen
"""


import tensorflow as tf
from tensorflow import nn
import tensorflow.keras as F


class BasicBlock(F.Model):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = F.layers.Conv2D(planes, kernel_size=3, 
                                     strides=stride, padding='same', 
                                     use_bias=False, data_format='channels_first')
        self.bn1 = F.layers.BatchNormalization(axis=1)
        self.conv2 = F.layers.Conv2D(planes, kernel_size=3,
                                     strides=1, padding='same', 
                                     use_bias=False, data_format='channels_first')
        self.bn2 = F.layers.BatchNormalization(axis=1)

        self.shortcut = F.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut.add(F.layers.Conv2D(self.expansion*planes, 
                                              kernel_size=1, strides=stride, 
                                              use_bias=False, data_format='channels_first'))
            self.shortcut.add(F.layers.BatchNormalization(axis=1))
        
    def call(self, x):       
        out = nn.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = nn.relu(out)
        return out



class Bottleneck(F.Model):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = F.layers.Conv2D(planes, kernel_size=1, use_bias=False, data_format='channels_first')
        self.bn1 = F.layers.BatchNormalization(axis=1)
        self.conv2 = F.layers.Conv2D(planes, kernel_size=3,
                               strides=stride, padding='same', use_bias=False, data_format='channels_first')
        self.bn2 = F.layers.BatchNormalization(axis=1)
        self.conv3 = F.layers.Conv2D(self.expansion*planes, kernel_size=1, use_bias=False, data_format='channels_first')
        self.bn3 = F.layers.BatchNormalization(axis=1)
        
        self.shortcut = F.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut.add(F.layers.Conv2D(self.expansion*planes, 
                                              kernel_size=1, strides=stride, use_bias=False))
            self.shortcut.add(F.layers.BatchNormalization(axis=1))
        
    def call(self, x):       
        out = nn.relu(self.bn1(self.conv1(x)))
        out = nn.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = nn.relu(out)
        return out   



class ResNet(F.Model):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = F.layers.Conv2D(64, kernel_size=3,
                               strides=1, padding='same', use_bias=False, data_format='channels_first')
        self.bn1 = F.layers.BatchNormalization(axis=1)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.flatten = F.layers.Flatten(data_format='channels_first')
        self.avgpool = F.layers.AveragePooling2D(4, data_format='channels_first')
        self.linear = F.layers.Dense(num_classes)
        

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return F.Sequential(layers)
    
    
    def call(self, x):
        out = nn.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = self.flatten(out)
        out = self.linear(out)
        return out


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])


def test(): 
    net = ResNet18()
    y = net(tf.random.normal((1, 3, 32, 32)))
    print(y.shape)
        
# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())

# tf.config.list_physical_devices






