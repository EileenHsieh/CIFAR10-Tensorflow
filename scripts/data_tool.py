#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 18 14:42:20 2020

@author: eileen
"""



import os
import pickle
import numpy as np
import imgaug.augmenters as iaa

import tensorflow as tf


class CIFAR10(tf.keras.utils.Sequence):
    def __init__(self, dataInfo=['DataRoot', ['data_batch_1','data_batch_2']], batch_size=128, augment=None):
        # dataInfo = ['../data/cifar-10-batches-py', ['data_batch_1','data_batch_2','data_batch_3','data_batch_4','data_batch_5']]
        # load data
        self.dataRoot = dataInfo[0]
        self.dataLists = dataInfo[1]
        self.augment = augment
        self.batch_size = batch_size

        self.data = []
        self.label = []
        for filename in self.dataLists:
            file_path = os.path.join(self.dataRoot, filename)
            with open(file_path, 'rb') as fo:
                meta = pickle.load(fo, encoding='latin1')
                self.data.append(meta['data'].astype('float32'))
                self.label.append(meta['labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # (,height, width, channel)
        self.label = tf.keras.utils.to_categorical(np.hstack(self.label))
        
        
    def __len__(self):
        return len(self.data)//self.batch_size

    def __getitem__(self, idx):
        image = self.data[idx*self.batch_size: (idx+1)*self.batch_size]
        label = self.label[idx*self.batch_size: (idx+1)*self.batch_size]
        if self.augment:
            image = self.augment(image)
        image = image.transpose(0,3,1,2) # (, channel,height, width)
        

        # sample = {'image': image, 'label': label}
        
        return (image, label)


#%%  
class ImgAugTransform1:
  def __init__(self):
    self.aug = iaa.Sequential([
        iaa.AdditiveGaussianNoise(scale=(0, 0.2*255)),
        iaa.GaussianBlur(sigma=(0, 3.0)),
        iaa.Affine(rotate=(-45, 45))

    ]) 
  def __call__(self, img):
    img = np.array(img)
    return self.aug.augment_images(img)

class ImgAugTransform2:
  def __init__(self):
    self.aug = iaa.Sequential([
        iaa.Fliplr(0.5),
        iaa.GaussianBlur(sigma=(0, 3.0)),
        iaa.Affine(translate_px=(10, -10)),
        iaa.Affine(rotate=(-45, 45))
    ]) 
  def __call__(self, img):
    img = np.array(img)
    return self.aug.augment_images(img)


