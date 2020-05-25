#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 19 22:52:57 2020

@author: eileen
"""


import os
WORKDIR = '/homes/eileen/CIFAR10/tfscripts'
os.chdir(WORKDIR)
os.environ['CUDA_VISIBLE_DEVICE'] = "1"

import data_tool
import resnet
import densenet
import plot

import numpy as np
from scipy.special import softmax
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import log_loss

import tensorflow as tf

import argparse


parser = argparse.ArgumentParser(description='TF CIFAR10 Training')
parser.add_argument('--ba', default=128, help='batchsize')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--model', default='densenet', type=str, help='choose model type')
parser.add_argument('--maxEpoch', default=50, type=int, help='max number of epoch')
parser.add_argument('--augType', default='aug2', type=str, help='augment type')
parser.add_argument('--verbose', default=0, type=int, help='verbose type')

args = parser.parse_args()

BEST_ACC = 0.0

#%% Variable
DATAROOT = '../data/cifar-10-batches-py'
TRAINLIST = ['data_batch_1','data_batch_2','data_batch_3','data_batch_4','data_batch_5']
TESTLIST = ['test_batch']
CLASSES = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


NUMWORKERS = 2

MODEL = {'resnet18': resnet.ResNet18(), 'densenet': densenet.densenet_cifar()}
AUGMENT = {'aug1': data_tool.ImgAugTransform1(), 'aug2': data_tool.ImgAugTransform2()}


#%% Dataloader
print('==> Preparing data..')
tr_augment = AUGMENT[args.augType] 
trainSet = data_tool.CIFAR10(dataInfo=[DATAROOT, TRAINLIST], batch_size=args.ba, augment=tr_augment)
testSet = data_tool.CIFAR10(dataInfo=[DATAROOT, TESTLIST], batch_size=100)


#%% Model
print('==> Building model..')
net = MODEL[args.model]
net.build((args.ba, 3, 32,32))


criterion = tf.keras.losses.CategoricalCrossentropy(from_logits=True) #(y_true, y_pred)
optimizer =tf.keras.optimizers.Adam(learning_rate=args.lr)


# Training
net.compile(loss=criterion, optimizer=optimizer, metrics=['acc'])
 
modName = '{}-ba{}-lr{}-{}'.format(args.model, args.ba, args.lr, args.augType)
logDir = '../logs/{}'.format(modName)
# checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=logDir+'/tracc_{acc:.4f}.h5',  
#                             verbose=0, 
#                             save_weights_only=False,
#                             save_best_only=False,
#                             )

# tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logDir)
tf_logger_train = tf.summary.create_file_writer(logDir + "/train")
tf_logger_train.set_as_default()
tf_logger_test = tf.summary.create_file_writer(logDir + "/test")
tf_logger_test.set_as_default()



#%%
best_model = None
for epoch in range(args.maxEpoch):
    history = net.fit(trainSet, epochs=1, steps_per_epoch=trainSet.__len__(), 
            verbose=args.verbose, shuffle=False)
    lossTrain, accTrain = history.history['loss'][0], history.history['acc'][0]
    
    y_true = []
    y_pred = []
    prob = []
    for idx, sample_batch in enumerate(testSet):
        x = sample_batch[0]
        pred_tmp = net.predict(x)
        y_true.append(np.argmax(sample_batch[1], axis=1))
        y_pred.append(np.argmax(pred_tmp, axis=1))
        prob.append(softmax(pred_tmp, axis=1))
     
    lossTest = log_loss(np.hstack(y_true), np.vstack(prob))
    accTest = accuracy_score(np.hstack(y_true), np.hstack(y_pred))
    preTest = precision_score(np.hstack(y_true), np.hstack(y_pred), average=None)[0]
    recTest = recall_score(np.hstack(y_true), np.hstack(y_pred), average=None)[0]
    
    cmTest = confusion_matrix(np.hstack(y_true), np.hstack(y_pred))
    cm_image = plot.plot_confusion_matrix(cmTest, CLASSES)
    cm_image = plot.plot_to_image(cm_image)
    
    rocTest = [np.hstack(y_true),np.vstack(prob)]
    roc_image = plot.plot_roc_curve(rocTest[0], rocTest[1])
    roc_image = plot.plot_to_image(roc_image)
    
 
    # logging       
    with tf_logger_train.as_default():
        tf.summary.scalar('acc', data=accTrain, step=epoch)
        tf.summary.scalar('loss', data=lossTrain, step=epoch)
        tf_logger_train.flush()

    with tf_logger_test.as_default():
        tf.summary.scalar('acc', data=accTest, step=epoch)
        tf.summary.scalar('loss', data=lossTest, step=epoch)
        tf.summary.scalar('preTest', data=preTest, step=epoch)
        tf.summary.scalar('recTest', data=recTest, step=epoch)
        tf.summary.image("Confusion Matrix", cm_image, step=epoch)
        tf.summary.image("ROC Curve", roc_image, step=epoch)
        tf_logger_test.flush()
        
    # Save model weights
    if accTest>BEST_ACC:
        best_model = net.get_weights()
        BEST_ACC = accTest

    
# save best model
net.set_weights(best_model)
net.save_weights(logDir+'/ts_acc{:.3}.h5'.format(BEST_ACC))
    
    