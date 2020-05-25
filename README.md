# CIFAR10-Tensorflow
* Data
The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images. The dataset can be downloaded from the [link](https://www.cs.toronto.edu/~kriz/cifar.html).
Create your own folder "./data" in the CIFAR10-Tensorflow folder and download the dateset to this path.
* Requirements
Python, Tensorflow, imgaug... the packages and corresponding version are listed in **Requirements.txt**.

* Scripts
    * data_tool.py: including the self-built dataloader (inherit from tf.keras.utils.Sequence) and image augmentation function (make use of [imgaug](https://github.com/aleju/imgaug)).
    * resnet.py and densenet.py: the tensorflow models built by the sub-classing method.
    * main.py: the main code for training and testing the model, evaluation scores including accuracy, precision, recall and AUC.
    * plot.py: define the function of plotting the confusion matrix and ROC curve.

* Results
    * Show results in Tensorboard<br>
    `tensorboard --logdir=./logs/resnet18-ba128-lr0.001-aug2`<br>
    * Comparison between different augmentation methods:<br>
    (1) GaussianNoise + GaussianBlur + rotation (**overfitting**!) 
    ![](https://i.imgur.com/lt7i0Mc.png)
    (2) GaussianNoise + GaussianBlur + rotation
    ![](https://i.imgur.com/tQ8sMhs.png)
