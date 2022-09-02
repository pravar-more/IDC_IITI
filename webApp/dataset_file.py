#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
from tensorflow.keras import models, layers
import matplotlib.pyplot as plt
from IPython.display import HTML
import os

#CONSTANTS
BATCH_SIZE = 32
IMAGE_SIZE = 256
CHANNELS=3
EPOCHS=1

#DATASET 
def dataset():
    train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        "res/TRAIN",
        seed=123,
        shuffle=False,
        image_size=(IMAGE_SIZE,IMAGE_SIZE),
        batch_size=BATCH_SIZE
    )
    val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        "res/TEST",
        seed=123,
        shuffle=False,
        image_size=(IMAGE_SIZE,IMAGE_SIZE),
        batch_size=BATCH_SIZE
    )
    
    return train_dataset,val_dataset

