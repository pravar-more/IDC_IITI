#!/usr/bin/env python
# coding: utf-8

# In[2]:


import tensorflow as tf
from tensorflow.keras import models, layers
import matplotlib.pyplot as plt
from IPython.display import HTML
import os


def trainModel():
    BATCH_SIZE = 32
    IMAGE_SIZE = 256
    CHANNELS=3
    EPOCHS=1

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

    class_names = train_dataset.class_names
    print(class_names)

    for image_batch, labels_batch in train_dataset.take(1):
        print(image_batch.shape)
        print(labels_batch.numpy())

    print(len(train_dataset))
    print(len(val_dataset))
    train_ds = train_dataset
    val_ds = val_dataset

    #DataSet Pipelinign

    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)

    #Building Model
    #DataSet Preprocessing
    resize_and_rescale = tf.keras.Sequential([
      layers.experimental.preprocessing.Resizing(IMAGE_SIZE, IMAGE_SIZE),
      layers.experimental.preprocessing.Rescaling(1.0/255),
    ])

    #Dqata Augment
    data_augmentation = tf.keras.Sequential([
      layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
      layers.experimental.preprocessing.RandomRotation(0.2),
    ])

    train_ds = train_ds.map(
        lambda x, y: (data_augmentation(x, training=True), y)
    ).prefetch(buffer_size=tf.data.AUTOTUNE)

    #Model Architecture
    INPUT_SHAPE = (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
    n_classes = 3

    model = models.Sequential([
        resize_and_rescale,
        data_augmentation,
        layers.Conv2D(32, kernel_size = (3,3), activation='relu', input_shape=INPUT_SHAPE),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64,  kernel_size = (3,3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64,  kernel_size = (3,3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(n_classes, activation='softmax'),
    ])

    #Model Build
    model.build(input_shape=INPUT_SHAPE)
    #print(model.summary())

    #Model compile
    model.compile(
        optimizer='adam',
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=['accuracy']
    )

    #model fit
    historyy = model.fit(
        train_ds,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1,
        validation_data=val_ds
    )

    return model


def evaluateModel(model,image):
    #evaluate model
    result = model.evaluate(image)

    '''#Save Model
    model_version = max([int(i) for i in os.listdir("saved_models_of.py")] +[0])+1 
    model.save(f"saved_models/{model_version}")'''
    
    return result


'''


def main():
    BATCH_SIZE = 32
    IMAGE_SIZE = 256
    CHANNELS=3
    EPOCHS=5
    
    test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        "res/TEST",
        seed=123,
        shuffle=False,
        image_size=(IMAGE_SIZE,IMAGE_SIZE),
        batch_size=BATCH_SIZE
    )
    model=trainModel()
    print(evaluateModel(model,image))
    return
    
    
    
    
if __name__=="__main__":
    main()
'''


# In[ ]:




