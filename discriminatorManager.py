import numpy as np
import tensorflow as tf
import random

def getDiscriminatorModel(max_target_len):
    model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(max_target_len - 1, 111)),  #63, 54  54+63-3=114
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')])
    return model


def getDataset(discriminator_dataX, discriminator_dataY):
    newOrder = [x for x in range(len(discriminator_dataY))]
    random.shuffle(newOrder)
    dataX = []
    dataY = []
    for order in newOrder:
        dataX.append(discriminator_dataX[order])
        dataY.append(discriminator_dataY[order])
    dataset = tf.data.Dataset.from_tensor_slices((dataX, dataY))
    return dataset



