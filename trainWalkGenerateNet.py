import os
import random
from glob import glob
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import dataLoader_walk_seperate as dataLoader
#import dataLoader_walk as dataLoader

import numpy as np
import pickle
import walkGenerateNet as wgn

#128 8 256

max_target_len = 85 #85  35

batch_size = wgn.batch_size
val_data_num = 256
#val_data_num = 3

optimizer = keras.optimizers.Adam(0.001)

print("start loading data!!!")
loader = dataLoader.trainDataLoader(None, pathNames = ["./Data/walkData/*.data"])
print("Finished")
import losses
losses.loader = loader
losses.batch_size = batch_size

checkPointFolder = './Checkpoints/Checkpoints_wgn1/'
model = wgn.walkGenerateNet(num_hid=128, target_maxlen=max_target_len, num_classes=dataLoader.humanDimensionWalkOutput, num_experts = 3, num_input_dimension = dataLoader.objDimention)
model.compile(optimizer=optimizer, loss=losses.basicMSE, metrics='mean_squared_error')

full_dataset = loader.getDataset3()
full_dataset = full_dataset.shuffle(1000, reshuffle_each_iteration=True)
val_dataset = full_dataset.take(val_data_num).batch(batch_size) 
train_dataset = full_dataset.skip(val_data_num).batch(batch_size)
train_dataset = train_dataset.shuffle(1000, reshuffle_each_iteration=True)


if not os.path.exists(checkPointFolder):
  os.makedirs(checkPointFolder)
if not os.path.exists(checkPointFolder):
  os.makedirs(checkPointFolder)
checkpoints = []
#fileName = "best_weights_{val_loss:.4f}.hdf5"
fileName = "cp.ckpt"
checkpoints.append(keras.callbacks.ModelCheckpoint(checkPointFolder+fileName, monitor='val_mean_squared_error', verbose=0, save_best_only=True, save_weights_only=True, mode='auto', period=1))
checkpoints.append(keras.callbacks.TensorBoard(log_dir=checkPointFolder+'logs', histogram_freq=0, write_graph=True, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None))

#history_callback = model.fit(train_dataset, epochs = epoch_num, validation_data=val_dataset, shuffle=True, callbacks=checkpoints)
history_callback = model.fit(train_dataset, epochs = 300, validation_data=val_dataset, shuffle=True, callbacks=checkpoints)
loss_history = history_callback.history["loss"]
numpy_loss_history = np.array(loss_history)
np.savetxt(checkPointFolder+"the_history.txt", numpy_loss_history, delimiter=",")
print("saved current status")
print(model.summary())
#full_dataset = full_dataset.unbatch()