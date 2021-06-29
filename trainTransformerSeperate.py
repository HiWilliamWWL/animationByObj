import os
import random
from glob import glob
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import dataLoader
import numpy as np
import pickle
import ABOtransformer

#128 8 256

max_target_len = 85
#'''
'''

'''


batch_size = ABOtransformer.batch_size
#epoch_num = 5


optimizer = keras.optimizers.Adam(0.001)
#model.compile(optimizer=optimizer, loss="mse", metrics="mse")
#model2.compile(optimizer=optimizer, loss=ABOtransformer.maskLabelLoss)


loader = dataLoader.trainDataLoader()
#full_dataset = full_dataset.shuffle(2)

checkPointFolder = './Checkpoints5/'

def startTrain(model = None, dataset = None, restart = False, ep=5):
  if restart:
    model.load_weights(checkPointFolder+"cp.ckpt")
    print("weight loaded")
  full_dataset = dataset.batch(batch_size)
  val_dataset = full_dataset.take(4) 
  train_dataset = full_dataset.skip(4)
  if not os.path.exists(checkPointFolder):
    os.makedirs(checkPointFolder)
  if not os.path.exists(checkPointFolder):
    os.makedirs(checkPointFolder)
  checkpoints = []
  #fileName = "best_weights_{val_loss:.4f}.hdf5"
  fileName = "cp.ckpt"
  checkpoints.append(keras.callbacks.ModelCheckpoint(checkPointFolder+fileName, monitor='loss', verbose=0, save_best_only=True, save_weights_only=True, mode='auto', period=1))
  checkpoints.append(keras.callbacks.TensorBoard(log_dir=checkPointFolder+'logs', histogram_freq=0, write_graph=True, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None))

  #history_callback = model.fit(train_dataset, epochs = epoch_num, validation_data=val_dataset, shuffle=True, callbacks=checkpoints)
  history_callback = model.fit(train_dataset, epochs = ep, validation_data=val_dataset, shuffle=True, callbacks=checkpoints)
  loss_history = history_callback.history["loss"]
  numpy_loss_history = np.array(loss_history)
  np.savetxt(checkPointFolder+"the_history.txt", numpy_loss_history, delimiter=",")
  print("saved")
  print("saved")
  print(model.summary())
  full_dataset = full_dataset.unbatch()

model1 = ABOtransformer.Transformer_newScheduleSampling(
  num_hid=128,
  num_head=8,
  num_feed_forward=128,
  source_maxlen=max_target_len,
  target_maxlen=max_target_len,
  num_layers_enc=4,
  num_layers_dec=3,
  num_classes=63,
  scheule_sampling_gt_rate = 80)
model1.compile(optimizer=optimizer, loss=ABOtransformer.maskLabelLoss, metrics=ABOtransformer.maskMSE)
model2 = ABOtransformer.Transformer_newScheduleSampling(
  num_hid=128,
  num_head=8,
  num_feed_forward=128,
  source_maxlen=max_target_len,
  target_maxlen=max_target_len,
  num_layers_enc=4,
  num_layers_dec=3,
  num_classes=63,
  scheule_sampling_gt_rate = 50)
model2.compile(optimizer=optimizer, loss=ABOtransformer.maskMSE, metrics=ABOtransformer.maskMSE)
model3 = ABOtransformer.Transformer_newScheduleSampling(
  num_hid=128,
  num_head=8,
  num_feed_forward=128,
  source_maxlen=max_target_len,
  target_maxlen=max_target_len,
  num_layers_enc=4,
  num_layers_dec=3,
  num_classes=63,
  scheule_sampling_gt_rate = 20)
model3.compile(optimizer=optimizer, loss=ABOtransformer.maskMSE, metrics=ABOtransformer.maskMSE)
model4 = ABOtransformer.Transformer_newScheduleSampling(
  num_hid=128,
  num_head=8,
  num_feed_forward=128,
  source_maxlen=max_target_len,
  target_maxlen=max_target_len,
  num_layers_enc=4,
  num_layers_dec=3,
  num_classes=63,
  scheule_sampling_gt_rate = 0)
model4.compile(optimizer=optimizer, loss=ABOtransformer.maskMSE, metrics=ABOtransformer.maskMSE)

full_dataset = loader.getDataset3()
startTrain(model1, full_dataset, False, 30)
full_dataset = loader.getDataset3()
startTrain(model2, full_dataset, True, 50)
full_dataset = loader.getDataset3()
startTrain(model3, full_dataset, True, 30)
full_dataset = loader.getDataset3()
startTrain(model4, full_dataset, True, 80)


print("train finished")