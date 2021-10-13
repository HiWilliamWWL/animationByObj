import os
import random
from glob import glob
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import dataLoader_angles as dataLoader
import numpy as np
import pickle
import ABOtransformer_angles as ABOtransformer

import discriminatorManager

#128 8 256

max_target_len = 85 #85  35
#'''
'''

'''


batch_size = ABOtransformer.batch_size
#epoch_num = 5

val_data_num = 4*3
val_data_num_batch = 1

optimizer = keras.optimizers.Adam(0.001)
#model.compile(optimizer=optimizer, loss="mse", metrics="mse")
#model2.compile(optimizer=optimizer, loss=ABOtransformer.maskLabelLoss)

print("load data!!!")
loader = dataLoader.trainDataLoader(None, ["./Data/subData/*.data"])
#loader = dataLoader.trainDataLoader()
print("Finished")
import losses
losses.loader = loader
losses.batch_size = batch_size
#full_dataset = full_dataset.shuffle(2)

discriminator_model = discriminatorManager.getDiscriminatorModel(max_target_len)
discriminator_model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(from_logits=False), metrics=['binary_accuracy'])
#discriminator_model.compile(optimizer='adam', loss=discriminatorManager.lossFn, metrics=['binary_accuracy'])



checkPointFolder = './Checkpoints/Checkpoints_angles3/'
#checkPointFolder = './Checkpoints_withoutDiscrimin1/'

def startTrain(model = None, dataset = None, restart = False, ep=5, trainDiscriminator = False, useDiscriminator = False):
  global discriminator_model, val_data_num, model_for_test, val_data_num_batch
  if restart:
    model.load_weights(checkPointFolder+"cp.ckpt")
    print("weight loaded")
  if useDiscriminator:
    model.setDiscriminator(discriminator_model, checkPointFolder+"dis.ckpt")
  #full_dataset = dataset.batch(batch_size)
  #val_dataset = full_dataset.take(val_data_num_batch) 
  #train_dataset = full_dataset.skip(val_data_num_batch)

  #full_dataset = dataset.batch(batch_size)
  full_dataset = dataset
  full_dataset.shuffle(100, reshuffle_each_iteration=True)
  val_dataset = full_dataset.take(val_data_num).batch(batch_size) 
  train_dataset = full_dataset.skip(val_data_num).batch(batch_size)
  train_dataset.shuffle(100, reshuffle_each_iteration=True)


  if not os.path.exists(checkPointFolder):
    os.makedirs(checkPointFolder)
  if not os.path.exists(checkPointFolder):
    os.makedirs(checkPointFolder)
  checkpoints = []
  #fileName = "best_weights_{val_loss:.4f}.hdf5"
  fileName = "cp.ckpt"
  checkpoints.append(keras.callbacks.ModelCheckpoint(checkPointFolder+fileName, monitor='val_final_result_maskMSE_angles', verbose=0, save_best_only=True, save_weights_only=True, mode='auto', period=1))
  checkpoints.append(keras.callbacks.TensorBoard(log_dir=checkPointFolder+'logs', histogram_freq=0, write_graph=True, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None))

  #history_callback = model.fit(train_dataset, epochs = epoch_num, validation_data=val_dataset, shuffle=True, callbacks=checkpoints)
  history_callback = model.fit(train_dataset, epochs = ep, validation_data=val_dataset, shuffle=True, callbacks=checkpoints)
  loss_history = history_callback.history["loss"]
  numpy_loss_history = np.array(loss_history)
  np.savetxt(checkPointFolder+"the_history.txt", numpy_loss_history, delimiter=",")
  print("saved current status")
  print(model.summary())
  full_dataset = full_dataset.unbatch()
  
  if trainDiscriminator:
    ABOtransformer.batch_size = 1
    itrd = iter(val_dataset.unbatch())
    discriminator_dataX = []
    discriminator_dataY = []
    model_for_test.load_weights(checkPointFolder+"cp.ckpt")
    for i in range(val_data_num):
      x,y = next(itrd)
      x = tf.dtypes.cast(x, tf.float32)
      #x = tf.dtypes.cast(x, tf.float32)
      testx = np.copy(x.numpy())
      testStart = y.numpy()[0,:]
      testStart = testStart.reshape((1,1,dataLoader.humanDimension2 + dataLoader.humanDimension1))
      #testResult = model_for_test.generate(testx, testStart)

      
      testResult = model_for_test.predict_step((x,y))
      generate_human_pose = testResult.numpy()[0, :, :]
      gt_human_pose = y.numpy()[1:,:]
      '''
      for i in range(20):  #joints
        generate_human_pose[:, 1+i] = generate_human_pose[:, 1+i] + generate_human_pose[:, 0]
        gt_human_pose[:, 1+i] = gt_human_pose[:, 1+i] + gt_human_pose[:, 0]
      '''
      #for i in range(5, 70):  #time
      discriminator_dataX.append(generate_human_pose[:, 3:])
      discriminator_dataX.append(gt_human_pose[:, 3:])
      discriminator_dataY.append(1.0)
      discriminator_dataY.append(0.0)
    ABOtransformer.batch_size = batch_size
    discriminator_dataSet = discriminatorManager.getDataset(discriminator_dataX, discriminator_dataY)
    discriminator_dataSet = discriminator_dataSet.batch(val_data_num)
    cb = keras.callbacks.ModelCheckpoint(checkPointFolder+"dis.ckpt", monitor='val_binary_accuracy', verbose=0, save_best_only=True, save_weights_only=True, mode='auto', period=1)
    discriminator_model.fit(discriminator_dataSet, epochs = ep * 2, validation_data=discriminator_dataSet, shuffle=True, callbacks=[cb])

modelsSet = [ ABOtransformer.Transformer_VAEinitalPose(
  num_hid=128,
  num_head=8,
  num_feed_forward=128,
  source_maxlen=max_target_len,
  target_maxlen=max_target_len,
  num_layers_enc=4,
  num_layers_dec=3,
  num_classes=dataLoader.humanDimension2 + dataLoader.humanDimension1,
  scheule_sampling_gt_rate = (3-i) * 25) for i in range(4)]

for i in range(4):
  modelsSet[i].compile(optimizer=optimizer, loss={"final_result": losses.maskLabelLoss_angles, "intial_human": losses.initialPoseLoss_angles},
  metrics={"final_result": losses.maskMSE_angles, "intial_human": losses.initialPoseLoss_angles})
  #modelsSet[i].compile(optimizer=optimizer, loss=ABOtransformer.maskLabelLoss, metrics=ABOtransformer.maskLabelLoss)

ABOtransformer.batch_size = 1
print("changing to 1")
model_for_test = ABOtransformer.Transformer_VAEinitalPose(
  num_hid=128,
  num_head=8,
  num_feed_forward=128,
  source_maxlen=max_target_len,
  target_maxlen=max_target_len,
  num_layers_enc=4,
  num_layers_dec=3,
  num_classes=dataLoader.humanDimension2 + dataLoader.humanDimension1)
model_for_test.compile(optimizer=optimizer, loss={"final_result": losses.maskLabelLoss_angles, "intial_human": losses.initialPoseLoss_angles},
  metrics={"final_result": losses.maskMSE_angles, "intial_human": losses.initialPoseLoss_angles})
ABOtransformer.batch_size = batch_size



full_dataset = loader.getDataset3()

startTrain(modelsSet[0], full_dataset, False, 100, True, False) #1
full_dataset = loader.getDataset3()
startTrain(modelsSet[1], full_dataset, True, 160, True, True)  #2
full_dataset = loader.getDataset3()
startTrain(modelsSet[2], full_dataset, True, 100, True, True)  #3
full_dataset = loader.getDataset3()
startTrain(modelsSet[3], full_dataset, True, 200, False, True)
'''
startTrain(modelsSet[0], full_dataset, False, 100, False, False) #1
full_dataset = loader.getDataset3()
startTrain(modelsSet[1], full_dataset, True, 120, False, False)  #2
full_dataset = loader.getDataset3()
startTrain(modelsSet[3], full_dataset, True, 160, False, False)
#full_dataset = loader.getDataset3()
#startTrain(modelsSet[3], full_dataset, True, 80, False, True)
'''
print("train finished")