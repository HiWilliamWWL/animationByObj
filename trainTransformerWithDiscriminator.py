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
import losses
import discriminatorManager

#128 8 256

max_target_len = 85
#'''
'''

'''


batch_size = ABOtransformer.batch_size
#epoch_num = 5

val_data_num = 4*2

optimizer = keras.optimizers.Adam(0.001)
#model.compile(optimizer=optimizer, loss="mse", metrics="mse")
#model2.compile(optimizer=optimizer, loss=ABOtransformer.maskLabelLoss)


loader = dataLoader.trainDataLoader()
#full_dataset = full_dataset.shuffle(2)

discriminator_model = discriminatorManager.getDiscriminatorModel()
discriminator_model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(from_logits=False), metrics=['binary_accuracy'])
#discriminator_model.compile(optimizer='adam', loss=discriminatorManager.lossFn, metrics=['binary_accuracy'])



checkPointFolder = './Checkpoints_withDiscrimin2/'
#checkPointFolder = './Checkpoints_withoutDiscrimin1/'

def startTrain(model = None, dataset = None, restart = False, ep=5, trainDiscriminator = False, useDiscriminator = False):
  global discriminator_model, val_data_num, model_for_test
  if restart:
    model.load_weights(checkPointFolder+"cp.ckpt")
    print("weight loaded")
  if useDiscriminator:
    model.setDiscriminator(discriminator_model)
  full_dataset = dataset.batch(batch_size)
  val_dataset = full_dataset.take(val_data_num) 
  train_dataset = full_dataset.skip(val_data_num)
  if not os.path.exists(checkPointFolder):
    os.makedirs(checkPointFolder)
  if not os.path.exists(checkPointFolder):
    os.makedirs(checkPointFolder)
  checkpoints = []
  #fileName = "best_weights_{val_loss:.4f}.hdf5"
  fileName = "cp.ckpt"
  checkpoints.append(keras.callbacks.ModelCheckpoint(checkPointFolder+fileName, monitor='val_final_result_maskMSE', verbose=0, save_best_only=True, save_weights_only=True, mode='auto', period=1))
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
      testStart = testStart.reshape((1,1,63))
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
      discriminator_dataX.append(generate_human_pose[:, :])
      discriminator_dataX.append(gt_human_pose[:, :])
      discriminator_dataY.append(1.0)
      discriminator_dataY.append(0.0)
    ABOtransformer.batch_size = batch_size
    discriminator_dataSet = discriminatorManager.getDataset(discriminator_dataX, discriminator_dataY)
    discriminator_dataSet = discriminator_dataSet.batch(val_data_num)
    discriminator_model.fit(discriminator_dataSet, epochs = ep * 10, validation_data=discriminator_dataSet, shuffle=True, callbacks=checkpoints)
  


modelsSet = [ ABOtransformer.Transformer_VAEinitalPose(
  num_hid=128,
  num_head=8,
  num_feed_forward=128,
  source_maxlen=max_target_len,
  target_maxlen=max_target_len,
  num_layers_enc=4,
  num_layers_dec=3,
  num_classes=63,
  scheule_sampling_gt_rate = (3-i) * 25) for i in range(4)]

for i in range(4):
  modelsSet[i].compile(optimizer=optimizer, loss={"final_result": losses.maskLabelLoss, "intial_human": losses.initialPoseLoss},
    metrics={"final_result": losses.maskMSE, "intial_human": "mse"})
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
  num_classes=63)
model_for_test.compile(optimizer=optimizer, loss={"final_result": losses.maskLabelLoss, "intial_human": losses.initialPoseLoss},
  metrics={"final_result": losses.maskMSE, "intial_human": losses.initialPoseLoss})
ABOtransformer.batch_size = batch_size


full_dataset = loader.getDataset3()
startTrain(modelsSet[0], full_dataset, False, 50, True, False) #50
full_dataset = loader.getDataset3()
startTrain(modelsSet[1], full_dataset, True, 80, True, True)  #80
full_dataset = loader.getDataset3()
startTrain(modelsSet[2], full_dataset, True, 50, True, True)
full_dataset = loader.getDataset3()
startTrain(modelsSet[3], full_dataset, True, 120, False, True)

print("train finished")