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

train_mode = False

max_target_len = 130
model = ABOtransformer.Transformer(
    num_hid=128,
    num_head=8,
    num_feed_forward=256,
    source_maxlen=max_target_len,
    target_maxlen=max_target_len,
    num_layers_enc=5,
    num_layers_dec=3,
    num_classes=63,
)



batch_size = 2
epoch_num = 50


optimizer = keras.optimizers.Adam(0.001)
#model.compile(optimizer=optimizer, loss="mse", metrics="mse")
model.compile(optimizer=optimizer, loss=ABOtransformer.maskLabelLoss, metrics=ABOtransformer.maskLabelLoss)


loader = dataLoader.trainDataLoader()
full_dataset = loader.getDataset3()
#full_dataset = full_dataset.shuffle(2)


if train_mode:

  full_dataset = full_dataset.batch(batch_size)
  val_dataset = full_dataset.take(6) 
  train_dataset = full_dataset.skip(6)
  if not os.path.exists('./Checkpoints1/'):
    os.makedirs('./Checkpoints1/')
  if not os.path.exists('./Checkpoints1/'):
    os.makedirs('./Checkpoints1/')
  checkpoints = []
  #fileName = "best_weights_{val_loss:.4f}.hdf5"
  fileName = "cp.ckpt"
  checkpoints.append(keras.callbacks.ModelCheckpoint('./Checkpoints1/'+fileName, monitor='loss', verbose=0, save_best_only=True, save_weights_only=True, mode='auto', period=1))
  checkpoints.append(keras.callbacks.TensorBoard(log_dir='./Checkpoints1/logs', histogram_freq=0, write_graph=True, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None))

  #history_callback = model.fit(train_dataset, epochs = epoch_num, validation_data=val_dataset, shuffle=True, callbacks=checkpoints)
  history_callback = model.fit(train_dataset, epochs = epoch_num, validation_data=val_dataset, shuffle=True, callbacks=checkpoints)
  loss_history = history_callback.history["loss"]
  numpy_loss_history = np.array(loss_history)
  np.savetxt("./Checkpoints1/the_history.txt", numpy_loss_history, delimiter=",")
  print("saved")
  # serialize weights to HDF5
  #model.save_weights("./Checkpoints1/weights.h5")


  #model.fit(full_dataset, epochs = 30)

  print(model.summary())
  full_dataset = full_dataset.unbatch()
else:
  #model(np.zeros((1,w,h,c)))
  model.load_weights("./Checkpoints1/cp.ckpt")


useTest = True
f1 = "./testResult_5b.pb"
f2 = "./testResult_5a.pb"
f3 = "./testResult_gt5.pb"
selectDataCount = 5

if useTest:
  print("1111")
  full_dataset = loader.getDataset3()
  full_dataset = full_dataset.batch(1)
  itrd = iter(full_dataset)
  for i in range(selectDataCount):
    next(itrd)
  x,y = next(itrd)
  testx = np.copy(x.numpy()[0, :, :])
  #print(testx[:10])
  testStart = y.numpy()[0,0,:]
  testPre = y.numpy()[0,1:,:]
  testPre = testPre.reshape((1,max_target_len - 1,63))
  testPreSave = np.copy(testPre)
  initalPos = np.copy(testStart)
  testResult = model.test_step2((x,y))
  firstFrame = np.copy(initalPos).reshape((1,63))
  with open(f1, 'wb') as pickle_file:
    #initalPos = testResult.numpy()[0,0,:]
    results = testResult.numpy()[0, :, :] / 5.0 +initalPos
    dataList = pickle.dump([np.concatenate((firstFrame, results), axis=0), testx], pickle_file)
  '''
  with open(f2, 'wb') as pickle_file:
    #initalPos = testResult.numpy()[0,0,:]
    dataList = pickle.dump([testPreSave[0, :, :] / 5.0 +initalPos, testx], pickle_file)
  '''
  print("2222")
  full_dataset = loader.getDataset3()
  itrd = iter(full_dataset)
  for i in range(selectDataCount):
    next(itrd)
  x,y = next(itrd)
  print(x)
  print(y.shape)
  testx = np.copy(x.numpy())
  #print(testx[:10])
  testStart = y.numpy()[0,:]
  testPre = y.numpy()[1:,:]

  testStart = testStart.reshape((1,1,63))
  testPre = testPre.reshape((1,max_target_len - 1,63))
  testPreSave = np.copy(testPre)
  #testX2 = testX[:,:,:]

  testResult = model.generate(testx, testStart, tf.convert_to_tensor(testPre, dtype=tf.float32) )
  print("back")

  with open(f2, 'wb') as pickle_file:
    initalPos = testResult.numpy()[0,0,:]
    dataList = pickle.dump([testResult.numpy()[0, 1:, :] / 5.0 +initalPos, testx], pickle_file)
  print(testx.shape)
  with open(f3, 'wb') as pickle_file:
    initalPos = testResult.numpy()[0,0,:]
    dataList = pickle.dump([testPreSave[0, :, :] / 5.0 +initalPos, testx], pickle_file)

print("job finished")