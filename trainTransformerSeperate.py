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
#128 8 256

max_target_len = 130
#'''
model = ABOtransformer.Transformer(
    num_hid=64,
    num_head=4,
    num_feed_forward=128,
    source_maxlen=max_target_len,
    target_maxlen=max_target_len,
    num_layers_enc=4,
    num_layers_dec=2,
    num_classes=63,
)
'''

'''


batch_size = 4
#epoch_num = 5


optimizer = keras.optimizers.Adam(0.001)
#model.compile(optimizer=optimizer, loss="mse", metrics="mse")
model.compile(optimizer=optimizer, loss=ABOtransformer.maskLabelLoss, metrics=ABOtransformer.maskLabelLoss)
#model2.compile(optimizer=optimizer, loss=ABOtransformer.maskLabelLoss)


loader = dataLoader.trainDataLoader()
#full_dataset = full_dataset.shuffle(2)

def startTrain(filePath = './Checkpoints1/', model = None, dataset = None, restart = False, ep=5):
  if restart:
    model.load_weights(filePath+"cp.ckpt")
    print("weight loaded")
  full_dataset = dataset.batch(batch_size)
  val_dataset = full_dataset.take(6) 
  train_dataset = full_dataset.skip(6)
  if not os.path.exists(filePath):
    os.makedirs(filePath)
  if not os.path.exists(filePath):
    os.makedirs(filePath)
  checkpoints = []
  fileName = "cp.ckpt"
  checkpoints.append(keras.callbacks.ModelCheckpoint(filePath+fileName, monitor='loss', verbose=0, save_best_only=True, save_weights_only=True, mode='auto', period=1))
  checkpoints.append(keras.callbacks.TensorBoard(log_dir=filePath+'logs', histogram_freq=0, write_graph=True, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None))

  #history_callback = model.fit(train_dataset, epochs = epoch_num, validation_data=val_dataset, shuffle=True, callbacks=checkpoints)
  history_callback = model.fit(train_dataset, epochs = ep, validation_data=val_dataset, shuffle=True, callbacks=checkpoints)
  #loss_history = history_callback.history["loss"]
  #numpy_loss_history = np.array(loss_history)
  #np.savetxt(filePath+"the_history.txt", numpy_loss_history, delimiter=",")
  print("saved")
  print(model.summary())
  full_dataset = full_dataset.unbatch()

if train_mode:
  model2 = ABOtransformer.Transformer_pre2pre(
    num_hid=64,
    num_head=4,
    num_feed_forward=128,
    source_maxlen=max_target_len,
    target_maxlen=max_target_len,
    num_layers_enc=4,
    num_layers_dec=3,
    num_classes=63)
  model2.compile(optimizer=optimizer, loss=ABOtransformer.maskLabelLoss)

  full_dataset = loader.getDataset3()
  startTrain('./Checkpoints4/', model, full_dataset, False, 5)
  full_dataset = loader.getDataset3()
  startTrain('./Checkpoints4/', model2, full_dataset, True, 15)
  full_dataset = loader.getDataset3()
  startTrain('./Checkpoints4/', model, full_dataset, True, 15)
  full_dataset = loader.getDataset3()
  startTrain('./Checkpoints4/', model2, full_dataset, True, 10)

else:
  #model(np.zeros((1,w,h,c)))
  model.load_weights("./Checkpoints3/cp.ckpt")


testFilePath = "./test2"
#useTest = True
f1 = testFilePath+"/testResult_a.pb"
f2 = testFilePath+"/testResult_b.pb"
f3 = testFilePath+"/testResult_t.pb"
selectDataCount = 3

print("1111")
full_dataset = loader.getDataset3()
full_dataset = full_dataset.batch(1)
itrd = iter(full_dataset)
for i in range(selectDataCount):
  next(itrd)
x,y = next(itrd)
testx = np.copy(x.numpy()[0, :, :])
#testx = testx * loader.pos_std + loader.pos_mean

#print(testx[:10])
testStart = y.numpy()[0,0,:]
testPre = y.numpy()[0,1:,:]
testPre = testPre.reshape((1,max_target_len - 1,63))
testPreSave = np.copy(testPre)
initalPos = np.copy(testStart)
#initalPos = initalPos * loader.pos_std + loader.pos_mean

testResult = model.test_step2((x,y))
firstFrame = np.copy(initalPos).reshape((1,63))
print(loader.delta_mean)
print(loader.delta_std)
print(loader.pos_mean)
print(loader.pos_std)

with open(f1, 'wb') as pickle_file:
  #initalPos = testResult.numpy()[0,0,:]
  #results = testResult.numpy()[0, :, :] * loader.delta_std  + loader.delta_mean +initalPos
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
  #dataList = pickle.dump([testResult.numpy()[0, 1:, :] * loader.delta_std + loader.delta_mean +initalPos, testx], pickle_file)
print(testx.shape)
with open(f3, 'wb') as pickle_file:
  initalPos = testResult.numpy()[0,0,:]
  dataList = pickle.dump([testPreSave[0, :, :] / 5.0 +initalPos, testx], pickle_file)
  #dataList = pickle.dump([testPreSave[0, :, :] * loader.delta_std + loader.delta_mean +initalPos, testx], pickle_file)

print("job finished")