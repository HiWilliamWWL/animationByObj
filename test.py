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

ABOtransformer.batch_size = 1
max_target_len = 85 #130


checkPointFolder = './Checkpoints5/'
testFilePath = "./test7"
useFormerTest = True
f1 = testFilePath+"/testResult_a.pb"
f2 = testFilePath+"/testResult_b.pb"
f3 = testFilePath+"/testResult_t.pb"
selectDataCount = 3  #1 3

objGlobalTrans = [.0, .0, 0.0]


objGlobalTrans = tf.reshape(tf.convert_to_tensor(objGlobalTrans), [1,1,1,3])
objGlobalTrans = tf.tile(objGlobalTrans,(1,max_target_len,12,1))
objGlobalTrans =tf.reshape(objGlobalTrans, (1, max_target_len, 36))


loader = dataLoader.trainDataLoader()
model = ABOtransformer.Transformer_newScheduleSampling(
    num_hid=128,
    num_head=8,
    num_feed_forward=128,
    source_maxlen=max_target_len,
    target_maxlen=max_target_len,
    num_layers_enc=4,
    num_layers_dec=3,
    num_classes=63
)
optimizer = keras.optimizers.Adam(0.001)
model.compile(optimizer=optimizer, loss=losses.maskLabelLoss, metrics=losses.maskLabelLoss)
model.load_weights(checkPointFolder+"cp.ckpt")
print("weight loaded")

if useFormerTest:
  print("1111")
  full_dataset = loader.getDataset3()
  full_dataset = full_dataset.batch(1)
  itrd = iter(full_dataset)
  for i in range(selectDataCount):
    next(itrd)
  x,y = next(itrd)

  x = tf.dtypes.cast(x, tf.float32)
  x += objGlobalTrans
  
  testx = np.copy(x.numpy()[0, :, :])
  testStart = y.numpy()[0,0,:]
  testPre = y.numpy()[0,1:,:]
  testPre = testPre.reshape((1,max_target_len - 1,63))
  testPreSave = np.copy(testPre)
  initalPos = np.copy(testStart)
  testResult = model.test_step2((x,y))
  firstFrame = np.copy(initalPos).reshape((1,63))
  with open(f1, 'wb') as pickle_file:
    #initalPos = testResult.numpy()[0,0,:]
    #results = testResult.numpy()[0, :, :] / 5.0 +initalPos
    #results = testResult.numpy()[0, :, :] * loader.delta_std  + loader.delta_mean +initalPos

    #results = testResult.numpy()[0, :, :] * loader.ppl_std  + loader.ppl_mean

    results = testResult.numpy()[0, :, :]
    results = results * loader.ppl_std  + loader.ppl_mean
    #results /= 5.0 
    results = results.reshape((max_target_len-1,21,3))
    print(results[:30, 0])
    print(results.shape)
    for i in range(20):
      results[:, 1+i] = results[:, 1+i] + results[:, 0]
    
    #dataList = pickle.dump([np.concatenate((firstFrame, results), axis=0), testx], pickle_file)
    dataList = pickle.dump([results, testx], pickle_file)
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

  x = tf.dtypes.cast(x, tf.float32)
  objGlobalTrans = tf.squeeze(objGlobalTrans, 0)
  x += objGlobalTrans

  #exit()
  testx = np.copy(x.numpy())
  print(testx)
  print(testx.shape)
  #exit()
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
    #dataList = pickle.dump([testResult.numpy()[0, 1:, :] / 5.0 +initalPos, testx], pickle_file)
    #dataList = pickle.dump([testResult.numpy()[0, 1:, :] * loader.delta_std + loader.delta_mean +initalPos, testx], pickle_file)
    
    #results = testResult.numpy()[0, :, :] * loader.ppl_std  + loader.ppl_mean
    results = testResult.numpy()[0, :, :]
    #results /= 5.0 
    results = results * loader.ppl_std  + loader.ppl_mean
    #results = results.reshape((max_target_len-1,21,3))
    for i in range(20):
      results[:, 1+i] = results[:, 1+i] + results[:, 0]
    dataList = pickle.dump([results, testx], pickle_file)
  with open(f3, 'wb') as pickle_file:
    initalPos = testResult.numpy()[0,0,:]
    #dataList = pickle.dump([testPreSave[0, :, :] / 5.0 +initalPos, testx], pickle_file)
    #dataList = pickle.dump([testPreSave[0, :, :] * loader.delta_std + loader.delta_mean +initalPos, testx], pickle_file)
    #results = testPreSave[0, :, :] * loader.ppl_std  + loader.ppl_mean
    
    #results = testPreSave / 5.0
    results = testPreSave * loader.ppl_std  + loader.ppl_mean
    results = results.reshape((max_target_len-1,21,3))
    for i in range(20):
      results[:, 1+i] += results[:, 0]
    #print(results[:, 0])
    dataList = pickle.dump([results, testx], pickle_file)

print("job finished")