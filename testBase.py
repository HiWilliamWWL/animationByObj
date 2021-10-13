import os
import random
from glob import glob
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import dataLoader
import numpy as np
import pickle
import baseFcRnn
import losses
#import thisIK

baseFcRnn.batch_size = 1
max_target_len = 85 #130


checkPointFolder = './CheckpointsB1/' #4
#checkPointFolder = './Checkpoints1/'
testFilePath = "./testB1" #7 1 2
useFormerTest = True

fN = testFilePath+"/testResult_c.pb"
ft = testFilePath+"/testResult_t.pb"

selectDataCount = 6  #1 3 2 6,7

#right, up, in
objGlobalTrans = [0.0, 0.0, 0.0]

objGlobalTrans = tf.reshape(tf.convert_to_tensor(objGlobalTrans), [1,1,1,3])
objGlobalTrans = tf.tile(objGlobalTrans,(1,max_target_len,12,1))
objGlobalTrans =tf.reshape(objGlobalTrans, (1, max_target_len, 36))

loader = dataLoader.trainDataLoader()
model = baseFcRnn.baseFC_RNN(
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
model.compile(optimizer=optimizer, loss={"final_result": losses.maskLabelLoss, "intial_human": losses.initialPoseLoss},
    metrics={"final_result": losses.maskMSE, "intial_human": losses.initialPoseLoss})
model.load_weights(checkPointFolder+"cp.ckpt")
print("weight loaded")

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

with open(fN, 'wb') as pickle_file:
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
with open(ft, 'wb') as pickle_file:
    results = testPreSave * loader.ppl_std  + loader.ppl_mean
    results = results.reshape((max_target_len-1,21,3))
    for i in range(20):
      results[:, 1+i] += results[:, 0]
    #print(results[:, 0])
    dataList = pickle.dump([results, testx], pickle_file)

print("error report")
pre_save = testResult.numpy()[0,:,:] * loader.ppl_std  + loader.ppl_mean
gt_save = testPreSave[0,:,:] * loader.ppl_std  + loader.ppl_mean
pre_save = np.sqrt(np.sum(pre_save**2, axis=1))
gt_save = np.sqrt(np.sum(gt_save**2, axis=1))
error_report = np.abs(pre_save - gt_save)
print(error_report)
print(error_report.shape)
print(np.sum(error_report))

print("job finished")