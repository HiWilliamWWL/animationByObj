import os
import random
from glob import glob
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import dataLoader_angles
import numpy as np
import pickle
import slidingBaseFc as sbf
import losses

sbf.batch_size = 1
max_target_len = 85 #130


selectDataCount = 4  #6

checkPointFolder = './Checkpoints/Checkpoints_sbf1/'
testFilePath = "./Tests/test_sbf1"


fN = testFilePath+"/testResult_c.pb"
ft = testFilePath+"/testResult_t.pb"
fP = testFilePath+"/testResult_p.pb"
fG = testFilePath+"/testResult_g.pb"

add_move = False
emphasize = False


loader = dataLoader_angles.trainDataLoader(None, ["./Data/liftOnly/test/*.data"])

model = sbf.SlidingBaseFc(num_hid=128, target_maxlen=max_target_len, num_classes=114)
optimizer = keras.optimizers.Adam(0.001)
model.compile(optimizer=optimizer, loss={"final_result": losses.maskLabelLoss_angles, "intial_human": losses.initialPoseLoss_angles},
  metrics={"final_result": losses.maskMSE_angles, "intial_human": losses.initialPoseLoss_angles})
model.load_weights(checkPointFolder+"cp.ckpt")
print("weight loaded")

full_dataset = loader.getDataset3()
itrd = iter(full_dataset)
for i in range(selectDataCount):
    next(itrd)
x,y = next(itrd)
x = tf.dtypes.cast(x, tf.float32)


testx = np.copy(x.numpy())

testx = testx.reshape((85,12,3))
speed = np.array([0.04,-0.01,0.0])
factor = np.array([1.4,1.,1.])
startFrame = 25
if add_move:
    for i in range(startFrame, max_target_len):
        acc = i - startFrame + 1
        testx[i] += acc*speed
if emphasize:
    for i in range(0, max_target_len):
        testx[i] *= factor

testx = testx.reshape((1, 85,36))
testStart = y.numpy()[0,:]
testPre = y.numpy()[:,:]

testStart = testStart.reshape((1,1,114))  #63 54 54+63-3=114
testPre = testPre.reshape((1,max_target_len,114))  #63 54 54+63-3=114
testPreSave = np.copy(testPre)
testPre = tf.convert_to_tensor(testPre, dtype=tf.float32)

testResult = model.generate(testx)

testx = testx.reshape((85,36))
markers1 = testx[0].reshape((12,3))
markers1 = markers1* loader.obj_std_saved + loader.obj_mean_saved
print(markers1)
print(np.mean(markers1, axis = 0))
exit()
with open(fP, 'wb') as pickle_file:
    results = testResult.numpy()[0, :, :]
    results = np.concatenate((results[:, :3], results[:, -60:]), axis=1)
    results[:, 3:] = results[:, 3:] * loader.ppl_std_saved_p  + loader.ppl_mean_saved_p
    results = results.reshape((max_target_len,21,3))
    for i in range(20):
        results[:, 1+i] = results[:, 1+i] + results[:, 0]
    results = results.reshape((max_target_len,63))
    dataList = pickle.dump([results, testx * loader.obj_std_saved + loader.obj_mean_saved], pickle_file)

with open(fG, 'wb') as pickle_file:
    #results = testPreSave * loader.ppl_std_saved  + loader.ppl_mean_saved
    results = testPreSave
    results = results[0, :, :]
    results = np.concatenate((results[:, :3], results[:, -60:]), axis=1)
    results[:, 3:] = results[:, 3:] * loader.ppl_std_saved_p  + loader.ppl_mean_saved_p
    results = results.reshape((max_target_len,21,3))
    for i in range(20):
        results[:, 1+i] = results[:, 1+i] + results[:, 0]
    results = results.reshape((max_target_len,63))
    dataList = pickle.dump([results, testx * loader.obj_std_saved + loader.obj_mean_saved], pickle_file)


print("error report__________time ave joint error")
pre_save = testResult.numpy()[0,:,-60:] * loader.ppl_std_p  + loader.ppl_mean_p
gt_save = testPreSave[0,:,-60:] * loader.ppl_std_p  + loader.ppl_mean_p
pre_save = pre_save.reshape((-1, 20,3))
gt_save = gt_save.reshape((-1, 20,3))
error = pre_save - gt_save

error = np.sqrt(np.sum(error**2, axis=2))
print(error)
print(error.shape)
error = np.mean(error, axis=1)
print(error)

print(np.sum(error))
print(np.sum(error) / 85.0)
#'''
print("job finished")