import os
import random
from glob import glob
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import dataLoader_angles
import numpy as np
import pickle
import ABOtransformer_angles as ABOtransformer
import losses
import skeletonHandle
from scipy.spatial.transform import Rotation as Rot
#import thisIK

ABOtransformer.batch_size = 1
max_target_len = 85 #85 #35


#checkPointFolder = './Checkpoints/Checkpoints_angles2/' #4
checkPointFolder = './Checkpoints/Checkpoints_angles2/'
#checkPointFolder = './Checkpoints_withoutDiscrimin/'

#checkPointFolder = './Checkpoints1/'
testFilePath = "./Tests/test_angles2" #7 1 2

fN = testFilePath+"/testResult_c.pb"
ft = testFilePath+"/testResult_t.pb"
fP = testFilePath+"/testResult_p.pb"
fG = testFilePath+"/testResult_g.pb"

selectDataCount = 0  #1 3 2 6,7

#right, up, in
objGlobalTrans = [0.0, 0.0, 0.0]

objGlobalTrans = tf.reshape(tf.convert_to_tensor(objGlobalTrans), [1,1,1,3])
objGlobalTrans = tf.tile(objGlobalTrans,(1,max_target_len,12,1))
objGlobalTrans =tf.reshape(objGlobalTrans, (1, max_target_len, 36))

loader = dataLoader_angles.trainDataLoader(None, ["./Data/liftOnly/test/wanted/*.data"])
#loader = dataLoader_angles.trainDataLoader(3, ["./Data/liftOnly/test/*.data"])
#loader = dataLoader_angles.trainDataLoader(15, "./Data/liftOnly/*.data")

add_move = True
emphasize = False

model = ABOtransformer.Transformer_VAEinitalPose(
    num_hid=128,
    num_head=8,
    num_feed_forward=128,
    source_maxlen=max_target_len,
    target_maxlen=max_target_len,
    num_layers_enc=4,
    num_layers_dec=3,
    num_classes=114   #63 54 54+63-3=114
)
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
objGlobalTrans = tf.squeeze(objGlobalTrans, 0)
x += objGlobalTrans

#exit()
testx = np.copy(x.numpy())
#testx /= loader.obj_std_saved
testx = testx.reshape((85,12,3))
speed = np.array([0.03,-0.01,0.0])
factor = np.array([1.4,1.,1.])
startFrame = 30
if add_move:
    for i in range(startFrame, max_target_len):
        acc = i - startFrame + 1
        testx[i] += acc*speed
if emphasize:
    for i in range(0, max_target_len):
        testx[i] *= factor
#print(testx[:10])
testStart = y.numpy()[0,:]
testPre = y.numpy()[1:,:]

testStart = testStart.reshape((1,1,114))  #63 54 54+63-3=114
testPre = testPre.reshape((1,max_target_len - 1,114))  #63 54 54+63-3=114
testPreSave = np.copy(testPre)
testPre = tf.convert_to_tensor(testPre, dtype=tf.float32)
#testX2 = testX[:,:,:]

testResult = model.generate(testx,  tf.convert_to_tensor(testStart, dtype=tf.float32),  testPre, loader)
print("finished running network")

with open(fN, 'wb') as pickle_file:
    #initalPos = testResult.numpy()[0,0,:]
    #dataList = pickle.dump([testResult.numpy()[0, 1:, :] / 5.0 +initalPos, testx], pickle_file)
    #dataList = pickle.dump([testResult.numpy()[0, 1:, :] * loader.delta_std + loader.delta_mean +initalPos, testx], pickle_file)

    #results = testResult.numpy()[0, :, :] * loader.ppl_std  + loader.ppl_mean
    results = testResult.numpy()[0, :, :]
    #results /= 5.0 
    #results = results * loader.ppl_std_saved  + loader.ppl_mean_saved
    #results = results.reshape((max_target_len-1,21,3))
    bodyWhole_T = loader.saved_Tpose
    bodyWhole_Result = []
    for i in range(max_target_len - 1):
        #results[:, 1+i] = results[:, 1+i] + results[:, 0]
        bodyCenter = results[i, :3]
        if i == 0:
            bodyCenter *= 0.0
        allRotsSave = results[i, 3:54].reshape((17,3)) * 3.1415926536
        JI = skeletonHandle.JointsInfo(bodyWhole_T)
        JI.apply_global_trans(bodyCenter)
        allRots = []
        for k in range(17):
            allRots.append(Rot.from_euler("xyz", allRotsSave[k]))
            #print(allRots[-1].as_euler("xyz", degrees=True))
        JI.forward_kinematics_parentsOnly_17Joints(allRots)
        bodyWhole_Result.append(JI.get_all_poses().reshape(63))
        #JI.apply_global_trans(-1.0 * bodyCenter)
        #print()
    dataList = pickle.dump([np.array(bodyWhole_Result), testx * loader.obj_std_saved + loader.obj_mean_saved], pickle_file)

with open(fP, 'wb') as pickle_file:
    initalPos = testResult.numpy()[0,0,:]
    #print(testResult.numpy()[0,0,:])
    #print(testResult.numpy()[0,1,:])
    #dataList = pickle.dump([testResult.numpy()[0, 1:, :] / 5.0 +initalPos, testx], pickle_file)
    #dataList = pickle.dump([testResult.numpy()[0, 1:, :] * loader.delta_std + loader.delta_mean +initalPos, testx], pickle_file)

    #results = testResult.numpy()[0, :, :] * loader.ppl_std  + loader.ppl_mean
    results = testResult.numpy()[0, :, :]  
    # results 84,114
    #results /= 5.0
    results = np.concatenate((results[:, :3], results[:, -60:]), axis=1)
    results[:, 3:] = results[:, 3:] * loader.ppl_std_saved_p  + loader.ppl_mean_saved_p
    results = results.reshape((max_target_len-1,21,3))
    #print(results[:, 0])
    for i in range(20):
        results[:, 1+i] = results[:, 1+i] + results[:, 0]
    results = results.reshape((max_target_len-1,63))
    dataList = pickle.dump([results, testx * loader.obj_std_saved + loader.obj_mean_saved], pickle_file)

with open(ft, 'wb') as pickle_file:
    #results = testPreSave * loader.ppl_std_saved  + loader.ppl_mean_saved
    results = testPreSave
    results = results[0, :, :]
    #results = results.reshape((max_target_len-1,21,3))
    bodyWhole_T = loader.saved_Tpose
    bodyWhole_Result = []
    for i in range(max_target_len - 1):
        bodyCenter = results[i, :3]
        allRotsSave = results[i, 3:-60].reshape((17,3)) * 3.1415926536
        JI = skeletonHandle.JointsInfo(bodyWhole_T)
        JI.apply_global_trans(bodyCenter)
        allRots = []
        for k in range(17):
            allRots.append(Rot.from_euler("xyz", allRotsSave[k]))
        JI.forward_kinematics_parentsOnly_17Joints(allRots)
        bodyWhole_Result.append(JI.get_all_poses().reshape(63))
        #JI.apply_global_trans(-1.0 * bodyCenter)
    dataList = pickle.dump([np.array(bodyWhole_Result), testx * loader.obj_std_saved + loader.obj_mean_saved], pickle_file)

with open(fG, 'wb') as pickle_file:
    #results = testPreSave * loader.ppl_std_saved  + loader.ppl_mean_saved
    results = testPreSave
    results = results[0, :, :]
    print(results.shape)
    results = np.concatenate((results[:, :3], results[:, -60:]), axis=1)
    results[:, 3:] = results[:, 3:] * loader.ppl_std_saved_p  + loader.ppl_mean_saved_p
    results = results.reshape((max_target_len-1,21,3))
    for i in range(20):
        results[:, 1+i] = results[:, 1+i] + results[:, 0]
    results = results.reshape((max_target_len-1,63))
    dataList = pickle.dump([results, testx * loader.obj_std_saved + loader.obj_mean_saved], pickle_file)    



#'''
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