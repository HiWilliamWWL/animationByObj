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
import skeletonHandle
from scipy.spatial.transform import Rotation as Rot

import walkAnnotationData
import walkGenerateNet as wgn

max_target_len = 85 #130

checkPointFolder_wgn = './Checkpoints/Checkpoints_wgn1/'
wgn.batch_size = 1
model_wgn = wgn.walkGenerateNet(num_hid=128, target_maxlen=max_target_len, num_classes=81, num_experts = 3, num_input_dimension = 64)
model_wgn.load_weights(checkPointFolder_wgn+"cp.ckpt")

sbf.batch_size = 1



selectDataCount = 6  #6  4

checkPointFolder_sbf = './Checkpoints/Checkpoints_sbf3/'
testFilePath = "./Tests/test_combine1"


fN = testFilePath+"/testResult_c.pb"
ft = testFilePath+"/testResult_t.pb"
fP = testFilePath+"/testResult_p.pb"
fG = testFilePath+"/testResult_g.pb"

add_move = True
emphasize = False


loader = dataLoader_angles.trainDataLoader(None, ["./Data/liftOnly/test/*.data"])
#loader = dataLoader_angles.trainDataLoader(None, ["./Data/liftOnly/test/wanted/*.data"])

model_sbf = sbf.SlidingBaseFc(num_hid=128, target_maxlen=max_target_len, num_classes=dataLoader_angles.humanDimension)
optimizer = keras.optimizers.Adam(0.001)
model_sbf.compile(optimizer=optimizer, loss={"final_result": losses.maskLabelLoss_angles, "intial_human": losses.initialPoseLoss_angles},
  metrics={"final_result": losses.maskMSE_angles, "intial_human": losses.initialPoseLoss_angles})
model_sbf.load_weights(checkPointFolder_sbf+"cp.ckpt")
print("weight loaded")

full_dataset = loader.getDataset3()
itrd = iter(full_dataset)
for i in range(selectDataCount):
    next(itrd)
x,y = next(itrd)
x = tf.dtypes.cast(x, tf.float32)


testx = np.copy(x.numpy())

testx = testx.reshape((85,12,3))
speed = np.array([0.037,-0.01,0.0])
factor = np.array([1.4,1.,1.])
startFrame = 35
if add_move:
    for i in range(startFrame, max_target_len):
        acc = i - startFrame + 1
        testx[i] += acc*speed
if emphasize:
    for i in range(0, max_target_len):
        testx[i] *= factor

testx = testx.reshape((1, 85, 36))
testStart = y.numpy()[0,:]
testPre = y.numpy()[:,:]

testStart = testStart.reshape((1, 1, loader.humanDimension))  #63 54 54+63-3=114
testPre = testPre.reshape((1, max_target_len, loader.humanDimension))  #63 54 54+63-3=114
testPreSave = np.copy(testPre)
testPre = tf.convert_to_tensor(testPre, dtype=tf.float32)

testResult = model_sbf.generate(testx)

testx = testx.reshape((85,36))
#print(np.mean( testx.reshape((85, 12, 3)), axis=1))
markers1 = testx[0].reshape((12,3))
markers1 = markers1* loader.obj_std_saved + loader.obj_mean_saved

legRelatedPoints = [0, 13, 14, 15, 19, 16, 17, 18, 20]
legRelatedPoints_noRoot = [12, 13, 14, 18, 15, 16, 17, 19]

with open(fN, 'wb') as pickle_file:
    results = testResult.numpy()[0, :, :]
    bodyWhole_T = loader.saved_Tpose
    bodyWhole_Result = []
    currentObjCenter = np.mean(testx.reshape((85,12,3)) * loader.obj_std_saved + loader.obj_mean_saved, axis=1)
    print(currentObjCenter.shape)
    phases = np.concatenate((np.linspace(0.2, 1.0, 6), np.linspace(0.9, -1.0, 8), np.linspace(-0.85, 0.0, 6)))
    #phases = np.concatenate((np.linspace(0.3, -1.0, 7), np.linspace(-0.9, 1.0, 10), np.linspace(0.9, 0.4, 3)))
    phaseCount = 0
    for i in range(max_target_len - 1):
        #check hand object distance
        #if > some threshold: generating walk motions
        
        allRotsSave_currentOutput = None
        allPosesSave_currentOutput = None
        bodyCenter_currentOutput = None
        
        bodyCenter = results[i, :3]      
        allRotsSave = results[i, 3:-60].reshape((21,6))
        allPosesSave = results[i, -60:].reshape((20,3))
        
        if i >350:
            allRotsSave_last = results[i-1, 3:-60].reshape((21,6))
            allPosesSave_last = results[i-1, -60:].reshape((20,3))
            bodyCenter_last = results[i-1, :3]
            bodyRots = allRotsSave_last[legRelatedPoints].flatten()
            bodyPoses = allPosesSave_last[legRelatedPoints_noRoot].flatten()
            currentPhase = phases[phaseCount % 20]  #!!!!!!!!  14
            skeletonDataX =  np.concatenate(([currentPhase],  bodyCenter_last, 
                                             bodyRots * np.sin(currentPhase), bodyPoses * np.sin(currentPhase)
                                             ,bodyRots * np.cos(currentPhase), bodyPoses * np.cos(currentPhase)))
            phaseCount += 1
            
            skeletonDataX = tf.convert_to_tensor(skeletonDataX.reshape((1, -1)), dtype=tf.float32)
            testResult = model_wgn.generate(skeletonDataX)
            testResult = testResult.numpy()[0, :]    #81
            
            #results[i, :3] = testResult[:3]  #bodyCenter
            bodyCenter_currentOutput = testResult[:3]
            allRotsSave_currentOutput = testResult[3:-24].reshape((9,6))  #9
            allPosesSave_currentOutput = testResult[-24:].reshape((8,3))  
            
            allRotsSave = results[i, 3:-60].reshape((21,6))
            allPosesSave = results[i, -60:].reshape((20,3))
            for count_j, jj in enumerate(legRelatedPoints):
                allRotsSave[jj, :] = allRotsSave_currentOutput[count_j, :]
            for count_j, jj in enumerate(legRelatedPoints_noRoot):
                allPosesSave[jj, :] = allPosesSave_currentOutput[count_j, :]
            results[i, 3:-60] = allRotsSave.flatten()
            results[i, -60:] = allPosesSave.flatten()
        
        
        
        JI = skeletonHandle.JointsInfo(bodyWhole_T)
        
        #JI.forward_kinematics_21Joints_vecs(allRotsSave)
        
        
        if i > 350:
            #pass
            #JI2 = skeletonHandle.JointsInfo(bodyWhole_T)
            JI.fk_rootY_test(-90)
            #JI.apply_global_trans(bodyCenter_currentOutput)
            JI.apply_global_trans(bodyCenter)
            JI.forward_kinematics_UpperBody_vecs(results[i, 3:-60].reshape((21,6))[ 1:13, :])
            JI.forward_kinematics_Legs_vecs(allRotsSave_currentOutput[1:, :])
            #allPoses[-8:, :] = allPosesSave_currentOutput[:, :] + bodyCenter
        else:
            JI.apply_global_trans(bodyCenter)
            JI.forward_kinematics_21Joints_vecs(allRotsSave)
        
        allPoses = JI.get_all_poses()
        bodyWhole_Result.append(allPoses.reshape(63))
    dataList = pickle.dump([np.array(bodyWhole_Result), testx * loader.obj_std_saved + loader.obj_mean_saved], pickle_file)
print("job finished")