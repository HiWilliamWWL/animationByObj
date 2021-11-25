import os
import random
from glob import glob
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import dataLoader_walk_seperate as dataLoader
import numpy as np
import pickle
import walkGenerateNet as wgn
import losses
import skeletonHandle
from scipy.spatial.transform import Rotation as Rot

wgn.batch_size = 1
max_target_len = 85 #130

checkPointFolder = './Checkpoints/Checkpoints_wgn1/'
testFilePath = "./Tests/test_wgn1"

fN = testFilePath+"/testResult_c.pb"
ft = testFilePath+"/testResult_t.pb"

loader = dataLoader.trainDataLoader(None, pathNames = ["./Data/walkData/oneFIle/*.data"])

model = wgn.walkGenerateNet(num_hid=128, target_maxlen=max_target_len, num_classes=dataLoader.humanDimensionWalkOutput, num_experts = 3, num_input_dimension = dataLoader.humanDimensionWalk)
model.load_weights(checkPointFolder+"cp.ckpt")
print("weight loaded")
bodyWhole_T = loader.saved_Tpose

full_dataset = loader.getDataset3()
itrd1 = iter(full_dataset)
itrd2 = iter(full_dataset)

#testx = np.copy(x_tf.numpy())
#testx = testx.reshape((85,36))

testx = np.zeros((85,36))
#y = y_tf.numpy()[:,:].reshape(max_target_len, dataLoader.humanDimensionWalk)


with open(fN, 'wb') as pickle_file:
    
    bodyWhole_Result = []
    for i in range(max_target_len - 1):
    #for i in range(3):
        x_tf,y_tf = next(itrd1)
        phase = x_tf.numpy()[0]
        x_tf = tf.expand_dims(x_tf, axis=0)
        testResult = model.generate(x_tf)
        
        testResult = testResult.numpy()[0, :] / np.sin(phase)     #81 
        
        bodyCenter = testResult[:3]
        allRotsSave = testResult[3:-24].reshape((9,6))
        JI = skeletonHandle.JointsInfo(bodyWhole_T)
        JI.apply_global_trans(bodyCenter)
        JI.forward_kinematics_Legs_vecs(allRotsSave)
        bodyWhole_Result.append(JI.get_all_poses().reshape(63))
    dataList = pickle.dump([np.array(bodyWhole_Result), testx * loader.obj_std_saved + loader.obj_mean_saved], pickle_file)


with open(ft, 'wb') as pickle_file:
    
    bodyWhole_Result = []
    for i in range(max_target_len - 1):
        x_tf,y_tf = next(itrd1)
        phase = x_tf.numpy()[0]
        results = y_tf.numpy()  / np.sin(phase)
        
        bodyCenter = results[:3]
        allRotsSave = results[3:-24].reshape((9,6))
        JI = skeletonHandle.JointsInfo(bodyWhole_T)
        JI.apply_global_trans(bodyCenter)
        JI.forward_kinematics_Legs_vecs(allRotsSave)
        bodyWhole_Result.append(JI.get_all_poses().reshape(63))
    dataList = pickle.dump([np.array(bodyWhole_Result), testx * loader.obj_std_saved + loader.obj_mean_saved], pickle_file)

print("job finished")