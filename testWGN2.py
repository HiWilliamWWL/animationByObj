import os
import random
from glob import glob
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
#import dataLoader_walk_seperate as dataLoader
import walkAnnotationData
import numpy as np
import pickle
import walkGenerateNet as wgn
import losses
import skeletonHandle
from scipy.spatial.transform import Rotation as Rot

wgn.batch_size = 1
max_target_len = 85 #130

checkPointFolder = './Checkpoints/Checkpoints_wgn2/'
testFilePath = "./Tests/test_wgn2"

fN = testFilePath+"/testResult_c.pb"
ft = testFilePath+"/testResult_t.pb"
fP = testFilePath+"/testResult_p.pb"

#loader = dataLoader.trainDataLoader(1, pathNames = ["./Data/walkData/oneFIle/*.data"])

model = wgn.walkGenerateNet(num_hid=256, target_maxlen=max_target_len, num_classes=190, num_experts = 3, num_input_dimension = 376)
model.load_weights(checkPointFolder+"cp.ckpt")
print("weight loaded")

full_dataset = walkAnnotationData.getTestSample()
bodyWhole_T = walkAnnotationData.saved_Tpose


itrd1 = iter(full_dataset)
itrd2 = iter(full_dataset)
itrd3 = iter(full_dataset)


with open(fN, 'wb') as pickle_file:
    
    bodyWhole_Result = []
    
    nextData = next(itrd1)
    o_tf, x_tf, y_tf = nextData
    
    saved_obj = o_tf.numpy()
    
    x_tf = tf.expand_dims(x_tf, axis=0)
    
    o_tf = tf.expand_dims(o_tf, axis=0)
    
    phase = x_tf.numpy()[0]
    
    
    testResult = model.generate([o_tf, x_tf])
    testResult = testResult.numpy()[0, :]    #84, 190
    
    for i in range(max_target_len - 1):
        bodyCenter = testResult[i, 1:4]
        allRotsSave = testResult[i, 4:-60].reshape((21,6))  #9
        JI = skeletonHandle.JointsInfo(bodyWhole_T)
        JI.apply_global_trans(bodyCenter)
        JI.forward_kinematics_21Joints_vecs(allRotsSave)
        bodyWhole_Result.append(JI.get_all_poses().reshape(63))
    dataList = pickle.dump([np.array(bodyWhole_Result), saved_obj ], pickle_file)


print("process GT")
with open(ft, 'wb') as pickle_file:
    bodyWhole_Result = []  
    nextData = next(itrd2)
    o_tf, x_tf, y_tf = nextData
    
    testResult = y_tf.numpy() 
    saved_obj = o_tf.numpy()
    
    x_tf = tf.expand_dims(x_tf, axis=0)
    
    o_tf = tf.expand_dims(o_tf, axis=0)
    
    phase = x_tf.numpy()[0]
    
    for i in range(max_target_len - 1):
        bodyCenter = testResult[i, 1:4]
        allRotsSave = testResult[i, 4:-60].reshape((21,6))  #9
        JI = skeletonHandle.JointsInfo(bodyWhole_T)
        JI.apply_global_trans(bodyCenter)
        JI.forward_kinematics_21Joints_vecs(allRotsSave)
        bodyWhole_Result.append(JI.get_all_poses().reshape(63))
    dataList = pickle.dump([np.array(bodyWhole_Result), saved_obj ], pickle_file)


with open(fP, 'wb') as pickle_file:
    
    
    nextData = next(itrd3)
    o_tf, x_tf, y_tf = nextData
    
    saved_obj = o_tf.numpy()
    
    x_tf = tf.expand_dims(x_tf, axis=0)
    
    o_tf = tf.expand_dims(o_tf, axis=0)
    
    phase = x_tf.numpy()[0]
    
    
    testResult = model.generate([o_tf, x_tf])
    testResult = testResult.numpy()[0, :]    #84, 190
    rootPos = testResult[:, 1:4].reshape((-1, 1, 3))
    jointPos = testResult[:, -60:].reshape((-1, 20, 3))
    bodyWhole_Result = jointPos + rootPos
    bodyWhole_Result = np.concatenate((rootPos, bodyWhole_Result), axis=-2)
        
        
    dataList = pickle.dump([np.array(bodyWhole_Result), saved_obj ], pickle_file)

print("job finished")