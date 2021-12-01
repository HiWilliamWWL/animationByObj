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

checkPointFolder = './Checkpoints/Checkpoints_wgn1/'
testFilePath = "./Tests/test_wgn1"

fN = testFilePath+"/testResult_c.pb"
ft = testFilePath+"/testResult_t.pb"
fP = testFilePath+"/testResult_p.pb"

#loader = dataLoader.trainDataLoader(1, pathNames = ["./Data/walkData/oneFIle/*.data"])

model = wgn.walkGenerateNet(num_hid=128, target_maxlen=max_target_len, num_classes=81, num_experts = 3, num_input_dimension = 64)
model.load_weights(checkPointFolder+"cp.ckpt")
print("weight loaded")
bodyWhole_T = walkAnnotationData.saved_Tpose


full_dataset = walkAnnotationData.getData()
itrd1 = iter(full_dataset)
itrd2 = iter(full_dataset)

#testx = np.copy(x_tf.numpy())
#testx = testx.reshape((85,36))

testx = np.zeros((85,36))
#y = y_tf.numpy()[:,:].reshape(max_target_len, dataLoader.humanDimensionWalk)


with open(fN, 'wb') as pickle_file:
    
    bodyWhole_Result = []
    while True:
    #for i in range(3):
        try:
            nextData = next(itrd1)
        except StopIteration:
            break
        x_tf, y_tf = nextData
        phase = x_tf.numpy()[0]
        x_tf = tf.convert_to_tensor(x_tf, dtype=tf.float32)
        
        x_tf = tf.expand_dims(x_tf, axis=0)
        
        testResult = model.generate(x_tf)
        testResult = testResult.numpy()[0, :]    #81 
        
        bodyCenter = testResult[:3]
        allRotsSave = testResult[3:-24].reshape((9,6))  #9
        JI = skeletonHandle.JointsInfo(bodyWhole_T)
        JI.apply_global_trans(bodyCenter)
        JI.forward_kinematics_Legs_vecs(allRotsSave)
        bodyWhole_Result.append(JI.get_all_poses().reshape(63))
    dataList = pickle.dump([np.array(bodyWhole_Result), testx ], pickle_file)



print("process GT")
with open(ft, 'wb') as pickle_file:
    
    bodyWhole_Result = []
    while True:
        try:
            x_tf,y_tf = next(itrd2)
        except StopIteration:
            break
        phase = x_tf.numpy()[0]
        results = y_tf.numpy() 
        
        bodyCenter = results[:3]
        allRotsSave = results[3:-24].reshape((9,6)) #3:-24  9
        #print(allRotsSave[0, :])
        #allRotsSave[0, :] = [.0, 1.0, .0, 1.0, .0, .0]
        #exit()
        JI = skeletonHandle.JointsInfo(bodyWhole_T)
        JI.apply_global_trans(bodyCenter)
        JI.forward_kinematics_Legs_vecs(allRotsSave)
        bodyWhole_Result.append(JI.get_all_poses().reshape(63))
    dataList = pickle.dump([np.array(bodyWhole_Result), testx], pickle_file)


itrd1 = iter(full_dataset)
with open(fP, 'wb') as pickle_file:
    
    bodyWhole_Result = []
    while True:
        bodyWhole_T2 = bodyWhole_T[:, :]
        try:
            nextData = next(itrd1)
        except StopIteration:
            break
        x_tf, y_tf = nextData
        phase = x_tf.numpy()[0]
        x_tf = tf.convert_to_tensor(x_tf, dtype=tf.float32)
        
        x_tf = tf.expand_dims(x_tf, axis=0)
        
        testResult = model.generate(x_tf)
        testResult = testResult.numpy()[0, :]    #81 
        
        
        bodyCenter = testResult[:3]
        allRotsSave = testResult[3:-24].reshape((9,6))  #9
        allPosesSave = testResult[-24:].reshape((8, 3))
        allPosesSave = np.concatenate((bodyCenter.reshape((1, 3)), allPosesSave), axis=0)
        #JI = skeletonHandle.JointsInfo(bodyWhole_T)
        #JI.apply_global_trans(bodyCenter)
        #JI.forward_kinematics_Legs_vecs(allRotsSave)
        rotsMap = {0:0, 13:1, 14:2, 15:3, 16:4, 17:5, 18:6, 19:7, 20:8}
        for i in range(21):
            if i in rotsMap:
                if i == 0:
                    bodyWhole_T2[i, :] = [.0, .0, .0]
                else:
                    
                    bodyWhole_T2[i, :] = allPosesSave[rotsMap[i], :]
            else:
                bodyWhole_T2[i, :] = [.0, .0, .0]
        aCopy = np.copy(bodyWhole_T2)
        bodyWhole_Result.append(aCopy)
    print(np.array(bodyWhole_Result))
    dataList = pickle.dump([np.array(bodyWhole_Result), testx ], pickle_file)

print("job finished")