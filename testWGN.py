import os
import random
from glob import glob
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import dataLoader_walk as dataLoader
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

loader = dataLoader.trainDataLoader(None, pathNames = ["./Data/walkData/*.data"])

model = wgn.walkGenerateNet(num_hid=128, target_maxlen=max_target_len, num_classes=dataLoader.humanDimensionWalkOutput, num_experts = 3, num_input_dimension = dataLoader.humanDimensionWalk)
model.load_weights(checkPointFolder+"cp.ckpt")
print("weight loaded")
bodyWhole_T = loader.saved_Tpose

full_dataset = loader.getDataset3()
itrd = iter(full_dataset)
x_tf,y_tf = next(itrd)

#testx = np.copy(x_tf.numpy())
#testx = testx.reshape((85,36))

testx = np.zeros((85,36))
y = y_tf.numpy()[:,:].reshape(max_target_len, dataLoader.humanDimensionWalk)


with open(fN, 'wb') as pickle_file:
    results = y[ :, 3:]

    resultsSaved = []
    
    bodyWhole_Result = []
    for i in range(max_target_len - 1):
    #for i in range(3):
        if i == 0:
            inputX = y[0, :]  #84
        else:
            phase = y[i, 0]
            direction1 = y[i, 1]
            direction2 = y[i, 2]
            lastOutput = resultsSaved[-1]   #81
            inputX = [phase, direction1, direction2] + lastOutput.tolist()
        inputX = np.array(inputX).reshape((1, 84))
        inputX = tf.convert_to_tensor(inputX, dtype=tf.float32)
        testResult = model.generate(inputX)
        testResult = testResult.numpy()[0, :]  #81

        resultsSaved.append(testResult)


        bodyCenter = testResult[:3]
        allRotsSave = testResult[3:-24].reshape((9,6))
        JI = skeletonHandle.JointsInfo(bodyWhole_T)
        JI.apply_global_trans(bodyCenter)
        JI.forward_kinematics_Legs_vecs(allRotsSave)
        bodyWhole_Result.append(JI.get_all_poses().reshape(63))
    dataList = pickle.dump([np.array(bodyWhole_Result), testx * loader.obj_std_saved + loader.obj_mean_saved], pickle_file)


with open(ft, 'wb') as pickle_file:
    results = y[ :, 3:]
    
    bodyWhole_Result = []
    for i in range(max_target_len - 1):
        bodyCenter = results[i, :3]
        allRotsSave = results[i, 3:-24].reshape((9,6))
        JI = skeletonHandle.JointsInfo(bodyWhole_T)
        JI.apply_global_trans(bodyCenter)
        JI.forward_kinematics_Legs_vecs(allRotsSave)
        bodyWhole_Result.append(JI.get_all_poses().reshape(63))
    dataList = pickle.dump([np.array(bodyWhole_Result), testx * loader.obj_std_saved + loader.obj_mean_saved], pickle_file)

print("job finished")