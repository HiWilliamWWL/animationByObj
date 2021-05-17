import tensorflow as tf
import numpy as np
import glob
import pickle
from scipy.spatial.transform import Rotation as R

startEnd = {"t1":[220,360], "t2":[60, 320], "t3":[240, 350], "t4":[120, 260], "t5":[85, 230], "t6":[60, 270]}

class trainDataLoader:
    def __init__(self):
        self.skeletonData = []
        self.objectPosData = []
        self.loadFromFile()
    
    def loadFromFile(self, pathName = "./liftOnly/*.data"):
        files = glob.glob(pathName)
        for f in files:
            print(f)
            with open(f, 'rb') as pickle_file:
                dataList = pickle.load(pickle_file)[0]
                start = 0
                end = 100
                for k in startEnd:
                    if k in f:
                        start = startEnd[k][0]
                        end = startEnd[k][1]
                step = (end - start) // 50 - 1
                bodyData = []
                rigidPos = []
                for i in range(start, end, step):
                    bodyData.append(dataList[0][i])
                    objPos = dataList[2][i][:3]
                    objRot = R.from_quat(dataList[2][i][3:])
                    objRot = objRot.as_euler("xyz").tolist()
                    rigidPos.append(np.array(objPos+objRot))
                self.skeletonData.append(np.array(bodyData))
                self.objectPosData.append(np.array(rigidPos))
    
    def generator(self):
        """
        Returns tuple of (inputs,outputs) where
        inputs  = (inp1,inp2,inp2)
        outputs = (out1,out2)
        """
        dt = np.float32
        for file_count in range(len(self.skeletonData)):
            lastObjState = np.zeros((6))
            for i in range(50-1):
                skeletonPart = self.skeletonData[file_count][i].astype(dt)
                skeletonPart = skeletonPart.reshape((21,3))

                objectWholeSeq = self.objectPosData[file_count][:50].astype(dt)
                objectWholeSeq = objectWholeSeq.reshape((50,6))

                currentPos = self.objectPosData[file_count][i]
                objectCurrent = np.concatenate((currentPos, currentPos - lastObjState), axis=0).astype(dt)
                lastObjState = currentPos
                resultPos = self.skeletonData[file_count][i + 1].astype(dt)
                resultPos = resultPos.reshape((21,3))

                inputs  = (skeletonPart, objectWholeSeq, objectCurrent)
                outputs = (resultPos)
                yield inputs,outputs

    def getDataset(self):
        # Create dataset from generator
        types = ( (tf.float32,tf.float32,tf.float32),
                (tf.float32) )
        shapes = (([21,3],[50,6],[12]),
                ([21,3]))
        data = tf.data.Dataset.from_generator(self.generator,output_types=types, output_shapes=shapes)
        return data

    def getDataset2(self):
        # Create dataset from generator
        skeletonPart = self.skeletonData
        dt = np.float32
        objPart = self.objectPosData
        for file_count in range(len(self.skeletonData)):
            skeletonPart[file_count] = np.array(skeletonPart[file_count][:60]).astype(dt)
            objPart[file_count] = np.array(objPart[file_count][:60]).astype(dt)
        dataset = tf.data.Dataset.from_tensor_slices((self.objectPosData, self.skeletonData))

        return dataset

#t = trainDataLoader()
#t.getDataset2()