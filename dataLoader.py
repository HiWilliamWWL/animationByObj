import tensorflow as tf
import numpy as np
import glob
import pickle
from scipy.spatial.transform import Rotation as R

#startEnd = {"t1":[220,360], "t2":[60, 320], "t3":[240, 350], "t4":[120, 260], "t5":[85, 230], "t6":[60, 270]}
startEnd = {"t1":[180,360], "t2":[30, 320], "t3":[180, 350], "t4":[60, 260], "t5":[30, 230], "t6":[20, 270]}

class trainDataLoader:
    def __init__(self):
        self.skeletonData = []
        self.objectPosData = []
        self.shapes = [[],[]]
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
                #step = (end - start) // 50 - 1
                step = 2
                bodyData = []
                rigidPos = []
                for i in range(start, end, step):
                    bodyCenter = np.array(dataList[0][i][:3])

                    bodyWhole = np.array(dataList[0][i]).reshape((21,3)) - bodyCenter
                    bodyWhole = bodyWhole.reshape((63))
                    bodyData.append(bodyWhole)
                    
                    objPos = np.array(dataList[2][i][:3]) - bodyCenter
                    objPos = objPos.tolist()
                    objRot = R.from_quat(dataList[2][i][3:])
                    objRot = objRot.as_euler("xyz").tolist()
                    rigidPos.append(np.array(objPos+objRot))
                self.skeletonData.append(np.array(bodyData))
                self.objectPosData.append(np.array(rigidPos))
                seqLength = len(bodyData)
                print(seqLength)
                self.shapes[0].append([seqLength, 6])
                self.shapes[1].append([seqLength, 63])
    
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

        max_length = 150

        obj_data = []
        skeleton_data = []

        for file_count in range(len(self.skeletonData)):
            pos_zeros = np.zeros((max_length, 63))
            obj_zeros = np.zeros((max_length, 6))
            inital_pos = np.array(skeletonPart[file_count][0]).astype(dt)
            delta_pos = (np.array(skeletonPart[file_count][1:]).astype(dt) - inital_pos) * 7.0
            pos_whole = np.concatenate((np.array(skeletonPart[file_count][0]).reshape((1,63)), delta_pos), axis=0).astype(dt)
            obj_whole = np.array(objPart[file_count][:]).astype(dt)

            pos_zeros[:pos_whole.shape[0],:pos_whole.shape[1]] = pos_whole
            obj_zeros[:obj_whole.shape[0],:obj_whole.shape[1]] = obj_whole

            obj_data.append(obj_zeros)
            skeleton_data.append(pos_zeros)
        dataset = tf.data.Dataset.from_tensor_slices((obj_data, skeleton_data))

        return dataset

#t = trainDataLoader()
#t.getDataset2()