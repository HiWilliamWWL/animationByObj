import tensorflow as tf
import numpy as np
import glob
import pickle
from scipy.spatial.transform import Rotation as R
import random

#startEnd = {"t1":[220,360], "t2":[60, 320], "t3":[240, 350], "t4":[120, 260], "t5":[85, 230], "t6":[60, 270]}
#startEnd = {"t1":[180,360], "t2":[30, 320], "t3":[180, 350], "t4":[60, 260], "t5":[30, 230], "t6":[20, 270],
#            "n1":[50, 270], "n2":[100, 290], "n3":[80, 300], "n4":[130, 290], "n5":[75, 210], "n6":[100, 270],
#            "n7":[100, 280], "n8":[70, 280], "n9":[120, 300], "n10":[20, 115], "n11":[100, 230], "n12":[70, 230],
#            "n13":[60, 200], "n14":[50, 180], "n15":[60, 190], "n16":[60, 160], "n17":[100, 240], "n18":[70, 270],
#            "n19":[90, 260], "n20":[70, 280], "n21":[80, 260], "n22":[80, 240], "n23":[90, 280]}

startEnd = {"t1":[360], "t2":[320], "t3":[350], "t4":[260], "t5":[230], "t6":[270],
            "n1":[270], "n2":[290], "n3":[300], "n4":[290], "n5":[210], "n6":[270],
            "n7":[280], "n8":[280], "n9":[300], "n10":[115], "n11":[230], "n12":[230],
            "n13":[200], "n14":[180], "n15":[190], "n16":[160], "n17":[240], "n18":[270],
            "n19":[260], "n20":[280], "n21":[260], "n22":[240], "n23":[280]}


humanDimension = 63
objDimention = 36



class trainDataLoader:
    def __init__(self):
        self.skeletonData = []
        self.objectPosData = []
        self.objectMarkerData = []
        self.shapes = [[],[]]
        self.obj_data = []
        self.skeleton_data = []
        self.loadFromFile()
        self.obj_mean = 0.0
        self.ppl_mean = 0.0
        self.obj_std = 0.0
        self.ppl_std = 0.0
        self.prepared = False
        self.pose_mean = None
        self.pose_cov_inv = None
        self.pose_mean_init = None
        self.pose_cov_inv_init = None

        self.flipData = True
        self.useRotation = True
        
    def updateNorm(self):
        temp_obj = np.array(self.obj_data).flatten()
        #temp_pos = np.concatenate((temp_pos, np.array(self.skeleton_data)[:, 0, :].flatten()))
        self.obj_mean = np.mean(temp_obj)
        self.obj_std = np.std(temp_obj)

        temp_ppl = np.array(self.skeleton_data).flatten()
        self.ppl_mean = np.mean(temp_ppl)
        self.ppl_std = np.std(temp_ppl)

        ppl_array = np.array(self.skeleton_data)
        ppl_array = (ppl_array - self.ppl_mean) / self.ppl_std
        initialFrame = ppl_array[:,:10,3:].reshape((-1,60))

        self.pose_mean = ppl_array[:,:,3:].mean(axis=(0,1))
        pose_cov = np.cov(ppl_array[:,:,3:].reshape((-1,60)).T)
        self.pose_cov_inv = np.linalg.inv(pose_cov)
        #self.pose_mean = ppl_array[:,:,3:].reshape((-1,3)).mean(axis=0)
        #pose_cov = np.cov(ppl_array[:,:,3:].reshape((-1,3)).T)
        #self.pose_cov_inv = np.linalg.inv(pose_cov)


        self.pose_mean_init = initialFrame.mean(axis=0)
        pose_cov_init = np.cov(initialFrame.reshape((-1,60)).T)
        self.pose_cov_inv_init = np.linalg.inv(pose_cov_init) #* 1e-11

        
    
    def loadFromFile(self, pathName = "./Data/liftOnly/*.data"):
        files = glob.glob(pathName)
        print("loading files from")
        print(pathName)
        for f in files:
            with open(f, 'rb') as pickle_file:
                dataList = pickle.load(pickle_file)[0]
                start = 0
                end = 100
                for k in startEnd:
                    if k in f:
                        start = startEnd[k][0]
                        try:
                            end = startEnd[k][1]
                        except IndexError:
                            end = start + 250
                #step = (end - start) // 50 - 1
                step = 3
                bodyData = []
                rigidPos = []
                markerPos = []
                worldCenter = np.array(dataList[0][start][:3])
                #[1,0,0], worldCenter->boxCenter
                wc_bc = np.array(dataList[2][start][:3]) - worldCenter
                wc_bc[1] = 0.0
                wc_bc = np.array([wc_bc, [0,1,0]])
                adj_rot = R.match_vectors(np.array([[1,0,0], [0,1,0]]), wc_bc)[0]
                #adj_rot = R.match_vectors(wc_bc, wc_bc)[0]
                #print(adj_rot.as_euler("xyz", degrees=True))
                #print(wc_bc)
                #exit()

                for i in range(start, end, step):
                    try:
                    #bodyCenter = np.array(dataList[0][i][:3])
                        
                        #worldCenter = np.array(dataList[2][0][:3])
                        

                        bodyCenter = adj_rot.apply(np.array(dataList[0][i][:3]) - worldCenter)
                        #centerDistant = np.array(dataList[0][i][:3]) - bodyCenter

                        bodyWhole = adj_rot.apply(np.array(dataList[0][i][:]).reshape((21,3))- worldCenter)

                        bodyWhole[1:, :] -= bodyCenter
                    except IndexError:
                        print(f)
                        print("the length is "+ str(i))
                        print()
                        break

                    #bodyWhole = np.concatenate((centerDistant.reshape((1,3)), bodyWhole), axis=0)

                    bodyWhole = bodyWhole.reshape((humanDimension))
                    bodyData.append(bodyWhole)
                    
                    objPos = adj_rot.apply(np.array(dataList[2][i][:3]) - worldCenter)
                    objPos = objPos.tolist()
                    objRot = R.from_quat(dataList[2][i][3:])
                    objRot = objRot.as_euler("xyz").tolist()
                    rigidPos.append(np.array(objPos+objRot))

                    objMarkers = adj_rot.apply(np.array(dataList[3][i]) - worldCenter)
                    markerPos.append(objMarkers.reshape((12*3)))

            self.skeletonData.append(np.array(bodyData))
            self.objectPosData.append(np.array(rigidPos))
            self.objectMarkerData.append(np.array(markerPos))
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

        for file_count in range(len(self.skeletonData)):
            pos_zeros = np.zeros((max_length, 63))
            obj_zeros = np.zeros((max_length, 6))
            inital_pos = np.array(skeletonPart[file_count][0]).astype(dt)
            delta_pos = (np.array(skeletonPart[file_count][1:]).astype(dt) - inital_pos) * 5.0
            pos_whole = np.concatenate((np.array(skeletonPart[file_count][0]).reshape((1,63)), delta_pos), axis=0).astype(dt)

            obj_whole = np.array(objPart[file_count][:]).astype(dt)

            pos_zeros[:pos_whole.shape[0],:pos_whole.shape[1]] = pos_whole
            obj_zeros[:obj_whole.shape[0],:obj_whole.shape[1]] = obj_whole

            self.obj_data.append(obj_zeros)
            self.skeleton_data.append(pos_zeros)
        dataset = tf.data.Dataset.from_tensor_slices((obj_data, skeleton_data))

        return dataset

    def prepareDataset(self):
        # Create dataset includes marker pos
        skeletonPart = self.skeletonData
        dt = np.float32
        objPart = self.objectMarkerData

        max_length = 85  #130

        for file_count in range(len(self.skeletonData)):
            pos_zeros = np.zeros((max_length, humanDimension))
            obj_zeros = np.zeros((max_length, objDimention))
            #inital_pos = np.array(skeletonPart[file_count][0]).astype(dt)
            #delta_pos = (np.array(skeletonPart[file_count][1:]).astype(dt) - inital_pos)
            #pos_whole = np.concatenate((np.array(skeletonPart[file_count][0]).reshape((1,63)), delta_pos), axis=0).astype(dt)
            pos_whole = np.array(skeletonPart[file_count][:]).astype(dt)
            
            obj_whole = np.array(objPart[file_count][:]).astype(dt)

            pos_zeros[:pos_whole.shape[0],:pos_whole.shape[1]] = pos_whole
            obj_zeros[:obj_whole.shape[0],:obj_whole.shape[1]] = obj_whole

            self.obj_data.append(obj_zeros)
            self.skeleton_data.append(pos_zeros)

            if self.flipData:
                pos_zeros = np.zeros((max_length, humanDimension))
                obj_zeros = np.zeros((max_length, objDimention))
                pos_whole = np.array(skeletonPart[file_count][:]).astype(dt)
                obj_whole = np.array(objPart[file_count][:]).astype(dt)
                pos_whole = np.flip(pos_whole, axis=0)
                obj_whole = np.flip(obj_whole, axis=0)

                pos_zeros[:pos_whole.shape[0],:pos_whole.shape[1]] = pos_whole
                obj_zeros[:obj_whole.shape[0],:obj_whole.shape[1]] = obj_whole
                self.obj_data.append(obj_zeros)
                self.skeleton_data.append(pos_zeros)


        self.updateNorm()
        for i in range(len(self.obj_data)):
            #self.obj_data[i] = (self.obj_data[i] - self.pos_mean) / self.pos_std
            #self.skeleton_data[i][0, :] = (self.skeleton_data[i][0, :] - self.pos_mean) / self.pos_std
            self.skeleton_data[i] = (self.skeleton_data[i] - self.ppl_mean) / ( self.ppl_std)
            #self.skeleton_data[i] *= 5.0
            #print(self.skeleton_data[i][:5])
        #exit()
        '''
        newOrder = [x for x in range(len(self.obj_data))]
        random.shuffle(newOrder)
        dataObj = []
        dataSkeleton = []
        print("here")
        for order in newOrder:
            dataObj.append(self.obj_data[order])
            dataSkeleton.append(self.skeleton_data[order])
        self.obj_data = dataObj
        self.skeleton_data = dataSkeleton
        '''
    
    def getDataset3(self):
        if not self.prepared:
            self.prepareDataset()
            self.prepared = True
        dataset = tf.data.Dataset.from_tensor_slices((self.obj_data, self.skeleton_data))
        return dataset
    
    def getSingleSample(self, file_num):
        x = self.obj_data[file_num]
        y = self.skeleton_data[file_num]
        return x,y


#t = trainDataLoader()
#t.getDataset3()