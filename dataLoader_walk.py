import tensorflow as tf
import numpy as np
import glob
import pickle
from scipy.spatial.transform import Rotation as R
import random
from skeletonHandle import JointsInfo
from matplotlib import pyplot as plt

#startEnd = {"t1":[220,360], "t2":[60, 320], "t3":[240, 350], "t4":[120, 260], "t5":[85, 230], "t6":[60, 270]}
#startEnd = {"t1":[180,360], "t2":[30, 320], "t3":[180, 350], "t4":[60, 260], "t5":[30, 230], "t6":[20, 270],
#            "n1":[50, 270], "n2":[100, 290], "n3":[80, 300], "n4":[130, 290], "n5":[75, 210], "n6":[100, 270],
#            "n7":[100, 280], "n8":[70, 280], "n9":[120, 300], "n10":[20, 115], "n11":[100, 230], "n12":[70, 230],
#            "n13":[60, 200], "n14":[50, 180], "n15":[60, 190], "n16":[60, 160], "n17":[100, 240], "n18":[70, 270],
#            "n19":[90, 260], "n20":[70, 280], "n21":[80, 260], "n22":[80, 240], "n23":[90, 280]}

startEndStandPos = {"t1":[360], "t2":[320], "t3":[350], "t4":[260], "t5":[230], "t6":[270],
            "n1":[270], "n2":[290], "n3":[300], "n4":[290], "n5":[210], "n6":[270],
            "n7":[280], "n8":[280], "n9":[300], "n10":[115], "n11":[230], "n12":[230],
            "n13":[200], "n14":[180], "n15":[190], "n16":[160], "n17":[240], "n18":[270],
            "n19":[260], "n20":[280], "n21":[260], "n22":[240], "n23":[280]}

startEnd = {"t1":[352], "t2":[320], "t3":[388], "t4":[257], "t5":[212], "t6":[283],
            "n1":[406], "n2":[434], "n3":[412], "n4":[426], "n5":[351], "n6":[383],
            "n7":[373], "n8":[357], "n9":[368], "n10":[179], "n11":[322], "n12":[299],
            "n13":[280], "n14":[257], "n15":[266], "n16":[273], "n17":[322], "n18":[325],
            "n19":[349], "n20":[368], "n21":[384], "n22":[299], "n23":[333], 
            "box2_p2":[332], "box2_p3":[158], "box2_p4":[380], "box2_p5":[316], "box3_p1":[272],"box3_p2":[307] ,
            "box5_p1":[78], "box5_p2":[81], "box5_p3":[84]}


humanDimension_pos = 20 * 3
humanDimension_rot = 21 * 6 

humanDimension = 3 + humanDimension_rot + humanDimension_pos  #global trans + all rots + joint trans = 3+126+60 = 189
humanDimensionWalk = 1+2+3+6+6*8+3*8
humanDimensionWalkOutput = humanDimensionWalk - 3
objDimention = 83 * 2  #36  fake now



class trainDataLoader:
    def __init__(self, num_files = None, pathNames = ["./Data/liftOnly/*.data", "./Data/data_new/*.data"]):
        
        self.maxLen = 85#85  35
        
        self.skeletonData = []
        self.objectPosData = []
        self.objectMarkerData = []
        self.shapes = [[],[]]
        self.obj_data = []
        self.skeleton_data = []
        self.saved_Tpose = None
        #self.is_walk = is_walk

        self.loadFromFile(num_files, pathNames)

        self.obj_mean = 0.0
        self.ppl_mean_p = 0.0
        self.ppl_mean_a = 0.0
        self.obj_std = 0.0
        self.ppl_std_p = 0.0
        self.ppl_std_a = 0.0
        self.prepared = False
        self.pose_mean = None
        self.pose_cov_inv = None
        self.pose_mean_init = None
        self.pose_cov_inv_init = None

        self.wholeMean = 0.0
        self.wholeStd = 0.0

        self.flipData = False
        self.useRotation = True

        self.obj_mean_saved = 0.20327034346397518
        self.obj_std_saved = 0.42398764560154273

        self.ppl_mean_saved_a = 0.09123124554080177
        self.ppl_std_saved_a = 0.7756390174495723
        self.ppl_mean_saved_p = 0.07805996929042856
        self.ppl_std_saved_p = 0.3341384650987687

        
        
    def updateNorm(self):
        temp_obj = np.array(self.obj_data).flatten()
        #temp_pos = np.concatenate((temp_pos, np.array(self.skeleton_data)[:, 0, :].flatten()))
        self.obj_mean = np.mean(temp_obj)
        self.obj_std = np.std(temp_obj)

        #temp_ppl_a = np.array(self.skeleton_data)[:, :, 3:17*3].flatten()
        #self.ppl_mean_a = np.mean(temp_ppl_a)
        #self.ppl_std_a = np.std(temp_ppl_a)
        temp_ppl_p = np.array(self.skeleton_data)[:, :, -24:].flatten()
        self.ppl_mean_p = np.mean(temp_ppl_p)
        self.ppl_std_p = np.std(temp_ppl_p)
        #temp_obj = (temp_obj - self.obj_mean) / self.obj_std
        #temp_obj /= 3.14159265
        temp_whole = np.array(self.skeleton_data)[:, :, :].flatten()
        self.wholeMean = np.mean(temp_whole)
        self.wholeStd = np.std(temp_whole)
        
        #plt.plot(temp_obj)
        #plt.show()

        
        #ppl_array = np.array(self.skeleton_data)[:, :10, 3:]
        #ppl_array = (ppl_array - self.ppl_mean) / self.ppl_std
        #ppl_array /= 3.1415926536
        #ppl_array *= 100.0
        #initialFrame = ppl_array.reshape((-1,51)).astype(np.float64)

        #self.pose_mean_angle = ppl_array[:,:,3:].mean(axis=(0,1))
        #pose_cov = np.cov(ppl_array[:,:,3:].reshape((-1,60)).T)
        #self.pose_cov_inv = np.linalg.inv(pose_cov)
        #self.pose_mean = ppl_array[:,:,3:].reshape((-1,3)).mean(axis=0)
        #pose_cov = np.cov(ppl_array[:,:,3:].reshape((-1,3)).T)
        #self.pose_cov_inv = np.linalg.inv(pose_cov)

        '''
        self.pose_mean_init = initialFrame.mean(axis=0)
        np.set_printoptions(threshold=30000)
        #print(self.pose_mean_init)
        #print()
        pose_cov_init = np.cov(initialFrame[:, :].T)
        #print(pose_cov_init[-1])
        #print(np.linalg.cond(pose_cov_init))
        #print( 1.0 / np.finfo(pose_cov_init.dtype).eps)
        #exit()
        self.pose_cov_inv_init = np.linalg.pinv(pose_cov_init) #* 1e-11
        #print(self.pose_cov_inv_init[-1])
        #print( self.pose_cov_inv_init.dot(pose_cov_init))
        #exit()
        '''
        

        
    
    def loadFromFile(self, num_files = None, pathNames = ["./Data/liftOnly/*.data", "./Data/data_new/*.data"]):
        print("loading files from")
        files = []
        playSpeedFactor = 1
        legRelatedPoints = [13, 14, 15, 19, 16, 17, 18, 20]
        for pathName in pathNames:
            print(pathName)
            files += glob.glob(pathName)
        if num_files is not None:
            files = files[:num_files]
        for f in files:
            print(f)
            def handleSingleFile(start, end, step):
                thisFileLegDis = []
                with open(f, 'rb') as pickle_file:
                    dataList = pickle.load(pickle_file)[0]
                    #print(len(dataList[0]))

                    
                    if "box5" in f:
                        step = 1
                    
                    for k in startEnd:
                        if k in f:
                            start = startEnd[k][0]
                            '''
                            try:
                                end = startEnd[k][1]
                            except IndexError:
                            '''
                            end = start + step * self.maxLen
                    #step = (end - start) // 50 - 1
                    
                    bodyData = []
                    rigidPos = []
                    markerPos = []
                    worldCenter = np.array(dataList[0][start][:3])
                    #[1,0,0], worldCenter->boxCenter
                    

                    #wc_bc = np.array(dataList[2][start][:3]) - worldCenter
                    wc_bc = np.array(dataList[0][start][16*3:16*3+3]) - np.array(dataList[0][start][13*3:13*3+3])                   
                    #wc_bc = np.mean(np.array(dataList[3][start])) - worldCenter
                    
                    wc_bc[1] = 0.0
                    wc_bc = wc_bc / np.linalg.norm(wc_bc)
                    wc_bc = np.array([[0,1,0], wc_bc])
                    adj_rot = R.match_vectors(np.array([[0,1,0], [0,0,1]]), wc_bc)[0]
                    #print(adj_rot.as_euler("xyz", degrees=True))

                    dataFrameCount = 0

                    for i in range(start, end, step):
                        #i += step
                        dataFrameCount += 1
                        if dataFrameCount > self.maxLen:
                            break
                        try:
                            midPointofHip = (np.array(dataList[0][i][13*3:13*3+3]) + np.array(dataList[0][i][16*3:16*3+3])) / 2.0
                            dataList[0][i][:3] = midPointofHip.tolist()
                            bodyCenter = adj_rot.apply(np.array(dataList[0][i][:3]) - worldCenter)
                            bodyWholePos = adj_rot.apply(np.array(dataList[0][i][:]).reshape((21,3))- np.array(dataList[0][i][:3]))
                            if "box2" in f or "box3" in f:
                                bodyWholePos[9] = bodyWholePos[10] - np.array([.0, .0, 0.1])  #fix data error
                                bodyWholePos[5] = bodyWholePos[6] + np.array([.0, .0, 0.1])  #fix data error
                            #bodyWhole[1:, :] -= bodyCenter
                        except IndexError:
                            print(f)
                            print("the length is "+ str(i))
                            print(len(dataList[0]))
                            print()
                            break

                        #bodyWhole = np.concatenate((centerDistant.reshape((1,3)), bodyWhole), axis=0)
                        JI = JointsInfo(bodyWholePos)
                        allRots = JI.get_all_rots_vecs()
                        if self.saved_Tpose is None and i > end - 10:
                            self.saved_Tpose = JI.generate_standard_Tpose_from_this_bone_length()

                        bodyWhole = [bodyCenter]
                        for rotI, rot in enumerate(allRots):
                            if rotI not in legRelatedPoints and rotI is not 0:
                                continue
                            bodyWhole.append(rot[0])  
                            bodyWhole.append(rot[1])#21 * 6 = 126
                        bodyWalkPos = bodyWholePos[legRelatedPoints]
                        
                        bodyWhole = np.array(bodyWhole).flatten()
                        bodyWhole = np.concatenate((bodyWhole, bodyWalkPos.flatten()))
                        
                        #get the direction control vec
                        bodyCenterNext = adj_rot.apply(np.array(dataList[0][i+step][:3]) - worldCenter)
                        controlVector = bodyCenterNext - bodyCenter
                        controlVector = np.array([controlVector[0], controlVector[2]])
                        
                        #get the phase
                        leftToe = adj_rot.apply(np.array(dataList[0][i][15*3:15*3+3]) - worldCenter)
                        rightToe = adj_rot.apply(np.array(dataList[0][i][18*3:18*3+3]) - worldCenter)
                        rightVec = np.array(dataList[0][start][16*3:16*3+3]) - np.array(dataList[0][start][13*3:13*3+3]) 
                        frontVec = np.cross(-1.0 * rightVec, np.array([.0, 1.0, .0]))
                        diffTwoToe = np.dot((leftToe - rightToe), frontVec)
                        thisFileLegDis.append(diffTwoToe)
                        bodyWhole = np.concatenate((controlVector, bodyWhole))
                        

                        #bodyWhole = bodyWhole.reshape((humanDimension2 + humanDimension1))
                        bodyData.append(bodyWhole)
                        
                        objPos = adj_rot.apply(np.array(dataList[2][i][:3]) - worldCenter)
                        objPos = objPos.tolist()
                        objRot = R.from_quat(dataList[2][i][3:])
                        objRot = objRot.as_euler("xyz").tolist()
                        rigidPos.append(np.array(objPos+objRot))

                        objMarkers = adj_rot.apply(np.array(dataList[3][i]) - worldCenter)
                        markerPos.append(objMarkers.reshape((12*3)))
                delta_phase = []
                countTemp = 0
                for phase in range(0, len(thisFileLegDis) - 1):
                    deltaHere = 0.0
                    if phase != countTemp + len(delta_phase):
                        print(phase)
                        print(countTemp)
                        print(delta_phase)
                        exit()
                    if phase == 0:
                        deltaHere = thisFileLegDis[phase + 1] - thisFileLegDis[phase]
                        if abs(deltaHere) < 0.0001 and abs(thisFileLegDis[phase]) < 0.03:
                            delta_phase.append(0.0)
                            countTemp = 0
                        else:
                            countTemp += 1
                        continue
                    else:
                        deltaHere = thisFileLegDis[phase] - thisFileLegDis[phase - 1]
                    if abs(deltaHere) < 0.0001 and abs(thisFileLegDis[phase]) < 0.03:
                        for ik in range(countTemp):
                            delta_phase.append(0.0)
                        delta_phase.append(0.0)
                        countTemp = 0
                    elif thisFileLegDis[phase - 1] > thisFileLegDis[phase] and thisFileLegDis[phase + 1] > thisFileLegDis[phase]:
                        temp = np.linspace(-0.07, -0.95, num = countTemp)
                        for tt in temp:
                            delta_phase.append(tt)
                        delta_phase.append(-1.0)
                        countTemp = 0
                        
                    elif thisFileLegDis[phase - 1] < thisFileLegDis[phase] and thisFileLegDis[phase + 1] < thisFileLegDis[phase]:
                        temp = np.linspace(0.07, 0.95, num = countTemp)
                        for tt in temp:
                            delta_phase.append(tt)
                        delta_phase.append(1.0)
                        countTemp = 0
                    elif (thisFileLegDis[phase - 1] > 0 and thisFileLegDis[phase] < 0) :
                        temp = np.linspace(0.95, 0.07, num = countTemp)
                        for tt in temp:
                            delta_phase.append(tt)
                        delta_phase.append(0.0)
                        countTemp = 0
                    elif (thisFileLegDis[phase - 1] < 0 and thisFileLegDis[phase] > 0):
                        temp = np.linspace( -0.95, -0.07,num = countTemp)
                        for tt in temp:
                            delta_phase.append(tt)
                        delta_phase.append(0.0)
                        countTemp = 0
                    else:
                        countTemp += 1
                countTemp += 1
                if thisFileLegDis[-1] < 0:
                    if delta_phase[-1] < -0.999:
                        temp = np.linspace( -0.95, -0.92+0.07 * countTemp,num = countTemp)
                        for tt in temp:
                            delta_phase.append(tt)
                    else:
                        temp = np.linspace( -0.07, -0.09 + -0.06 * countTemp,num = countTemp)
                        for tt in temp:
                            delta_phase.append(tt)
                elif thisFileLegDis[-1] > 0:
                    if delta_phase[-1] > 0.999:
                        temp = np.linspace( 0.95, 0.93-0.07 * countTemp,num = countTemp)
                        for tt in temp:
                            delta_phase.append(tt)
                    else:
                        temp = np.linspace( 0.07, 0.09 + 0.06 * countTemp,num = countTemp)
                        for tt in temp:
                            delta_phase.append(tt)
                print(len(delta_phase))
                phase_angle = np.array(delta_phase) * 3.14159265359
                phase_angle_sin = np.sin(phase_angle)
                phase_angle_cos = np.cos(phase_angle)
                bodyDataSave = np.array(bodyData).copy()   # (85, 83)?
                bodyData = np.concatenate(( np.array(delta_phase).reshape(85, 1), np.array(bodyData)), axis=1)
                self.skeletonData.append(bodyData)
                self.objectPosData.append(np.array(rigidPos))

                markerPos1 = np.einsum("a,ab->ab", phase_angle_sin, bodyDataSave)
                markerPos2 = np.einsum("a,ab->ab", phase_angle_cos, bodyDataSave)
                markerPos = np.concatenate((markerPos1, markerPos2), axis=-1)
                self.objectMarkerData.append(np.array(markerPos))

                #seqLength = len(bodyData)

                #self.shapes[0].append([seqLength, 6])
                #self.shapes[1].append([seqLength, humanDimension2])
            
            if "walk" in f or "tina_box2_w_" in f:
                handleSingleFile(1, 4*85, 4)
                handleSingleFile(2, 4*85, 4)
                handleSingleFile(3, 4*85, 4)
                if "wwl_board_w_001" in f:
                    handleSingleFile(85*4+1, 85*4+1+4*85, 4)
                    handleSingleFile(3, 4+5*85, 5)
                    handleSingleFile(4, 4+5*85, 5)
                else:
                    handleSingleFile(3, 3+5*85, 5)
                    handleSingleFile(4, 3+5*85, 5)
                #handleSingleFile(2*85, 2*85*2, 2)
            else:
                start = 0
                end = 100
                step = 3 #3 7
                handleSingleFile(start, end, step)
    

    def prepareDataset(self):
        # Create dataset includes marker pos
        skeletonPart = self.skeletonData
        dt = np.float32
        objPart = self.objectMarkerData

        max_length = self.maxLen  #130

        for file_count in range(len(self.skeletonData)):
            pos_zeros = np.zeros((max_length, humanDimensionWalk))
            obj_zeros = np.zeros((max_length, objDimention))
            #inital_pos = np.array(skeletonPart[file_count][0]).astype(dt)
            #delta_pos = (np.array(skeletonPart[file_count][1:]).astype(dt) - inital_pos)
            #pos_whole = np.concatenate((np.array(skeletonPart[file_count][0]).reshape((1,63)), delta_pos), axis=0).astype(dt)
            pos_whole = np.array(skeletonPart[file_count][:]).astype(dt)  #85, 3+17*3+60 = 114
            
            obj_whole = np.array(objPart[file_count][:]).astype(dt)

            pos_zeros[:pos_whole.shape[0],:pos_whole.shape[1]] = pos_whole
            obj_zeros[:obj_whole.shape[0],:obj_whole.shape[1]] = obj_whole

            #pos_zeros = pos_zeros[:, 3:] #remove

            self.obj_data.append(obj_zeros)
            self.skeleton_data.append(pos_zeros)

            if self.flipData:
                pos_zeros = np.zeros((max_length, humanDimensionWalk))
                obj_zeros = np.zeros((max_length, objDimention))
                pos_whole = np.array(skeletonPart[file_count][:]).astype(dt)
                obj_whole = np.array(objPart[file_count][:]).astype(dt)
                pos_whole = np.flip(pos_whole, axis=0)
                obj_whole = np.flip(obj_whole, axis=0)

                wold_center = np.copy(pos_whole[0, :3])
                print(wold_center)
                #exit()
                pos_whole_temp = pos_whole.reshape((self.maxLen, 38, 3)) #114=3+ 17*3 + 20 * 3
                pos_whole_temp[:, 0, :] -= wold_center
                #pos_whole_temp[:, -20:, :] -= wold_center
                #pos_whole[:, :] -= wold_center
                pos_whole = pos_whole_temp.reshape((self.maxLen, 38 * 3))

                obj_whole_temp = obj_whole.reshape((self.maxLen,12,3))
                obj_whole_temp -= wold_center
                obj_whole = obj_whole_temp.reshape((self.maxLen,36))

                pos_zeros[:pos_whole.shape[0],:pos_whole.shape[1]] = pos_whole
                obj_zeros[:obj_whole.shape[0],:obj_whole.shape[1]] = obj_whole
                self.obj_data.append(obj_zeros)
                self.skeleton_data.append(pos_zeros)


        self.updateNorm()
        save_file_num = 0
        write_mode = 0  #0->None 1->pose 2->angles
        for i in range(len(self.obj_data)):
            #self.skeleton_data[i][:, -24:] = (self.skeleton_data[i][:, -24:] - self.ppl_mean_p) / self.ppl_std_p
            self.skeleton_data[i][:, :] = (self.skeleton_data[i][:, :] - self.wholeMean) / self.wholeStd
            self.obj_data[i] = (self.obj_data[i] - self.obj_mean) / self.obj_std 
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
            print("++++++++++ mean  std ++++++++++=")
            print(self.obj_mean)
            print(self.obj_std)
            print()
            print(self.ppl_mean_a)
            print(self.ppl_std_a)
            print(self.ppl_mean_p)
            print(self.ppl_std_p)
        #dataset = tf.data.Dataset.from_tensor_slices((self.obj_data, self.skeleton_data))
        dataset = tf.data.Dataset.from_tensor_slices(( self.skeleton_data, self.obj_data))
        return dataset
    
    def getSingleSample(self, file_num):
        x = self.obj_data[file_num]
        y = self.skeleton_data[file_num]
        return x,y


#t = trainDataLoader(num_files = None, pathNames = ["./Data/walkData/*.data"])
#t = trainDataLoader()
#t.getDataset3()