from matplotlib import animation
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
import glob
import pickle
import time
import tensorflow as tf
from skeletonHandle import JointsInfo
from scipy.spatial.transform import Rotation as R


fig = plt.figure()
ax = fig.add_subplot(111 , projection="3d")

startFrame = 0
step = 4

target_maxlen = 85

connections = [(0, 1), (1, 2), (2, 3), (3, 4), 
                             (2, 5), (5, 6), (6, 7), (7, 8), 
                             (2, 9), (9, 10), (10, 11), (11, 12),
                             (0, 13), (13, 14), (14, 15), 
                             (0, 16), (16, 17), (17, 18)]

def updateHumanObj(frame, *fargs):
    global startFrame, step
    ax.clear()
    bodyData, objPoint, scat = fargs
    z_points = bodyData[frame, :, 2] #* -1.0
    x_points = bodyData[frame, :, 0]
    y_points = bodyData[frame, :, 1]
    print(frame * step + startFrame)
    for connect in connections:
        a,b = connect
        ax.plot([x_points[a], x_points[b]],[y_points[a],y_points[b]],[z_points[a],z_points[b]], color="b")
    ax.scatter3D(x_points, y_points, z_points, color="r")

    thisObjPoint = objPoint[frame].reshape((12,3))
    z_points = thisObjPoint[ :, 2] #* -1.0
    x_points = thisObjPoint[ :, 0] 
    y_points = thisObjPoint[ :, 1]
    ax.scatter3D(x_points, y_points, z_points, color="g")
    #'''
    ax.plot([0.0, 0.0],[-0.7,1.2,],[0.0,0.0], color="b")
    ax.plot([-1.2,1.2],[0.0,0.0,],[0.0,0.0], color="r")
    ax.plot([0.0, 0.0],[0.0,0.0,],[-1.2,1.2], color="g")
    
    return ax


def visualize(skeledonData, objPoint, startFrame = 0):
    global fig, ax
    bodyData = skeledonData
    lenFrame = skeledonData.shape[0]
    
    bodyData = bodyData.reshape((lenFrame, 21, 3))
    ax.yaxis.set_label_position("top")
    ax.view_init(elev=117., azim=-88.)
    scat = ax.scatter(bodyData[0,:,0], bodyData[0,:,1], bodyData[0,:,2], c='r', marker = 'o',alpha=0.5, s=100)
    #time.sleep(.01)

    ani = animation.FuncAnimation(fig, updateHumanObj, frames= range(lenFrame), interval = 50, repeat_delay=100,
                                  fargs=(bodyData, objPoint, scat))
    plt.show()

#left_forward:-1, right_forward: 1, EOF:-10, Start:10
zyf_001_phase = {80:10, 490:0, 542:-1, 610:1, 800:-10}
zyf_002_phase = {100:10, 545:0, 615:1, 750:-10}
tinalf_001_phase = {40:0, 120:-1, 165:0, 460:0, 520:1, 600:-1, 710:1}
tinalf_003_phase = {100:0, 180:1, 220:0, 500:0, 600:-1, 685:1, 800:-1, 910:1}
tinalayfd_003_phase = {140:0, 172:1, 215:0, 520:0, 580:1, 650:-1, 690:0, 850:-10}

pathName = ["./Data/box_walk_small/zyf_box_lay_fd_001.data", 
            "./Data/box_walk_small/zyf_box_lay_fd_002.data", 
            "./Data/box_walk_small/tina_box2_l_f_001.data",
            "./Data/box_walk_small/tina_box2_l_f_003.data",
            "./Data/box_walk_small/tina_box2_lay_fd_003.data"]




phasesInfo = [zyf_001_phase, zyf_002_phase, tinalf_001_phase, tinalf_003_phase, tinalayfd_003_phase]
resultPhase = [{}, {}]

#doing visualize
'''
with open(pathName[0], 'rb') as f:
    dataList = pickle.load(f)[0]
    
    
    print(len(dataList[0]))
    print("-----------------------")
    startFrame = 100
    endFrame = 545
    #endFrame = startFrame + 2
    videoLength = endFrame - startFrame
    skeledonData = dataList[0][startFrame:endFrame:step][:]
    videoLength = len(skeledonData)
    skeledonData = np.array(skeledonData).reshape((videoLength, 21, 3))
    
    objData = dataList[3][startFrame:endFrame:step][:]
    objData = np.array(objData).reshape((videoLength, 12, 3))
    
    visualize(skeledonData, objData, 0)
exit()
'''


def getSeq(count, start, end, checkOutPhases, thisFileInfo):
    # start end: real frame idx in the dataList. e.g. 490; 542
    if count == 0:
        if checkOutPhases[start] == 10:
            numOfPoints = 28
            theSeqIdx = np.around(np.linspace(start, end, num = numOfPoints)).astype(np.int)
            thePhase = np.linspace(0.0, 0.0, num = numOfPoints)
            for i,idx in enumerate(theSeqIdx):
                thisFileInfo[idx] = thePhase[i]
        else:
            #thisFileInfo[start - 5] = 0.0
            thisFileInfo[start - 5] = 0.0
            numOfPoints = 6
            theSeqIdx = np.around(np.linspace(start, end, num = numOfPoints)).astype(np.int)
            thePhase = np.linspace(0.0, checkOutPhases[end], num = numOfPoints)
            for i,idx in enumerate(theSeqIdx):
                thisFileInfo[idx] = thePhase[i]
    elif checkOutPhases[end] == -10:
        numOfPoints = 18
        theSeqIdx = np.around(np.linspace(start + step, end, num = numOfPoints)).astype(np.int)
        thePhase = np.linspace(0.0, 0.0, num = numOfPoints)
        for i,idx in enumerate(theSeqIdx):
            thisFileInfo[idx] = thePhase[i]
    else: 
        numOfPoints = 12
        if (checkOutPhases[start] == -1 or checkOutPhases[start] == 1) and checkOutPhases[end] == 0.0:
            numOfPoints = 6
        elif checkOutPhases[start] == 0 and checkOutPhases[end] == 0:
            numOfPoints = 28
        theSeqIdx = np.around(np.linspace(start, end, num = numOfPoints)).astype(np.int)
        thePhase = np.linspace(checkOutPhases[start], checkOutPhases[end], num = numOfPoints)
        for i,idx in enumerate(theSeqIdx[1:]):
            thisFileInfo[idx] = thePhase[i+1]

# packed data stored here
allObjectMarkers = []
allDataX = []
allDataY = []

legRelatedPoints = [0, 13, 14, 15, 19, 16, 17, 18, 20]  #8+1  7+1
num_joints = 21

saved_Tpose = None

def prepareDataset(num_file = 5, padList = [-7, -5, -2, -1, 0, 1, 2, 3, 5, 7, ], fixNum = None):
    global saved_Tpose
    for f_num in range(num_file):
        if fixNum is not None:
            f_num = fixNum
        with open(pathName[f_num], 'rb') as f:
            dataList = pickle.load(f)[0]
            checkOutPhases = phasesInfo[f_num]
            startList = list(checkOutPhases.keys())[:-1]
            endList = list(checkOutPhases.keys())[1:]
            thisFileInfo_origin = {}
            for count in range(len(startList)):
                getSeq(count, startList[count], endList[count], checkOutPhases, thisFileInfo_origin)
            print(thisFileInfo_origin)
            
            for file_pad in padList:
                thisFileInfo = {}
                for kk in thisFileInfo_origin.keys():
                    thisFileInfo[kk + file_pad] = thisFileInfo_origin[kk]
                
                skeletonDataX = []
                skeletonDataY = []
                objectMarkers = []
                worldCenter = None
                print("this file length")
                thisFileLength = len(thisFileInfo)
                print(thisFileLength)
                for frameI, frame in enumerate(thisFileInfo):   #[0, 1, 2, ...]   [30, 42, 54, 66, ...]
                    if frameI == 0:
                        worldCenter = np.array(dataList[0][frame][:3])
                        wc_bc = np.array(dataList[0][frame][16*3:16*3+3]) - np.array(dataList[0][frame][13*3:13*3+3])
                        wc_bc[1] = 0.0
                        wc_bc = wc_bc / np.linalg.norm(wc_bc)
                        wc_bc = np.array([[0,1,0], wc_bc])
                        adj_rot = R.match_vectors(np.array([[0,1,0], [0,0,1]]), wc_bc)[0]
                        #print(R.as_euler(adj_rot, "xyz"))
                        
                    #bodyCenter = adj_rot.apply(np.array(dataList[0][frame][:3]) - worldCenter)
                    #bodyWholePos = adj_rot.apply(np.array(dataList[0][frame][:]).reshape((21,3))- np.array(dataList[0][frame][:3]))
                        
                    skeledonData_current = dataList[0][frame][:]
                    phase = thisFileInfo[frame]
                    phaseLabel = 0.0
                    if phase > 0.001 or phase < -0.001:
                        phaseLabel = 1.0
                    skeledonData_current = np.array(skeledonData_current).reshape(21, 3)
                    skeledonData_current[1:, :] -= skeledonData_current[0, :]  #
                    
                    skeledonData_current[0, :] = adj_rot.apply(skeledonData_current[0, :] - worldCenter)
                    skeledonData_current[1:, :] = adj_rot.apply(skeledonData_current[1:, :])
                    
                    JI = JointsInfo(skeledonData_current)
                    
                    #skeledonData_current[0, :] -= worldCenter
                    
                    #'''
                    if frameI == 0:
                        allRotsEul = JI.get_all_rots()
                        for ti in range(21):
                            #print(allRotsEul[ti].as_euler("xyz", degrees=True))
                            pass
                        #exit()
                    #'''
                    
                    allRots = JI.get_all_rots_vecs()
                    if saved_Tpose is None and frameI == 0:
                        saved_Tpose = JI.generate_standard_Tpose_from_this_bone_length()
                    
                    bodyRots = []
                    for rotI, rot in enumerate(allRots):
                        #if rotI not in legRelatedPoints: #and rotI is not 0:
                            #continue
                        bodyRots += rot[0].tolist()  
                        bodyRots += rot[1].tolist()
                    
                    bodyRots = np.array(bodyRots)
                    #skeledonData_current = skeledonData_current[legRelatedPoints]
                    bodyCenter = skeledonData_current.reshape(( num_joints*3))[:3]
                    bodyPoses = skeledonData_current.reshape(( num_joints*3))[3:]
                    
                    objMarkers = adj_rot.apply(np.array(dataList[3][frame]) - worldCenter).reshape(12 * 3)
                    
                    objectMarkers.append(objMarkers)
                    currentX = np.concatenate(([phase],  bodyCenter, 
                                                        bodyRots  * np.sin(phase * 3.14159265359) , bodyPoses  * np.sin(phase * 3.14159265359)
                                                        ,bodyRots  * np.cos(phase * 3.14159265359) , bodyPoses  * np.cos(phase * 3.14159265359)   ))
                    currentY = np.concatenate(([phase], bodyCenter, bodyRots, bodyPoses))
                    skeletonDataX.append(currentX)
                    skeletonDataY.append(currentY)
                    
                    if frameI == thisFileLength - 1:
                        last_markers = objMarkers[:]
                        last_SkeletonX = currentX[:]
                        last_SkeletonY = currentY[:]
                        restLength = target_maxlen - thisFileLength
                        for add_frame in range(restLength):
                            objectMarkers.append(last_markers)
                            skeletonDataX.append(last_SkeletonX)
                            skeletonDataY.append(last_SkeletonY)
                        print("pad to ") 
                        print(restLength + thisFileLength)
                
                
                
                objectMarkers = np.array(objectMarkers)
                skeletonDataX = np.array(skeletonDataX)
                skeletonDataY = np.array(skeletonDataY)
                
                allDataX.append(skeletonDataX[:-1, :])
                allDataY.append(skeletonDataY[1:, :])
                
                allObjectMarkers.append(objectMarkers[1:, :])
                
                print(np.array(allObjectMarkers).shape)
                print(np.array(allDataX).shape)
                print(np.array(allDataY).shape)
            

'''
allDataY_vis = np.array(allDataY)
num_data = len(allDataY)
allDataY_vis_global = allDataY_vis[:, 1:4].reshape((num_data, 1, 3))
allDataY_vis = allDataY_vis_global + allDataY_vis[:, -60:].reshape((num_data, num_joints - 1, 3))
allDataY_vis = np.concatenate((allDataY_vis_global, allDataY_vis), axis=1)
visualize(allDataY_vis, np.array(allObjectMarkers).reshape((num_data, 12, 3)), 0)
'''

def getData():
    prepareDataset()
    dataset = tf.data.Dataset.from_tensor_slices((allObjectMarkers, allDataX, allDataY))
    return dataset

def getTestSample():
    prepareDataset(1, [0], fixNum=1)
    dataset = tf.data.Dataset.from_tensor_slices((allObjectMarkers, allDataX, allDataY))
    return dataset