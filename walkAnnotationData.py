from matplotlib import animation
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
import glob
import pickle
import time
import tensorflow as tf



fig = plt.figure()
ax = fig.add_subplot(111 , projection="3d")

startFrame = 0



connections = [(0, 1), (1, 2), (2, 3), (3, 4), 
                             (2, 5), (5, 6), (6, 7), (7, 8), 
                             (2, 9), (9, 10), (10, 11), (11, 12),
                             (0, 13), (13, 14), (14, 15), 
                             (0, 16), (16, 17), (17, 18)]

def updateHumanObj(frame, *fargs):
    global startFrame
    ax.clear()
    bodyData, objPoint, scat = fargs
    z_points = bodyData[frame, :, 2] #* -1.0
    x_points = bodyData[frame, :, 0]
    y_points = bodyData[frame, :, 1]
    print(frame + startFrame)
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

wwl_walk_phase = {30:0, 113:1, 192:-1, 259:1, 340:-1, 418:1, 493:-1}
yl_walk_phase = {78:0, 139:-1, 220:1, 300:-1, 373:1, 463:-1}
pathName = ["./Data/walkData/oneFile/wwl_board_w_001.data", "./Data/walkData/yl_table_walk_001.data"]
phasesInfo = [wwl_walk_phase, yl_walk_phase]
resultPhase = [{}, {}]

#doing visualize stuff
'''
with open(pathName[0], 'rb') as f:
    dataList = pickle.load(f)[0]
    
    print(len(dataList[0]))
    print("-----------------------")
    startFrame = 373
    endFrame = 480
    #endFrame = startFrame + 2
    videoLength = endFrame - startFrame
    skeledonData = dataList[0][startFrame:endFrame][:]
    skeledonData = np.array(skeledonData).reshape((videoLength, 21, 3))
    
    visualize(skeledonData, np.zeros([videoLength, 12, 3]), 0)
'''

def getSeq(start, end, checkOutPhases, thisFileInfo):
    if checkOutPhases[start] == 0:
        numOfPoints = 3
        theSeqIdx = np.around(np.linspace(start, end, num = numOfPoints)).astype(np.int)
        thePhase = np.linspace(0.0, checkOutPhases[end], num = numOfPoints)
        for i,idx in enumerate(theSeqIdx):
            thisFileInfo[idx] = thePhase[i]
    else: 
        numOfPoints = 6
        theSeqIdx = np.around(np.linspace(start, end, num = numOfPoints)).astype(np.int)
        thePhase = np.linspace(checkOutPhases[start], checkOutPhases[end], num = numOfPoints)
        for i,idx in enumerate(theSeqIdx[1:]):
            thisFileInfo[idx] = thePhase[i+1]

allDataX = []
allDataY = []

legRelatedPoints = [0, 13, 14, 15, 19, 16, 17, 18, 20]  #8+1
for f_num in range(2):
    with open(pathName[f_num], 'rb') as f:
        dataList = pickle.load(f)[0]
        checkOutPhases = phasesInfo[f_num]
        startList = list(checkOutPhases.keys())[:-1]
        endList = list(checkOutPhases.keys())[1:]
        thisFileInfo = {}
        for count in range(len(startList)):
            getSeq(startList[count], endList[count], checkOutPhases, thisFileInfo)
        skeletonData = []
        for frame in thisFileInfo:
            skeledonData_current = dataList[0][frame][:]
            phase = thisFileInfo[frame]
            skeledonData_current = np.array(skeledonData_current).reshape(21, 3)
            skeledonData_current[1:, :] -= skeledonData_current[0, :]
            skeledonData_current = skeledonData_current[legRelatedPoints]
            skeledonData_current = np.concatenate(([phase], skeledonData_current.reshape(( 9*3))))
            skeletonData.append(skeledonData_current)
        skeletonData = np.array(skeletonData)
        allDataX += skeletonData[:-1, :].tolist()
        allDataY += skeletonData[1:, :].tolist()
        
def getData():
    dataset = tf.data.Dataset.from_tensor_slices((allDataX, allDataY))
    return dataset
        