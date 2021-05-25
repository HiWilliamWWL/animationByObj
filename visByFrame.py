import numpy as np
import pickle
import math
from numpy import linalg as LA
from scipy.spatial.transform import Rotation as R
import glob
import matplotlib.pyplot as plt
from math import sqrt, log
import time


#fileName = "./part0/wwl_table_drag_r_002.data"
#fileName = "./part1/tina_box2_l1s_b_003.data"
objName = "tripod"

#fileName = "./part0/zyf_box_l1s_b_001.data"
#fileName = "./liftOnly/t1.data"
fileName = "./dataPlace/dataReady/May7Task_023.data"
connections = [(0, 1), (1, 2), (2, 3), (3, 4), 
                             (2, 5), (5, 6), (6, 7), (7, 8), 
                             (2, 9), (9, 10), (10, 11), (11, 12),
                             (0, 13), (13, 14), (14, 15), 
                             (0, 16), (16, 17), (17, 18), (15,19), (18,20)]
    

def getNormVecs(xc, yc, zc, markerDataFrame0):
    x1,x2 = xc
    y1,y2 = yc
    z1,z2 = zc
    x1 = markerDataFrame0[x1]
    x2 = markerDataFrame0[x2]
    y1 = markerDataFrame0[y1]
    y2 = markerDataFrame0[y2]
    z1 = markerDataFrame0[z1]
    z2 = markerDataFrame0[z2]
    x = x2 - x1
    y = y2 - y1
    z = z2 - z1
    x = x / LA.norm(x)
    y = y / LA.norm(y)
    z = z / LA.norm(z)
    return (x,y,z)

def relocateData(dataList, objName, startFrame = 0):
    bodyData = np.array(dataList[0])
    lenFrame = bodyData.shape[0]
    print(bodyData.shape)
    name = objName
    bbx= 0.1
    bby = 0.1
    bbz = 0.1
    bb = np.array([[[bbx,bby,bbz],[-bbx,bby,bbz],[bbx,-bby,bbz],[bbx,bby,-bbz],[-bbx,-bby,bbz],[-bbx,bby,-bbz],[bbx,-bby,-bbz],[-bbx,-bby,-bbz] ]])
    bb = np.tile(bb, (lenFrame, 1, 1))
    #objData = np.array(dataList[2])
    markerData = np.array(dataList[3])
    rigidPos = np.array(dataList[2])
    print(rigidPos.shape)
    print(markerData.shape)
    #print(rigidPos[200:400, :3])
    '''
    newCoPos = np.mean(np.array(dataList[3][0]), axis=0)
    bodyData = bodyData[:] - np.tile(newCoPos, 21)
    #objData = objData[:] - np.concatenate((newCoPos, np.array([.0, .0, .0, .0])))
    markerData = markerData[:] - newCoPos
    if name == "box":
        x,y,z = getNormVecs((0,11), (1,2), (8,0), markerData[0])
    elif name == "tripod":
        x,y,z = getNormVecs((9,11), (7,10), (1,2), markerData[0])
    elif name == "board":
        x,y,z = getNormVecs((2,1), (2,9), (3,9), markerData[0])
    elif name == "budget":
        x,y,z = getNormVecs((9,7), (8,5), (1,0), markerData[0])
    elif name == "chair":
        x,y,z = getNormVecs((0,2), (0,5), (0,1), markerData[0])
    elif name == "table":
        x,y,z = getNormVecs((11,1), (9,11), (9,7), markerData[0])
    else:
        return dataList
    b = np.array([[1.0, .0 , .0],[.0, 1.0, .0],[.0, .0, 1.0]])
    a = np.vstack((x,y,z))
    rot = R.align_vectors(a, b)[0]
    rot = rot.inv()
    '''
    #markerData = rot.apply(markerData)
    #'''
    for i in range(lenFrame):
        #thisBodyData = bodyData[i].reshape((21,3))
        #thisBodyData = rot.apply(thisBodyData)
        #bodyData[i] = thisBodyData.reshape((63))
        #markerData[i] = rot.apply(markerData[i])
        thisFrameRot = R.from_quat(rigidPos[i, 3:])
        #print(thisFrameRot.as_euler("xyz", degrees=True))
        #print(rigidPos[i, :3])
        #print()
        bb[i] = thisFrameRot.apply(bb[i])
        bb[i] = bb[i, :, :] + rigidPos[i, :3]
        #bb = 
    
    #return [bodyData.tolist(), dataList[1], dataList[2], markerData.tolist()]
    #print(a)
    bodyData = bodyData.reshape((lenFrame, 21, 3))
    from mpl_toolkits import mplot3d
    import matplotlib.pyplot as plt

    #fig = plt.figure()
    #ax = plt.axes(projection="3d")
    
    #plt.show()
    fig = plt.figure()
    ax = fig.add_subplot(111 , projection="3d")
    ax.yaxis.set_label_position("top")
    ax.view_init(elev=117., azim=-88.)
    
    fig.show()
    #plt.show()
    for displayFrame in range(startFrame, startFrame+350):
        #'''
        z_points = bodyData[displayFrame, :, 2]
        x_points = bodyData[displayFrame, :, 0]
        y_points = bodyData[displayFrame, :, 1]
        #print(bodyData[displayFrame, :])
        #print()
        for connect in connections:
            a,b = connect
            ax.plot([x_points[a], x_points[b]],[y_points[a],y_points[b]],[z_points[a],z_points[b]], color="r")
        ax.scatter3D(x_points, y_points, z_points, color="r")
        #'''
        
        z_points = markerData[displayFrame, :, 2]
        x_points = markerData[displayFrame, :, 0]
        y_points = markerData[displayFrame, :, 1]
        ax.plot([x_points[6], x_points[5]],[y_points[6],y_points[5]],[z_points[6],z_points[5]], color="r")
        ax.plot([x_points[6], x_points[11]],[y_points[6],y_points[11]],[z_points[6],z_points[11]], color="r")
        ax.scatter3D(x_points, y_points, z_points, color="b")

        z_points = rigidPos[displayFrame, 2]
        x_points = rigidPos[displayFrame, 0]
        y_points = rigidPos[displayFrame, 1]
        ax.scatter3D(x_points, y_points, z_points, color="g")
        ax.plot([0.0, 0.0],[-0.2,0.7,],[0.0,0.0], color="b")
        ax.plot([-0.2, 0.7],[0.0,0.0,],[0.0,0.0], color="r")
        ax.plot([0.0, 0.0],[0.0,0.0,],[-0.2,0.7], color="g")
        z_points = bb[displayFrame,:, 2]
        x_points = bb[displayFrame,:, 0]
        y_points = bb[displayFrame,:, 1]
        ax.plot([x_points[0], x_points[1]],[y_points[0],y_points[1]],[z_points[0],z_points[1]], color="g")
        ax.plot([x_points[1], x_points[5]],[y_points[1],y_points[5]],[z_points[1],z_points[5]], color="g")
        ax.scatter3D(x_points, y_points, z_points, color="black")
        
        fig.canvas.draw()
        print(displayFrame)
        ax.clear()
        #plt.cla()
    #plt.cla()
    #plt.waitKey()
    print("end")
    startFrame+= 350
    #startFrame = 0
    ax = plt.axes(projection="3d")
    ax.yaxis.set_label_position("top")
    ax.view_init(elev=117., azim=-88.)
    #'''
    z_points = bodyData[startFrame, :, 2]
    x_points = bodyData[startFrame, :, 0]
    y_points = bodyData[startFrame, :, 1]
    for connect in connections:
            a,b = connect
            ax.plot([x_points[a], x_points[b]],[y_points[a],y_points[b]],[z_points[a],z_points[b]], color="r")
    ax.scatter3D(x_points, y_points, z_points, color="r")
    #'''
    z_points = markerData[startFrame, :, 2]
    x_points = markerData[startFrame, :, 0]
    y_points = markerData[startFrame, :, 1]
    ax.plot([x_points[6], x_points[5]],[y_points[6],y_points[5]],[z_points[6],z_points[5]], color="r")
    ax.plot([x_points[6], x_points[11]],[y_points[6],y_points[11]],[z_points[6],z_points[11]], color="r")
    ax.scatter3D(x_points, y_points, z_points, color="b")

    z_points = rigidPos[startFrame, 2]
    x_points = rigidPos[startFrame, 0]
    y_points = rigidPos[startFrame, 1]
    
    ax.scatter3D(x_points, y_points, z_points, color="g")
    ax.plot([0.0, 0.0],[-0.2,0.7,],[0.0,0.0], color="b")
    ax.plot([-0.2, 0.7],[0.0,0.0,],[0.0,0.0], color="r")
    ax.plot([0.0, 0.0],[0.0,0.0,],[-0.2,0.7], color="g")
    
    z_points = bb[startFrame, :, 2]
    x_points = bb[startFrame, :, 0]
    y_points = bb[startFrame, :, 1]
    ax.plot([x_points[0], x_points[1]],[y_points[0],y_points[1]],[z_points[0],z_points[1]], color="g")
    ax.plot([x_points[1], x_points[5]],[y_points[1],y_points[5]],[z_points[1],z_points[5]], color="g")
    ax.scatter3D(x_points, y_points, z_points, color="black")
    
    plt.show()

with open(fileName, 'rb') as f:
    dataList = pickle.load(f)[0]
    #dataList = pickle.load(f)
    print(len(dataList))
    startFrame = 20
    '''
    for i in range(len(dataList[0])):
        if dataList[1][i] > 0:
            startFrame = i
            print(startFrame)
            break
    '''
    #relocateData(dataList, objName, startFrame+400)
    relocateData(dataList, objName, startFrame)