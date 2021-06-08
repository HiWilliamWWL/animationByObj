import numpy as np
import pickle
import math
from numpy import linalg as LA
from scipy.spatial.transform import Rotation as R
import glob
import matplotlib.pyplot as plt
from math import sqrt, log
import time
import sys

#fileName = "./part0/wwl_table_drag_r_002.data"
#fileName = "./part1/tina_box2_l1s_b_003.data"
objName = "tripod"

#fileName = "testResult.pb"
testFilePath = "./test1/"
print(testFilePath)
fileName = testFilePath + "testResult_a.pb"
if len(sys.argv) == 3:
    fileName = testFilePath+"testResult_t.pb"
elif len(sys.argv) == 2:
    fileName = testFilePath+"testResult_b.pb"

connections = [(0, 1), (1, 2), (2, 3), (3, 4), 
                             (2, 5), (5, 6), (6, 7), (7, 8), 
                             (2, 9), (9, 10), (10, 11), (11, 12),
                             (0, 13), (13, 14), (14, 15), 
                             (0, 16), (16, 17), (17, 18)]
    

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

def relocateData(dataList, objPoint, startFrame = 0):
    bodyData = np.array(dataList)
    lenFrame = len(dataList)
    
    bodyData = bodyData.reshape((lenFrame, 21, 3))
    from mpl_toolkits import mplot3d
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111 , projection="3d")
    ax.yaxis.set_label_position("top")
    ax.view_init(elev=117., azim=-88.)
    
    
    fig.show()
    #plt.show()
    showRange = [ i for i in range(startFrame, lenFrame)]
    showRange = [i for i in range(50)] + showRange
    restart = 0
    for displayFrame in showRange:
        time.sleep(.01)
        z_points = bodyData[displayFrame, :, 2] * -1.0
        x_points = bodyData[displayFrame, :, 0] 
        y_points = bodyData[displayFrame, :, 1]
        for connect in connections:
            a,b = connect
            ax.plot([x_points[a], x_points[b]],[y_points[a],y_points[b]],[z_points[a],z_points[b]], color="b")
        ax.scatter3D(x_points, y_points, z_points, color="r")

        thisObjPoint = objPoint[1].reshape((12,3))
        z_points = thisObjPoint[ :, 2] * -1.0
        x_points = thisObjPoint[ :, 0] 
        y_points = thisObjPoint[ :, 1]
        ax.scatter3D(x_points, y_points, z_points, color="g")
        #'''
        ax.plot([0.0, 0.0],[-0.7,1.2,],[0.0,0.0], color="b")
        ax.plot([-1.2,1.2],[0.0,0.0,],[0.0,0.0], color="r")
        ax.plot([0.0, 0.0],[0.0,0.0,],[-1.2,1.2], color="g")
        
        
        fig.canvas.draw()
        print(displayFrame)
        ax.clear()
        #plt.cla()
    #plt.cla()
    #plt.waitKey()
    print("end")
    startFrame+= lenFrame - 1
    #startFrame = 0
    ax = plt.axes(projection="3d")
    ax.yaxis.set_label_position("top")
    ax.view_init(elev=117., azim=-88.)
    #'''
    z_points = bodyData[startFrame, :, 2]
    x_points = bodyData[startFrame, :, 0]
    y_points = bodyData[startFrame, :, 1]
    ax.scatter3D(x_points, y_points, z_points, color="r")
    #'''
    
    for connect in connections:
        a,b = connect
        ax.plot([x_points[a], x_points[b]],[y_points[a],y_points[b]],[z_points[a],z_points[b]], color="b")
    ax.plot([0.0, 0.0],[-0.7,1.2,],[0.0,0.0], color="b")
    ax.plot([-1.2,1.2],[0.0,0.0,],[0.0,0.0], color="r")
    ax.plot([0.0, 0.0],[0.0,0.0,],[-1.2,1.2], color="g")
    thisObjPoint = objPoint[1].reshape((12,3))
    z_points = thisObjPoint[ :, 2] * -1.0
    x_points = thisObjPoint[ :, 0] 
    y_points = thisObjPoint[ :, 1]
    ax.scatter3D(x_points, y_points, z_points, color="g")
    plt.show()

with open(fileName, 'rb') as f:
    dataList,obj = pickle.load(f)
    print(obj[:3])
    '''
    for i in range(len(dataList[0])):
        if dataList[1][i] > 0:
            startFrame = i
            print(startFrame)
            break
    '''
    relocateData(dataList, obj, 0)