import numpy as np
import pickle

from matplotlib import animation
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import sys
import time

fig = plt.figure()
ax = fig.add_subplot(111 , projection="3d")

objName = "tripod"

testFilePath = "./Tests/test_wgn1/"


print(testFilePath)
fileName = testFilePath + "testResult_a.pb"
print(sys.argv)
option = sys.argv[1]
if option == 't':
    fileName = testFilePath+"testResult_t.pb"
elif option == 'g':
    fileName = testFilePath+"testResult_g.pb"
elif option == 'c':
    fileName = testFilePath+"testResult_c.pb"
elif option == 'r':
    fileName = testFilePath+"testResult_r.pb"
elif option == 'o':
    fileName = testFilePath+"testResult_o.pb"
elif option == 'p':
    fileName = testFilePath+"testResult_p.pb"

connections = [(0, 1), (1, 2), (2, 3), (3, 4), 
                             (2, 5), (5, 6), (6, 7), (7, 8), 
                             (2, 9), (9, 10), (10, 11), (11, 12),
                             (0, 13), (13, 14), (14, 15), 
                             (0, 16), (16, 17), (17, 18)]
    
'''
def update(frame,*fargs):
    #print(frame)
    #print(fargs)
    data, scat = fargs
    print("calling update")
    ax.clear()
    ax.scatter3D(data[frame,0,:], data[frame,1,:], data[frame,2,:], color="r")
    return ax
'''

def updateHumanObj(frame, *fargs):
    ax.clear()
    bodyData, objPoint, scat = fargs
    print(frame)
    print(bodyData[frame, :])
    print("_______________")
    z_points = bodyData[frame, :, 2] #* -1.0
    x_points = bodyData[frame, :, 0]
    y_points = bodyData[frame, :, 1]
    #print(x_points)
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
    


def relocateData(dataList, objPoint, startFrame = 0):
    global fig, ax
    bodyData = np.array(dataList)
    lenFrame = len(dataList)
    
    bodyData = bodyData.reshape((lenFrame, 21, 3))
    ax.yaxis.set_label_position("top")
    ax.view_init(elev=117., azim=-88.)
    scat = ax.scatter(bodyData[0,:,0], bodyData[0,:,1], bodyData[0,:,2], c='r', marker = 'o',alpha=0.5, s=100)
    time.sleep(.1)

    ani = animation.FuncAnimation(fig, updateHumanObj, frames= range(lenFrame), interval = 50, repeat_delay=100,
                                  fargs=(bodyData, objPoint, scat))
    plt.show()


with open(fileName, 'rb') as f:
    dataList,obj = pickle.load(f)
    #print(obj.shape)
    #exit()
    '''
    for i in range(len(dataList[0])):
        if dataList[1][i] > 0:
            startFrame = i
            print(startFrame)
            break
    '''
    relocateData(dataList, obj, 0)