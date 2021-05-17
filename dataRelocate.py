import csv
import numpy as np
import pygame
from pygame.locals import *
import pickle
from OpenGL.GL import *
from OpenGL.GLU import *
import glob
from random import shuffle
from scipy.spatial.transform import Rotation as R
#from train3D_new2 import useModel


class DataManager():

    def __init__(self, fileName, objName):
        self.dataList = []
        self.objList = []
        self.trainX = []
        self.trainY = []
        self.markerX = []
        print("We are collecting from   " + fileName)
        with open(fileName, 'rb') as f:
            self.dataList = pickle.load(f)[0]
        with open(objName, 'rb') as f:
            self.objList = np.array(pickle.load(f))   #12*5*2*3   Points*Directions*2*3   12*5*4*3
        #AllData.append([selectedPRecod, operate, curFrame])
        #AllData:    [bodyData, ifTouchObj, objData, totalMarker]:  [[x,y,z], ...], [0,0,1,1,...], [[x,y,z,rx,ry,rz,rw], ...] [[x,y,z], ...]
        
        step = sampleLength // 20 #no of frames for each step, step is for getting key-frame
        prediectStep = 10 #no of steps
        videoLength = len(self.dataList[0])
        print("video length:  "+str(videoLength))

if __name__ == "__main__":
    files = glob.glob("../rawData/dataReady/gen_action/part3/*.data")
    #files = ["../rawData/dataReady/whole/part1/wjp_cone_w_002.data"]
    objNames = ["board", "box", "budget", "chair", "table", "tripod", "cone"]
    #dm = DataManager()
    fileCount = 0
    for fileName in files:
        print(fileName)
        for objName in objNames:
            if objName in fileName:
                objFileName = "../physicalSimulatorSimple/affordance/obj_" + objName + ".data"
                break
         dm = DataManager(fileName, objFileName)

    #./human_chair_basic/*.csv
    #files = glob.glob("./processedData/test/*.csv")
    files = glob.glob("./processedData/*.csv")
    #files = glob.glob("./processedData/test/*l1s*.csv")
    #files = glob.glob("./processedData/chair_test/*.csv")
    #files += glob.glob("./completeData/wjp_chair/t*.csv")
    wholeDataset = []
    #shuffle(files)
    for fileName in files:
        print(fileName)
        inThisFile = getTrainData(fileName)
        wholeDataset.append(inThisFile)
        print(len(inThisFile[0]))
        print(len(inThisFile[1]))
        print(len(inThisFile[2]))
        print(len(inThisFile[3]))
        print()
        with open('./dataReady/' + fileName.split('/')[-1][:-4] + '.data', 'wb') as fp:
            pickle.dump([inThisFile], fp)
    #with open('./dataReady/wjpChair3.data', 'wb') as fp:
    exit()
    with open('./dataReady/all/all.data', 'wb') as fp:
        pickle.dump(wholeDataset, fp)