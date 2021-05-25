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

bodyParts = ["Hip", "Ab", "Chest", "Neck", "Head", \
            "LShoulder", "LUArm", "LFArm", "LHand", \
            "RShoulder", "RUArm", "RFArm", "RHand", \
            "LThigh", "LShin", "LFoot", \
            "RThigh", "RShin", "RFoot",\
            "LToe", "RToe"]\

bodyBones = ["Hip Ab", "Ab Chest", "Chest Neck", "Neck Head", \
            "Neck LShoulder", "Neck RShoulder",  \
            "LShoulder LUArm", "LUArm LFArm", "LFArm LHand", \
            "RShoulder RUArm", "RUArm RFArm", "RFArm RHand",\
            "Hip LThigh", "LThigh LShin", "LShin LFoot", \
            "Hip RThigh", "RThigh RShin", "RShin RFoot", \
            "LFoot LToe", "RFoot RToe"]\

validDigit = "-0987654321"

#All Controls
TotalFrame = 0
#GAP = 4
GAP = -1
DISPLAY_FIELDS = False  #modify Line 141 as well
#for display the first 100 frams, just un-comment the draw-call in main()

#objectMarkers = ["Chair:Marker"+str(i) for i in range(1, 9)]
objectMarkers = []
objname = ["chair2", "chair3", "chair4", "chair", "box2", "box3", "box4","box", "board", "tripod", "table2", "table1", "budget", "cone"]
thisObj = ""
env_file = ""
global_start_obj = None

def orderChanger(objPoints):
    #lst idx means new pos, while lst value means original pos
    wantedOrder = [0, 1, 2, 3, 5, 8, 7, 11, 6, 9, 4, 10]
    copyPoints = []
    for idx in wantedOrder:
        copyPoints.append(objPoints[idx])
    return copyPoints

def getPartPos(fileName): #p5
    print(fileName)
    global objname, thisObj, objectMarkers, global_start_obj
    with open(fileName) as csvfile:
        reader = csv.DictReader(csvfile)
        theKeys = []
        validNames = [name for name in reader.fieldnames if len(name) > 0  and name[0] not in validDigit]
        theMap = {name:[] for name in validNames}
        if DISPLAY_FIELDS:
            for name in validNames:
                print(name)
            exit()

        def findName(listOfHints, excludeList = []):
            result = []
            for name in validNames:
                goodChoice = True
                for hint in listOfHints:
                    if hint not in name:
                        goodChoice = False
                        break
                for exclude in excludeList:
                    if exclude in name:
                        goodChoice = False
                        break
                if goodChoice:
                    result.append(name)
            if len(result) == 0:
                return None
            elif len(result) == 1:
                return result[0]
            for rest in result:
                if "Marker10" not in rest and "Marker11" not in rest and "Marker12" not in rest:
                    return rest
            return result
        global TotalFrame
        
        for each_name in objname:
            if findName([each_name]) is not None:
                thisObj = each_name
                break
        print(thisObj)
        #exit()
        TotalFrame = 0
        for i, row in enumerate(reader):
            if i > GAP:
                for name in validNames:
                    theMap[name].append(row[name])
            TotalFrame += 1
        bodyPositions = {}
        markerPos = {}
        rigidBody = []
        for count in range(TotalFrame):
            for part in bodyParts:
                theNameX = findName([part, "Skeleton 002", "Position X"], ["Marker"])
                theNameY = findName([part, "Skeleton 002", "Position Y"], ["Marker"])
                theNameZ = findName([part, "Skeleton 002", "Position Z"], ["Marker"])
                thePosition = []
                xValue = theMap[theNameX][count]
                yValue = theMap[theNameY][count]
                zValue = theMap[theNameZ][count]
                if len(xValue) > 0 and xValue[0] in validDigit \
                    and yValue[0] in validDigit and zValue[0] in validDigit: \

                    thePosition = [float(xValue), float(yValue), float(zValue)]
                if len(thePosition) < 3:
                    if count > 0:
                        indexHere = bodyParts.index(part)
                        if indexHere > 0 and indexHere < 20:
                            add1 = np.array(bodyPositions[bodyParts[indexHere-1]][-1])
                            add2 = np.array(bodyPositions[bodyParts[indexHere+1]][-1])
                            thePosition = (0.5 * (add1 + add2)).tolist()
                        elif indexHere > 0:
                            thePosition = bodyPositions[bodyParts[indexHere-1]][-1]
                        elif indexHere < 20:
                            thePosition = bodyPositions[bodyParts[indexHere+1]][-1]
                        else:
                            thePosition = [.0,.0,.0]
                    else:
                        indexHere = bodyParts.index(part)
                        thePosition = bodyPositions[bodyParts[indexHere-1]][-1]
                    
                        
                    '''
                    else:
                        thePosition = bodyPositions[part][-1]
                    '''
                if count == 0:
                    bodyPositions[part] = [np.array(thePosition)]
                else:
                    bodyPositions[part].append(np.array(thePosition))
            objectMarkers = [thisObj + ":Marker"+str(i) for i in range(1, 13)]
            for marker in objectMarkers:
                #for chair wwl
                theNameX = findName([marker, "Rigid", "Position X"])
                theNameY = findName([marker,  "Rigid", "Position Y"])
                theNameZ = findName([marker,  "Rigid", "Position Z"])
                thePosition = []
                xValue = theMap[theNameX][count]
                yValue = theMap[theNameY][count]
                zValue = theMap[theNameZ][count]
                if len(xValue) > 0 and xValue[0] in validDigit \
                    and yValue[0] in validDigit and zValue[0] in validDigit: \

                    thePosition = [float(xValue), float(yValue), float(zValue)]
                if len(thePosition) < 3:
                    if count == 0:
                        thePosition = [None,None,None]
                    else:
                        thePosition = markerPos[marker][-1]
                if count == 0:
                    markerPos[marker] = [np.array(thePosition)]
                else:
                    markerPos[marker].append(np.array(thePosition))
            rigidBodyX = findName(["Rigid", "Body "+thisObj, "Position X"])
            rigidBodyY = findName(["Rigid", "Body "+thisObj, "Position Y"])
            rigidBodyZ = findName(["Rigid", "Body "+thisObj, "Position Z"])
            rigidBodyRW = findName(["Rigid", "Body "+thisObj, "Rotation W"])
            rigidBodyRX = findName(["Rigid", "Body "+thisObj, "Rotation X"])
            rigidBodyRY = findName(["Rigid", "Body "+thisObj, "Rotation Y"])
            rigidBodyRZ= findName(["Rigid", "Body "+thisObj, "Rotation Z"])
            thePosition = []  #We should keep rotat as XYZW
            xValue = theMap[rigidBodyX][count]
            yValue = theMap[rigidBodyY][count]
            zValue = theMap[rigidBodyZ][count]
            rxValue = theMap[rigidBodyRX][count]
            ryValue = theMap[rigidBodyRY][count]
            rzValue = theMap[rigidBodyRZ][count]
            rwValue = theMap[rigidBodyRW][count]
            #print("Hi  {}   {}    {}   {}".format(rxValue, ryValue, rzValue, rwValue))
            if len(xValue) > 0 and xValue[0] in validDigit \
                and yValue[0] in validDigit and zValue[0] in validDigit and rwValue[0] in validDigit \
                and rxValue[0] in validDigit and ryValue[0] in validDigit and rzValue[0] in validDigit:
                '''
                if count > 0:
                    r_current = R.from_quat([rxValue,ryValue,rzValue,rwValue])
                    r_result = r_current * (r_start.inv())
                    rxValue, ryValue, rzValue,rwValue = r_result.as_quat().tolist()
                else:
                    rxValue, ryValue, rzValue,rwValue = [0.0, 0.0, 0.0, 1.0]
                '''
                #rxValue, ryValue, rzValue = r.as_euler("xyz").tolist()
                #print("   {}  {}  {}".format(rxValue, ryValue, rzValue) )
                #thePosition = [float(xValue), float(yValue), float(zValue), float(rxValue), float(ryValue), float(rzValue)]
                if count == 0 or global_start_obj is None:
                    global_start_obj = [float(xValue), float(yValue), float(zValue), float(rxValue), float(ryValue),
                               float(rzValue), float(rwValue)]
                    thePosition = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
                else:
                    thePosition = np.array([float(xValue), float(yValue), float(zValue), float(rxValue), float(ryValue),
                               float(rzValue), float(rwValue)])
                    transThis = thePosition[:3] - global_start_obj[:3]
                    rotThis = R.from_quat([float(rxValue), float(ryValue), float(rzValue), float(rwValue)])
                    rotOrigin = R.from_quat([global_start_obj[3], global_start_obj[4], global_start_obj[5], global_start_obj[6]])
                    rotDif = rotThis * (rotOrigin.inv())
                    thePosition[:3] = rotDif.inv().apply(transThis)
                    thePosition[3:] = rotDif.as_quat()
                    #thePosition = thePosition.tolist()
            '''
            if len(thePosition) < 6:
                if len(rigidBody) == 0:
                    #rigidBody = [.0,.0,.0,.0,.0,.0]
                    rigidBody = [.0, .0, .0, .0, .0, .0, .0]
                else:
                    rigidBody.append(rigidBody[-1])
            else:
            '''
            rigidBody.append(thePosition)
        print("next one  ")
        return bodyPositions, markerPos, rigidBody

'''
redoList = [[3, 8, 1],[2, 9, 0],[1, 10, 3], [0, 11, 2],
            [7, 8, 5],[6, 9, 4],[5, 10, 7],[4, 11, 6],
            [11, 0, 9],[10, 1, 8],[9, 2, 11],[8, 3, 10]]  #table
'''
redoList = [[None, None, None],[2, 5, 3],[1, 6, 4], [4, 7, 1], [3, 8, 2],
            [6, 10, 7],[5, 11, 8],[8, 3, 5],[7, 4, 6],
            [None, 0, None], [11, 5, None],[10, 6, None]]  #chair
def getTrainData(fileName):
    global TotalFrame, thisObj, env_file, global_start_obj
    #nickName = fileName[-6:-4]
    global_start_obj = None
    allPos, objPos, rigidBody= getPartPos(fileName)
    count = 0
    bodyData = []
    ifTouchObj = []
    objData = []
    totalMarker = []
    startObj = np.array(rigidBody[0][:])
    startTouch = False
    env_file = open("./dataPlace/dataReady/env/env_" + fileName.split('/')[-1][:-4] + ".txt","w")
    env_file.writelines("{}\n".format(TotalFrame))
    centerAtFrame = 0
    firstFrameRepeat = 0
    tripodLink = {}
    tripodLink["table"] = [(1,3), (1,5), (4,5), (3,4), (2,11), (6,9), (7,8), (10,12)]
    tripodLink["box"] = [(1,3), (1,2), (2,12), (3,12), (4,5), (5,11), (9,11), (4,9), (1,9), (3,4), (12,5), (2,11)]
    tripodLink["chair"] = [(1,6), (4,6), (3,4), (1,3), (6,8), (4,8), (1,8), (3,8), (1,2), (3,5), (1,7), (3,7), (5,7), (2,7), (2,5), (2,9), (1,11), (5, 10), (3,12)]
    tripodLink["board"] = [(1, 12), (11, 12), (9, 11), (3, 9), (3, 5), (5, 2), (2, 8), (8, 1), (3, 10), (10, 4), (2, 6), (6, 7)]
    tripodLink["budget"] = [(1,3), (1,5), (3,4), (4,5), (1,6), (3,6), (1,8), (5,8), (4,9), (5,9), (3,10), (4,10), (6,12), (6,2), (8,2), (8,7), (9,11), (9,7), (10,12), (10,11), (2,7), (2,12), (11,12), (7,11)]
    tripodLink["tripod"] = [(10,11), (11,12), (8,11), (8,9), (3,9), (5,7), (4,6), (1,2)]
    tripodLink["cone"] = [(1,2), (2,3), (3,12), (1,12), (1,4), (2,9), (12,8), (3,11), (9,6), (4,5), (8,7), (10,11), (5,6), (5,7), (7,10), (6,10)]
    obj_type_name = thisObj
    if obj_type_name[-1].isdigit():
        obj_type_name = obj_type_name[:-1]

    theLink = tripodLink[obj_type_name]
    for i in range(len(theLink)):
        env_file.write(str(theLink[i]))
        env_file.write(";")
    env_file.write("\n")
    while count < TotalFrame:
        try:
            #center = np.array(allPos["Hip"][centerAtFrame])
            center = np.array(global_start_obj[:3])
            center_rot = R.from_quat([global_start_obj[3], global_start_obj[4], global_start_obj[5], global_start_obj[6]])
            #center = np.array(objPos["Chair:Marker2"][count])
            '''
            if np.sum(center) < 0.00001:
                #TotalFrame += 1
                count += 1
                centerAtFrame += 1
                continue
                #center = [0.0, 0.0, 0.0]
            '''
            thisFrameBody = []
            for part in bodyParts:
                pos = allPos[part][count]
                if pos.shape[0] < 1:
                    pos = center
                pos = np.array(pos)
                pos -= center
                pos = center_rot.inv().apply(pos)
                thisFrameBody += pos.tolist()
            bodyData.append(thisFrameBody)
            if count > 0 and not startTouch:
                thisObjPos = np.array(rigidBody[count][:])
                if thisObjPos.shape[0] > 1:
                    if np.sum(np.abs(thisObjPos - startObj)) > 0.1:
                        startTouch = True
            if count > 0 and startTouch:
                ifTouchObj.append(1.0)
            else:
                ifTouchObj.append(0.0)
            current6D = np.array(rigidBody[count][:])
            if (current6D.shape[0] == 7):
                objData.append(current6D.tolist())
            else:
                objData.append(objData[-1])
            #result6D = np.concatenate(((current6D[:3] - center), current6D[3:]))
            

            objRigidPos = rigidBody[count][:3]
            objRigidOri = np.array(rigidBody[count][3:])

            markerData = [[] for n in range(12)]
            def getMarkerPos(num):
                global objectMarkers
                #key = "Chair:Marker"+str(num)
                key = objectMarkers[num]
                pos = np.array(objPos[key][count]) - center
                pos = center_rot.inv().apply(pos)
                if pos[0] is None:
                    pos = None
                return pos
            origin = np.array(rigidBody[count][:3])

            #index(0-base) - value(1-base)
            changeOrder = [1,2,3,4,5,6,7,8,9,10,11,12]
            #box1-box2
            if thisObj is "box2":
                changeOrder = [12, 11, 1, 2, 4, 7, 5, 10, 9, 6, 8, 3]
            #box1-box3
            elif thisObj is "box3":
                changeOrder = [12, 11, 4, 1, 8, 3, 2, 7, 5, 10, 6, 9]
            #chair1-chair2
            elif thisObj is "chair2":
                changeOrder = [6, 8, 12, 1, 5, 4, 11, 9, 7, 3, 2, 10]

            #chair1-chair3
            elif thisObj is "chair3":
                changeOrder = [4, 7, 10, 2, 9, 1, 8, 5, 6, 11, 3, 12]
            #chair1-chair2
            elif thisObj is "chair4":
                changeOrder = [6, 4, 10, 11, 1, 12, 5, 7, 8, 2, 9, 3]
            #table1-table2
            elif thisObj is "table2":
                changeOrder = [10, 12, 11, 4, 6, 7, 5, 1, 9, 2, 8, 3]
            
            for markerCount in range(12):
                markerData[markerCount] = getMarkerPos(changeOrder[markerCount] - 1)

            for mc in range(12):
                if markerData[mc] is None:
                    backups = (redoList[mc][0], redoList[mc][2], redoList[mc][1])
                    for theDir,b in enumerate(backups):
                        if b is not None and markerData[b] is not None:
                            markerData[mc] = markerData[b]
                            theDiff = markerData[b][theDir] - origin[theDir]
                            markerData[mc][theDir] = origin[theDir] - theDiff
                            break
                    if markerData[mc] is None:
                        markerData[mc] = origin
            markerData = np.array(markerData)
            totalMarker.append(markerData.tolist())
            if firstFrameRepeat > 0:
                firstFrameRepeat -= 1
            else:
                count += 1
            
            env_file.writelines("Frame {}\n".format(count))
            for countPrint in range(len(objRigidPos)):
                env_file.write(str(objRigidPos[countPrint]))
                if countPrint <= 1:
                    env_file.writelines(",")
                else:
                    env_file.writelines(";")
            for countPrint in range(objRigidOri.shape[0]):
                env_file.write(str(objRigidOri[countPrint]))
                if countPrint <= 2:
                    env_file.write(",")
                else:
                    env_file.write("\n")
            markerDataLst = markerData.tolist()
            for countPrint in range(0, len(markerDataLst), 3):
                env_file.writelines("{},{},{};".format(markerDataLst[countPrint], markerDataLst[countPrint + 1], markerDataLst[countPrint + 2]))
            for countPrint in range(0, len(thisFrameBody), 3):
                env_file.writelines("{},{},{};".format(thisFrameBody[countPrint], thisFrameBody[countPrint + 1], thisFrameBody[countPrint + 2]))
            env_file.write("\n")
            
        except IndexError:
            break
    env_file.close()
    return [bodyData, ifTouchObj, objData, totalMarker]


if __name__ == "__main__":
    #__name__ = "not"
    #visualizeData("./completeData/wjp_bucket/1109BucketL6.csv")
    #exit()

    #./human_chair_basic/*.csv
    #files = glob.glob("./processedData/test/*.csv")
    
    files = glob.glob("./dataPlace/process/*.csv")
    #files = glob.glob("./dataPlace/process/May7Task_003.csv")
    
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
        with open('./dataPlace/dataReady/' + fileName.split('/')[-1][:-4] + '.data', 'wb') as fp:
            pickle.dump([inThisFile], fp)
    #with open('./dataReady/wjpChair3.data', 'wb') as fp:
    exit()
    with open('./dataPlace/dataReady/all/all.data', 'wb') as fp:
        pickle.dump(wholeDataset, fp)