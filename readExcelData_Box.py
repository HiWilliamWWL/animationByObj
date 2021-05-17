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
DISPLAY_FIELDS = False

#objectMarkers = ["Chair:Marker"+str(i) for i in range(1, 9)]
objectMarkers = ["Box:Marker"+str(i) for i in range(1, 13)]

def orderChanger(objPoints):
    #lst idx means new pos, while lst value means original pos
    wantedOrder = [0, 1, 2, 3, 5, 8, 7, 11, 6, 9, 4, 10]
    copyPoints = []
    for idx in wantedOrder:
        copyPoints.append(objPoints[idx])
    return copyPoints

def getPartPos(fileName = './human_chair_basic/s7.csv'): #p5
    print(fileName)
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
        TotalFrame = 0
        for i, row in enumerate(reader):
            if i > GAP:
                for name in validNames:
                    theMap[name].append(row[name])
            TotalFrame += 1
            if TotalFrame > 420:
                break
        bodyPositions = {}
        markerPos = {}
        rigidBody = []
        for count in range(TotalFrame):
            for part in bodyParts:
                theNameX = findName([part, "Skeleton", "Position X"], ["Marker"])
                theNameY = findName([part, "Skeleton", "Position Y"], ["Marker"])
                theNameZ = findName([part, "Skeleton", "Position Z"], ["Marker"])
                thePosition = []
                xValue = theMap[theNameX][count]
                yValue = theMap[theNameY][count]
                zValue = theMap[theNameZ][count]
                if len(xValue) > 0 and xValue[0] in validDigit \
                    and yValue[0] in validDigit and zValue[0] in validDigit: \

                    thePosition = [float(xValue), float(yValue), float(zValue)]
                if len(thePosition) < 3:
                    if count == 0:
                        indexHere = bodyParts.index(part)
                        if indexHere > 0:
                            thePosition = bodyPositions[bodyParts[indexHere-1]][-1]
                        else:
                            thePosition = [.0,.0,.0]
                    else:
                        thePosition = bodyPositions[part][-1]
                if count == 0:
                    bodyPositions[part] = [np.array(thePosition)]
                else:
                    bodyPositions[part].append(np.array(thePosition))
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
            rigidBodyX = findName(["Rigid", "Body Box", "Position X"])
            rigidBodyY = findName(["Rigid", "Body Box", "Position Y"])
            rigidBodyZ = findName(["Rigid", "Body Box", "Position Z"])
            rigidBodyRW = findName(["Rigid", "Body Box", "Rotation W"])
            rigidBodyRX = findName(["Rigid", "Body Box", "Rotation X"])
            rigidBodyRY = findName(["Rigid", "Body Box", "Rotation Y"])
            rigidBodyRZ= findName(["Rigid", "Body Box", "Rotation Z"])
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
                r = R.from_quat([rxValue,ryValue,rzValue,rwValue])
                #rxValue, ryValue, rzValue = r.as_euler("xyz").tolist()
                #print("   {}  {}  {}".format(rxValue, ryValue, rzValue) )
                #thePosition = [float(xValue), float(yValue), float(zValue), float(rxValue), float(ryValue), float(rzValue)]
                thePosition = [float(xValue), float(yValue), float(zValue), float(rxValue), float(ryValue),
                               float(rzValue), float(rwValue)]
            if len(thePosition) < 6:
                if len(rigidBody) == 0:
                    #rigidBody = [.0,.0,.0,.0,.0,.0]
                    rigidBody = [.0, .0, .0, .0, .0, .0, .0]
                else:
                    rigidBody.append(rigidBody[-1])
            else:
                rigidBody.append(thePosition)
        print("next one  ")
        return bodyPositions, markerPos, rigidBody


def draws(allPos, objPos, frame):
    glBegin(GL_LINES)
    frame = int(frame)
    #glEnable(GL_PROGRAM_POINT_SIZE)
    glColor(1.0, 1.0, 1.0)
    for bone in bodyBones:
        j1, j2 = bone.split(" ")
        pos1 = allPos[j1][frame]
        pos2 = allPos[j2][frame]
        if pos1.shape[0] < 1 or pos2.shape[0] < 1:
            continue
        glVertex3fv((pos1[0], pos1[1], pos1[2]))
        glVertex3fv((pos2[0], pos2[1], pos2[2]))
    glEnd()
    glBegin(GL_POINTS)
    glColor(0.0, 1.0, 0.0)
    for part in bodyParts:
        pos = allPos[part][frame]
        if pos.shape[0] < 1:
            continue
        glVertex3fv((pos[0], pos[1], pos[2]))
    glColor(1.0, 0.0, 0.0)
    glEnd()
    biggerCount = 1
    for markerIDX in range(7, 12):
        marker = objectMarkers[markerIDX]
        glPointSize(6 * biggerCount)
        glBegin(GL_POINTS)
        biggerCount += 1
        pos = objPos[marker][frame]
        if pos[0] is None:
            continue
        glVertex3fv((pos[0], pos[1], pos[2]))
        glEnd()
    #glEnd()
    glPointSize(5)
    #sth = input()

def visualizeData(fileName):
    allPos, objPos, rigidBody = getPartPos(fileName)
    pygame.init()
    display = (800,600)
    pygame.display.set_mode(display, DOUBLEBUF|OPENGL)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, (display[0]/display[1]), 0.1, 50.0)
    rx, ry = (0,0)
    tx, ty = (0,0)
    count = 500
    zpos = 5
    rotate = move = False
    glMatrixMode(GL_MODELVIEW)
    count = 0
    #useModel
    thisData1X = []
    startObj = np.array(rigidBody[0][:])
    #print(startObj)
    #exit()
    oriScale = startObj[:3]
    #startObj = (np.array(startObj[:3]) / 0.5).tolist() + (np.array(startObj[3:]) / 3.14159).tolist()
    startObj[3:] = (np.array(startObj[3:]) / 3.14159).tolist() 
    while count < TotalFrame:
        print(count)
        if count%2 == 0:
            thisFrameBody = []
            for part in bodyParts:
                pos = allPos[part][count]
                if pos.shape[0] < 1:
                    pos = [0.0, 0.0, 0.0]
                pos = np.array(pos)
                pos -= np.array(oriScale)
                thisFrameBody += pos.tolist()
            
            thisData1X.append(thisFrameBody + startObj.tolist())
            if len(thisData1X) == 20:
                #useModel(np.array(thisData1X))    #prediction
                thisData1X = thisData1X[1:]
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                pygame.quit()
                quit()
            if e.type == KEYDOWN and e.key == K_w:
                zpos = max(1, zpos-1)
            elif e.type == KEYDOWN and e.key == K_s:
                zpos += 1
            elif e.type == MOUSEBUTTONDOWN:
                if e.button == 1: rotate = True
                elif e.button == 3: move = True
            elif e.type == MOUSEBUTTONUP:
                if e.button == 1: rotate = False
                elif e.button == 3: move = False
            elif e.type == MOUSEMOTION:
                i, j = e.rel
                if rotate:
                    rx += i
                    ry += j
                if move:
                    tx += i
                    ty -= j
        
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
        glEnable(GL_DEPTH_TEST)
        draws(allPos, objPos, count)
        glLoadIdentity()
        

        #glRotatef(1, 3, 1, 1)
        #glTranslate(tx/20., ty/20., - zpos)
        glTranslatef(0.0,0.0, -zpos)
        glRotate(ry, 1, 0, 0)
        glRotate(rx, 0, 1, 0)
        

        pygame.display.flip()
        pygame.time.wait(5)
        if count < 100:
            count += 1

'''
redoList = [[3, 8, 1],[2, 9, 0],[1, 10, 3], [0, 11, 2],
            [7, 8, 5],[6, 9, 4],[5, 10, 7],[4, 11, 6],
            [11, 0, 9],[10, 1, 8],[9, 2, 11],[8, 3, 10]]  #table
'''
redoList = [[None, None, None],[2, 5, 3],[1, 6, 4], [4, 7, 1], [3, 8, 2],
            [6, 10, 7],[5, 11, 8],[8, 3, 5],[7, 4, 6],
            [None, 0, None], [11, 5, None],[10, 6, None]]  #chair
def getTrainData(fileName):
    global TotalFrame
    #nickName = fileName[-6:-4]
    allPos, objPos, rigidBody= getPartPos(fileName)
    count = 0
    bodyData = []
    ifTouchObj = []
    objData = []
    totalMarker = []
    startObj = np.array(rigidBody[0][:])
    startTouch = False
    while count < TotalFrame:
        try:
            center = np.array(allPos["Hip"][count])
            #center = np.array(objPos["Chair:Marker2"][count])
            if center.shape[0] < 1:
                TotalFrame += 1
                count += 1
                continue
                #center = [0.0, 0.0, 0.0]
            thisFrameBody = []
            for part in bodyParts:
                pos = allPos[part][count]
                if pos.shape[0] < 1:
                    pos = center
                pos = np.array(pos)
                pos -= center
                thisFrameBody += pos.tolist()
            bodyData.append(thisFrameBody)
            if count > 0 and not startTouch:
                thisObj = np.array(rigidBody[count][:])
                if thisObj.shape[0] > 1:
                    if np.sum(np.abs(thisObj - startObj)) > 0.1:
                        startTouch = True
            if count > 0 and startTouch:
                ifTouchObj.append(1.0)
            else:
                ifTouchObj.append(0.0)
            objData.append(rigidBody[count][:])

            markerData = [[] for n in range(12)]
            def getMarkerPos(num):
                #key = "Chair:Marker"+str(num)
                key = objectMarkers[num]
                pos = np.array(objPos[key][count]) - center
                if pos[0] is None:
                    pos = None
                return pos
            origin = np.array(rigidBody[count][:3])

            #for full Chair
            markerData[0] = getMarkerPos(0)
            markerData[1] = getMarkerPos(1)
            markerData[2] = getMarkerPos(2)
            markerData[3] = getMarkerPos(3)
            markerData[4] = getMarkerPos(5)
            markerData[5] = getMarkerPos(8)
            markerData[6] = getMarkerPos(7)
            markerData[7] = getMarkerPos(11)
            markerData[8] = getMarkerPos(6)
            markerData[9] = getMarkerPos(9)
            markerData[10] = getMarkerPos(4)
            markerData[11] = getMarkerPos(10)

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
            count += 1
        except IndexError:
            break
    return [bodyData, ifTouchObj, objData, totalMarker]


if __name__ == "__main__":
    __name__ = "not"
    #visualizeData("./completeData/wjp_box/1102BoxL1.csv")
    #exit()

    #./human_chair_basic/*.csv
    files = glob.glob("./completeData/wjp_box/*.csv")
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
    #with open('./dataReady/wjpChair3.data', 'wb') as fp:
    with open('./dataReady/wjbBox.data', 'wb') as fp:
        pickle.dump(wholeDataset, fp)