import numpy as np
from scipy.spatial.transform import Rotation as Rot
from numpy import linalg as LA
import skeletonHandle
import glob
import pickle

startEnd = {"t1":[360], "t2":[320], "t3":[350], "t4":[260], "t5":[230], "t6":[270],
            "n1":[270], "n2":[290], "n3":[300], "n4":[290], "n5":[210], "n6":[270],
            "n7":[280], "n8":[280], "n9":[300], "n10":[115], "n11":[230], "n12":[230],
            "n13":[200], "n14":[180], "n15":[190], "n16":[160], "n17":[240], "n18":[270],
            "n19":[260], "n20":[280], "n21":[260], "n22":[240], "n23":[280]}

humanDimension = 63
objDimention = 36

joints_connection = [(0, 1), (1, 2), (2, 3), (3, 4), 
                                    (2, 5), (5, 6), (6, 7), (7, 8), 
                                    (2, 9), (9, 10), (10, 11), (11, 12),
                                    (0, 13), (13, 14), (14, 15), (15, 19), 
                                    (0, 16), (16, 17), (17, 18), (18, 20)]

joints_Tpose_norm = [[.0, 1.0, .0], [.0, 1.0, .0], [.0, 1.0, .0], [.0, 1.0, .0], 
                                    [0.1736482, 0.984808, .0], [1.0, .0, .0], [1.0, .0, .0], [1.0, .0, .0], 
                                    [-0.1736482, 0.984808, .0], [-1.0, .0, .0], [-1.0, .0, .0], [-1.0, .0, .0],
                                    [1.0, .0, .0], [.0, -1.0, .0], [.0, -1.0, .0], [.0, .0, 1.0], 
                                    [-1.0, .0, .0], [.0, -1.0, .0], [.0, -1.0, .0], [.0, .0, 1.0]]


def loadFromFile(pathName = "./Data/liftOnly/*.data", getFrame = 0):
        files = glob.glob(pathName)
        print("loading files from")
        print(pathName)
        for f in files[:1]:
            print(f)
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
                midPointofHip = (np.array(dataList[0][start][13*3:13*3+3]) + np.array(dataList[0][start][16*3:16*3+3])) / 2.0
                dataList[0][start][:3] = midPointofHip.tolist()
                worldCenter = np.array(dataList[0][start][:3])
                #[1,0,0], worldCenter->boxCenter
                wc_bc = np.array(dataList[2][start][:3]) - worldCenter
                wc_bc[1] = 0.0
                wc_bc = np.array([wc_bc, [0,1,0]])
                adj_rot = Rot.align_vectors(np.array([[1,0,0], [0,1,0]]), wc_bc)[0]
                bodyDataT = []
                bodyDataR = []

                #for i in range(start, end, step): 190
                for i in range(start, end, step):  #do once only
                    try:
                        midPointofHip = (np.array(dataList[0][i][13*3:13*3+3]) + np.array(dataList[0][i][16*3:16*3+3])) / 2.0
                        dataList[0][i][:3] = midPointofHip.tolist()
                        bodyCenter = adj_rot.apply(np.array(dataList[0][i][:3]) - worldCenter)
                        bodyWhole = adj_rot.apply(np.array(dataList[0][i][:]).reshape((21,3))- worldCenter)
                        #bodyWhole[:, :] -= bodyCenter
                    except IndexError:
                        print(f)
                        print("the length is "+ str(i))
                        print()
                        break
                    print(bodyWhole.shape) #(21,3)
                    '''
                    bone_length = []
                    for connection in joints_connection:
                        s,e = connection
                        diffHere = bodyWhole[e] - bodyWhole[s]
                        diffHere = np.linalg.norm(diffHere)
                        bone_length.append(diffHere)
                    bone_length = np.array(bone_length)
                    '''
                    JI = skeletonHandle.JointsInfo(bodyWhole)
                    allRots = JI.get_parent_17_rots()
                    #exit()
                    print("MMMMMMMMMMMMMMMMMMM_start_MMMMMMMMMMMMMMMMMMMMMM")
                    bodyWhole_T = JI.generate_standard_Tpose_from_this_bone_length()
                    '''
                    bodyWhole_T = np.array([[.0, .0, .0] for k in range(21) ])
                    for k, connection in enumerate(joints_connection):
                        s,e = connection
                        bodyWhole_T[e] = bodyWhole_T[s] + np.array(joints_Tpose_norm[k]) * bone_length[k]
                    '''

                    #bodyWhole_T = np.array(joints_Tpose_pose)
                    #rot_T = Rot.from_euler("xyz", [.0, 87.0, .0], degrees=True)
                    #bodyWhole_T = rot_T.apply(bodyWhole_T)
                    JI2 = skeletonHandle.jointsInfo(bodyWhole_T, bodyCenter)
                    JI2.forward_kinematics_parentsOnly_17Joints(allRots)
                    bodyWhole_Result = JI2.get_all_poses()
                    
                    
                    #bodyWhole  bodyWhole_T    bodyWhole_Result
                    bodyWhole = bodyWhole.reshape((humanDimension))
                    #bodyWhole = bodyWhole_Result.reshape((humanDimension))
                    bodyData.append(bodyWhole)

                    bodyWhole_T = bodyWhole_T.reshape((humanDimension))
                    bodyDataT.append(bodyWhole_T)

                    bodyWhole_Result = bodyWhole_Result.reshape((humanDimension))
                    bodyDataR.append(bodyWhole_Result)
                    
                    objPos = adj_rot.apply(np.array(dataList[2][i][:3]) - worldCenter)
                    objPos = objPos.tolist()
                    objRot = Rot.from_quat(dataList[2][i][3:])
                    objRot = objRot.as_euler("xyz").tolist()
                    rigidPos.append(np.array(objPos+objRot))

                    objMarkers = adj_rot.apply(np.array(dataList[3][i]) - worldCenter)
                    markerPos.append(objMarkers.reshape((12*3)))


                with open("./Tests/skeletonTest/testResult_o.pb", 'wb') as pickle_file:
                    pickle.dump([np.array(bodyData).reshape((-1, 21*3)), np.array(markerPos).reshape((-1, 12*3))], pickle_file)
                with open("./Tests/skeletonTest/testResult_t.pb", 'wb') as pickle_file:
                    pickle.dump([np.array(bodyDataT).reshape((-1, 21*3)), np.array(markerPos).reshape((-1, 12*3))], pickle_file)
                with open("./Tests/skeletonTest/testResult_r.pb", 'wb') as pickle_file:
                    pickle.dump([np.array(bodyDataR).reshape((-1, 21*3)), np.array(markerPos).reshape((-1, 12*3))], pickle_file)
                    
                print("saved data and ready for draw")

loadFromFile()