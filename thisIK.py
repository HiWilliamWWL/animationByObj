import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.optimize import minimize


def solver(targetPos, shoulderPos, elbowPos, handPos):
    '''
        S-----E----H = T
          l1     l2
    '''
    local_Shoulder = np.array([.0, .0, .0])
    local_elbow = np.array(elbowPos) - np.array(shoulderPos)
    local_hand = np.array(handPos) - np.array(elbowPos)
    targetPos = np.array(targetPos) - np.array(shoulderPos)
    def getHandPos(x):
        #r1 = x[:3]
        #r2 = x[3:]
        r1 = R.from_euler('xyz', [x[0], x[1], x[2]], degrees=False)
        r2 = R.from_euler('xyz', [x[3], x[4], x[5]], degrees=False)
        new_elbow = r1.apply(local_elbow)
        new_hand = r2.apply(local_hand)
        v = np.sum(np.square(new_hand + new_elbow - targetPos))
        #print(v)
        return v
    x0 = np.array([-0.01 for i in range(6)])
    #print(minimize.show_options())
    #exit()
    res = minimize(getHandPos, x0, method='BFGS', options={'maxitr': 100, 'disp': False, 'ftpl': 0.0001})
    #res = minimize(getHandPos, x0, method='BFGS', options={'xatol': 1e-5, 'disp': True})
    rs = res.x
    r1 = R.from_euler('xyz', [rs[0], rs[1], rs[2]], degrees=False)
    r2 = R.from_euler('xyz', [rs[3], rs[4], rs[5]], degrees=False)
    new_elbow = r1.apply(local_elbow)
    new_hand = r2.apply(local_hand)
    return shoulderPos + new_elbow, shoulderPos + new_hand + new_elbow

'''
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
s1 = np.array([0.0, 0.0, 0.0])
#s2 = np.array([3.0, 3.0 , 1.0])
#s3 = np.array([4.0, 4.5, 2.0])

s2 = np.array([0.03873201 ,0.18654145, 0.02128091])
s3 = np.array([0.06904891, 0.38034597 ,0.08602412])

target = np.array([0.14329947532997397, 0.8550339624681507, 0.24460446648599232])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

s2, s3 = solver(target, s1, s2, s3)

ax.plot([s1[0], s2[0]], [s1[1], s2[1]],zs=[s1[2], s2[2]])
ax.plot([s2[0], s3[0]], [s2[1], s3[1]],zs=[s2[2], s3[2]])
ax.scatter3D(target[0], target[1], target[2], cmap='Greens')

plt.show()
#Axes3D.plot()
'''