import os
import pybullet as p
import pybullet_data
import math
import numpy as np
import time

p.connect(p.GUI)
p.resetSimulation()
p.setRealTimeSimulation(1)
p.setGravity(0,0,-9.8)
p.setPhysicsEngineParameter(numSolverIterations=200)
p.setPhysicsEngineParameter(solverResidualThreshold=1e-30)
        
# get current path
urdfRootPath=pybullet_data.getDataPath()
# load plane URDF
planeUid = p.loadURDF(os.path.join(urdfRootPath,"plane.urdf"), basePosition=[0,0,0])
dofbot_path = os.path.join(os.path.dirname(__file__), 'arm.urdf')
baseOrn = p.getQuaternionFromEuler([0, -math.pi, 0])
armUid = p.loadURDF(dofbot_path, basePosition=[0,0,0.4], baseOrientation = baseOrn, useFixedBase=True)

# change the appearance of DOFBOT parts
p.changeVisualShape(armUid, -1, rgbaColor=[0,0,0,1])
p.changeVisualShape(armUid, 0, rgbaColor=[0,1,0,1])
p.changeVisualShape(armUid, 1, rgbaColor=[1,1,0,1])
p.changeVisualShape(armUid, 2, rgbaColor=[0,1,0,1])
p.changeVisualShape(armUid, 3, rgbaColor=[1,1,0,1])
p.changeVisualShape(armUid, 4, rgbaColor=[0,0,0,1])
p.changeVisualShape(armUid, 5, rgbaColor=[0,1,0,0])
p.changeVisualShape(armUid, 6, rgbaColor=[0,0,1,0])
p.changeVisualShape(armUid, 7, rgbaColor=[1,0,0,0.5])
# reset pose of all DOFBOT joints
rest_poses_dofbot = [0, 0, 0, 0, 0] # stay upright

x_in = p.addUserDebugParameter("X", -0.5, 0.5, 0)
y_in = p.addUserDebugParameter("Y", -0.5, 0.5, 0.2)
z_in = p.addUserDebugParameter("Z", -1.57, 1.57, 0.25)
rollId = p.addUserDebugParameter("roll", -3.14, 3.14, 0)
pitchId = p.addUserDebugParameter("pitch", -3.14, 3.14, np.pi)
yawId = p.addUserDebugParameter("yaw", -np.pi/2, np.pi/2, np.pi/2)
numJoints = p.getNumJoints(armUid)
stepCounter = 0
while(True):
    targetX = p.readUserDebugParameter(x_in)
    targetY = p.readUserDebugParameter(y_in)
    targetZ = p.readUserDebugParameter(z_in)
    roll = p.readUserDebugParameter(rollId)
    pitch = p.readUserDebugParameter(pitchId)
    yaw = p.readUserDebugParameter(yawId)
    orn = p.getQuaternionFromEuler([roll, pitch, yaw])
    jointPoses = p.calculateInverseKinematics(armUid, 4, [targetX, targetY, targetZ], orn, maxNumIterations=1000,
                                                  residualThreshold=.01)
    for i in range(5):
        p.setJointMotorControl2(armUid, i, p.POSITION_CONTROL, targetPosition = jointPoses[i])
    if(stepCounter % 1000 == 0):
        print("X error: ", round(p.getLinkState(armUid, 4)[0][0]-targetX, 5))
        print("Y error: ", round(p.getLinkState(armUid, 4)[0][1]-targetY, 5))
        print("Z error: ", round(p.getLinkState(armUid, 4)[0][2]-targetZ, 5))
    stepCounter += 1



        


