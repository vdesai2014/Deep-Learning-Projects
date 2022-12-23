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
planeUid = p.loadURDF(os.path.join(urdfRootPath,"plane.urdf"), basePosition=[0,0,-0.65])
# load table URDF
tableUid = p.loadURDF(os.path.join(urdfRootPath, "table/table.urdf"),basePosition=[0.5,0,-0.65])
dofbot_path = os.path.join(os.path.dirname(__file__), 'arm.urdf')

armUid = p.loadURDF(dofbot_path, basePosition=[0,-0.2,0], useFixedBase=True)

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

joints = []
joints.append(p.addUserDebugParameter("0", -1.57, 1.57, 0))
joints.append(p.addUserDebugParameter("1", -1.57, 1.57, 0))
joints.append(p.addUserDebugParameter("2", -1.57, 1.57, 0))
joints.append(p.addUserDebugParameter("3", -1.57, 1.57, 0))
joints.append(p.addUserDebugParameter("4", -1.57, 3.14, 0))
joints.append(p.addUserDebugParameter("5", -1.57, 3.14, 0))
joints.append(p.addUserDebugParameter("6", -1.57, 3.14, 0))
joints.append(p.addUserDebugParameter("7", -1.57, 3.14, 0))

stepCounter = 0
while(True):
    #print(stepCounter)
    stepCounter += 1
    for i in range(8):
        jointVal = p.readUserDebugParameter(joints[i])
        p.setJointMotorControl2(armUid, i, p.POSITION_CONTROL, targetPosition=jointVal)
        


