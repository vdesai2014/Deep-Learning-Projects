import os
import pybullet as p
import pybullet_data
import math
import numpy as np
import time
from DofbotComms import RealWorldDofbot
from array import *

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
baseOrn = p.getQuaternionFromEuler([0, 0, 0])
armUid = p.loadURDF(dofbot_path, basePosition=[0,0,0], baseOrientation = baseOrn, useFixedBase=True)

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
"""
jointPoses = [0, -1.57, 0, 1.57, 0]
for i in range(5):
        p.setJointMotorControl2(armUid, i, p.POSITION_CONTROL, targetPosition = jointPoses[i])
for i in range(100):
    p.stepSimulation()
print("X: ", p.getLinkState(armUid, 4)[0])
print("Y: ", p.getLinkState(armUid, 4)[1])
print("Z: ", p.getLinkState(armUid, 4)[2])
time.sleep(100)
"""

p.setRealTimeSimulation(1)
x_in = p.addUserDebugParameter("X", -0.5, 0.5, 0)
y_in = p.addUserDebugParameter("Y", -0.5, 0.5, 0.165)
z_in = p.addUserDebugParameter("Z", -1.57, 1.57, 0.18)
rollId = p.addUserDebugParameter("roll", -3.14, 3.14, 0)
pitchId = p.addUserDebugParameter("pitch", -3.14, 3.14, 0)
yawId = p.addUserDebugParameter("yaw", -np.pi/2, np.pi/2, np.pi/2)
numJoints = p.getNumJoints(armUid)

def transformAngles(servoAngles):
    newAngles = [0, 0, 0, 0, 0, 0]
    for i in range(len(servoAngles)):
        newAngles[i] = int((servoAngles[i] + 1.57)*57.3)
    newAngles[5] = 90.0
    return newAngles

#dofbot = RealWorldDofbot()
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
    
    intArray = array('f', jointPoses)
    time.sleep(0.001)
    #dofbot.send_joint_pos(transformAngles(intArray))
    


jointZero = p.addUserDebugParameter("Base Joint", -1.57, 1.57, 0)
jointOne = p.addUserDebugParameter("Servo Two", -1.57, 1.57, 0)
jointTwo = p.addUserDebugParameter("Servo Three", -1.57, 1.57, 0)
jointThree = p.addUserDebugParameter("Servo Four", -1.57, 1.57, 0)
jointFour = p.addUserDebugParameter("Servo Five", -1.57, 3.14, 0)
jointFive = p.addUserDebugParameter("Servo Six", -1.57, 1.57, 0)

def transformAngles(servo):
    for i in range(len(servoAngles)):
        servo[i] = int((servoAngles[i] + 1.57)*57.3)
    return servo

#dofbot = RealWorldDofbot()
stepCounter = 0 
p.setRealTimeSimulation(1)
while(True):
    servoAngles = []
    servoAngles.append(p.readUserDebugParameter(jointZero))
    servoAngles.append(p.readUserDebugParameter(jointOne))
    servoAngles.append(p.readUserDebugParameter(jointTwo))
    servoAngles.append(p.readUserDebugParameter(jointThree))
    servoAngles.append(p.readUserDebugParameter(jointFour))
    servoAngles.append(p.readUserDebugParameter(jointFive))
    for i in range(len(servoAngles)):
        print("Setting joint " + str(i) + " to postion: " + str(servoAngles[i]))
        p.setJointMotorControl2(armUid, i, p.POSITION_CONTROL, targetPosition = servoAngles[i])
    #time.sleep(0.001)
    #dofbot.send_joint_pos(transformAngles(servoAngles))
    stepCounter += 1

        


