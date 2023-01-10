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
p.setGravity(0,0,-9.8)
p.resetDebugVisualizerCamera(cameraDistance=0.5,
                                     cameraYaw=0,
                                     cameraPitch=-40,
                                     cameraTargetPosition=[0, 0, 0.4])
# get current path
urdfRootPath=pybullet_data.getDataPath()
# load plane URDF
planeUid = p.loadURDF(os.path.join(urdfRootPath,"plane.urdf"), basePosition=[0,0,0])
dofbot_path = os.path.join(os.path.dirname(__file__), 'fiverrArm.urdf')
box_path = os.path.join(os.path.dirname(__file__), 'box.urdf')
baseOrn = p.getQuaternionFromEuler([0, -np.pi, 0])
id = p.loadURDF(dofbot_path, basePosition=[0,0,0.4], baseOrientation = baseOrn, useFixedBase=True)
box = p.loadURDF(box_path, basePosition=[0.05,0.125,0.01], baseOrientation = baseOrn)
box1 = p.loadURDF(box_path, basePosition=[0,0.125,0.01], baseOrientation = baseOrn)
box2 = p.loadURDF(box_path, basePosition=[-0.05,0.125,0.01], baseOrientation = baseOrn)
#p.addUserDebugLine((0, 0.125, 0), (0, 0.125, 1), lineWidth = 10.0)
p.changeDynamics(box, -1, mass=2.0, lateralFriction=1.5)
print("Mass: ", p.getDynamicsInfo(box, -1)[0])
print("Lateral Friction: ", p.getDynamicsInfo(box, -1)[1])
#boxTwo = p.loadURDF(box_path, basePosition=[0,0.1,0.03], baseOrientation = baseOrn)
p.changeVisualShape(id, -1, rgbaColor=[0,0,0,1])
p.changeVisualShape(id, 0, rgbaColor=[0,1,0,1])
p.changeVisualShape(id, 1, rgbaColor=[1,1,0,1])
p.changeVisualShape(id, 2, rgbaColor=[0,1,0,1])
p.changeVisualShape(id, 3, rgbaColor=[1,1,0,1])
p.changeVisualShape(id, 4, rgbaColor=[0,0,0,1])
#p.changeVisualShape(id, 5, rgbaColor=[0,1,0,0])
##p.changeVisualShape(id, 6, rgbaColor=[0,0,1,0])
##p.changeVisualShape(id, 7, rgbaColor=[1,0,0,0.5])

def jointType(joint):
    if(joint == 0):
        return "Revolute"
    elif(joint == 4):
        return "Fixed"

for i in range(p.getNumJoints(id)):
    print(" ")
    print("~~~~~~")
    print("Joint Index: ", p.getJointInfo(id, i)[0])
    print("Joint Name: ", p.getJointInfo(id, i)[1])
    print("Joint Type", jointType(p.getJointInfo(id, i)[2]))
    print("Joint Upper Limit: ", p.getJointInfo(id, i)[9])
    print("Joint Lower Limit: ", p.getJointInfo(id, i)[8])
    print("~~~~~~")
    print(" ")
"""
jointZero = p.addUserDebugParameter("g1", -1.57, 1.57, 0)
jointOne = p.addUserDebugParameter("f1", -1.57, 1.57, 0)

while(True):
    g1 = p.readUserDebugParameter(jointZero)
    f1 = p.readUserDebugParameter(jointOne)
    for i in range(5):
        p.setJointMotorControl2(id, i, p.POSITION_CONTROL, targetPosition = 0)
    p.setJointMotorControl2(id, 5, p.POSITION_CONTROL, targetPosition = g1)
    p.setJointMotorControl2(id, 6, p.POSITION_CONTROL, targetPosition = -g1)
    p.setJointMotorControl2(id, 7, p.POSITION_CONTROL, targetPosition = g1)
    p.setJointMotorControl2(id, 8, p.POSITION_CONTROL, targetPosition = -g1)
    p.setJointMotorControl2(id, 9, p.POSITION_CONTROL, targetPosition = -g1)
    p.setJointMotorControl2(id, 10, p.POSITION_CONTROL, targetPosition = -g1)
    p.setJointMotorControl2(id, 11, p.POSITION_CONTROL, targetPosition = -g1)
    p.setJointMotorControl2(id, 12, p.POSITION_CONTROL, targetPosition = -g1)
    p.stepSimulation()

p.setRealTimeSimulation(1)

def transformAngles(servoAngles):
    newAngles = [0, 0, 0, 0, 0, 0]
    for i in range(5):
        newAngles[i] = int((servoAngles[i] + 1.57)*57.3)
    newAngles[5] = int((servoAngles[5]+1.57)*101)
    return newAngles

#dofbot = RealWorldDofbot()
roll = 0
pitch = np.pi
yaw = -4*np.pi/2
orn = p.getQuaternionFromEuler([roll, pitch, yaw])

for i in range(10):
    targetX = 0
    targetY = 0.164
    targetZ = 0.221
    g1 = -1.57
    jointPoses = p.calculateInverseKinematics(id, 4, [targetX, targetY, targetZ], orn, maxNumIterations=1000,
                                                  residualThreshold=.01)
    for i in range(5):
        p.setJointMotorControl2(id, i, p.POSITION_CONTROL, targetPosition = jointPoses[i])
    p.setJointMotorControl2(id, 5, p.POSITION_CONTROL, targetPosition = g1)
    p.setJointMotorControl2(id, 6, p.POSITION_CONTROL, targetPosition = -g1)
    p.setJointMotorControl2(id, 7, p.POSITION_CONTROL, targetPosition = g1)
    p.setJointMotorControl2(id, 8, p.POSITION_CONTROL, targetPosition = -g1)
    p.setJointMotorControl2(id, 9, p.POSITION_CONTROL, targetPosition = -g1)
    p.setJointMotorControl2(id, 10, p.POSITION_CONTROL, targetPosition = -g1)
    p.setJointMotorControl2(id, 11, p.POSITION_CONTROL, targetPosition = -g1)
    p.setJointMotorControl2(id, 12, p.POSITION_CONTROL, targetPosition = -g1)
    intArray = array('f', jointPoses)
    transformAngles(intArray)
    #dofbot.send_joint_pos(transformAngles(intArray))
    p.stepSimulation()
    time.sleep(0.2)

for i in range(10):
    targetX = 0
    targetY = 0.164
    targetZ = 0.083
    g1 = -1.57
    jointPoses = p.calculateInverseKinematics(id, 4, [targetX, targetY, targetZ], orn, maxNumIterations=1000,
                                                  residualThreshold=.01)
    for i in range(5):
        p.setJointMotorControl2(id, i, p.POSITION_CONTROL, targetPosition = jointPoses[i], maxVelocity=3.0)
    p.setJointMotorControl2(id, 5, p.POSITION_CONTROL, targetPosition = g1)
    p.setJointMotorControl2(id, 6, p.POSITION_CONTROL, targetPosition = -g1)
    p.setJointMotorControl2(id, 7, p.POSITION_CONTROL, targetPosition = g1)
    p.setJointMotorControl2(id, 8, p.POSITION_CONTROL, targetPosition = -g1)
    p.setJointMotorControl2(id, 9, p.POSITION_CONTROL, targetPosition = -g1)
    p.setJointMotorControl2(id, 10, p.POSITION_CONTROL, targetPosition = -g1)
    p.setJointMotorControl2(id, 11, p.POSITION_CONTROL, targetPosition = -g1)
    p.setJointMotorControl2(id, 12, p.POSITION_CONTROL, targetPosition = -g1)
    p.stepSimulation()
    intArray = array('f', jointPoses)
    transformAngles(intArray)
    #dofbot.send_joint_pos(transformAngles(intArray))
    time.sleep(0.2)

time.sleep(1)

for i in range(10):
    g1 = -0.4
    p.setJointMotorControl2(id, 5, p.POSITION_CONTROL, targetPosition = g1)
    p.setJointMotorControl2(id, 6, p.POSITION_CONTROL, targetPosition = -g1)
    p.setJointMotorControl2(id, 7, p.POSITION_CONTROL, targetPosition = g1)
    p.setJointMotorControl2(id, 8, p.POSITION_CONTROL, targetPosition = -g1)
    p.setJointMotorControl2(id, 9, p.POSITION_CONTROL, targetPosition = -g1)
    p.setJointMotorControl2(id, 10, p.POSITION_CONTROL, targetPosition = -g1)
    p.setJointMotorControl2(id, 11, p.POSITION_CONTROL, targetPosition = -g1)
    p.setJointMotorControl2(id, 12, p.POSITION_CONTROL, targetPosition = -g1)
    p.stepSimulation()
    time.sleep(0.2)

for i in range(100):
    targetX = 0
    targetY = 0.164
    targetZ = 0.23
    g1 = -0.4
    jointPoses = p.calculateInverseKinematics(id, 4, [targetX, targetY, targetZ], orn, maxNumIterations=1000,
                                                  residualThreshold=.01)
    for i in range(5):
        p.setJointMotorControl2(id, i, p.POSITION_CONTROL, targetPosition = jointPoses[i], maxVelocity=3.0)
    p.setJointMotorControl2(id, 5, p.POSITION_CONTROL, targetPosition = g1)
    p.setJointMotorControl2(id, 6, p.POSITION_CONTROL, targetPosition = -g1)
    p.setJointMotorControl2(id, 7, p.POSITION_CONTROL, targetPosition = g1)
    p.setJointMotorControl2(id, 8, p.POSITION_CONTROL, targetPosition = -g1)
    p.setJointMotorControl2(id, 9, p.POSITION_CONTROL, targetPosition = -g1)
    p.setJointMotorControl2(id, 10, p.POSITION_CONTROL, targetPosition = -g1)
    p.setJointMotorControl2(id, 11, p.POSITION_CONTROL, targetPosition = -g1)
    p.setJointMotorControl2(id, 12, p.POSITION_CONTROL, targetPosition = -g1)
    p.stepSimulation()
    intArray = array('f', jointPoses)
    transformAngles(intArray)
    #dofbot.send_joint_pos(transformAngles(intArray))
    time.sleep(1)

"""
x_in = p.addUserDebugParameter("X", -0.5, 0.5, 0)
y_in = p.addUserDebugParameter("Y", -0.5, 0.5, 0.164)
z_in = p.addUserDebugParameter("Z", -1.57, 1.57, 0.2212)
g_in = p.addUserDebugParameter("Gripper", -1.57, 0.5, -1.57)
rollId = p.addUserDebugParameter("roll", -3.14, 3.14, 0)
pitchId = p.addUserDebugParameter("pitch", -3.14, 3.14, np.pi)
yawId = p.addUserDebugParameter("yaw", -np.pi, np.pi, 0)

while(True):
    targetX = p.readUserDebugParameter(x_in)
    targetY = p.readUserDebugParameter(y_in)
    targetZ = p.readUserDebugParameter(z_in)
    g1 = p.readUserDebugParameter(g_in)
    roll = p.readUserDebugParameter(rollId)
    pitch = p.readUserDebugParameter(pitchId)
    yaw = p.readUserDebugParameter(yawId)
    orn = p.getQuaternionFromEuler([roll, pitch, yaw])
    jointPoses = p.calculateInverseKinematics(id, 4, [targetX, targetY, targetZ], orn, maxNumIterations=1000,
                                                  residualThreshold=.01)
    for i in range(5):
        p.setJointMotorControl2(id, i, p.POSITION_CONTROL, targetPosition = jointPoses[i])
    p.setJointMotorControl2(id, 5, p.POSITION_CONTROL, targetPosition = g1)
    p.setJointMotorControl2(id, 6, p.POSITION_CONTROL, targetPosition = -g1)
    p.setJointMotorControl2(id, 7, p.POSITION_CONTROL, targetPosition = g1)
    p.setJointMotorControl2(id, 8, p.POSITION_CONTROL, targetPosition = -g1)
    p.setJointMotorControl2(id, 9, p.POSITION_CONTROL, targetPosition = -g1)
    p.setJointMotorControl2(id, 10, p.POSITION_CONTROL, targetPosition = -g1)
    p.setJointMotorControl2(id, 11, p.POSITION_CONTROL, targetPosition = -g1)
    p.setJointMotorControl2(id, 12, p.POSITION_CONTROL, targetPosition = -g1)

    print(p.getLinkState(id, 4)[0])
    p.stepSimulation()


jointZero = p.addUserDebugParameter("Base Joint", -1.57, 1.57, 0)
jointOne = p.addUserDebugParameter("Servo Two", -1.57, 1.57, 0)
jointTwo = p.addUserDebugParameter("Servo Three", -1.57, 1.57, 0)
jointThree = p.addUserDebugParameter("Servo Four", -1.57, 1.57, 0)
jointFour = p.addUserDebugParameter("Servo Five", -1.57, 3.14, 0)
jointFive = p.addUserDebugParameter("Servo Six", -1.57, 1.57, 0)

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
        #print("Setting joint " + str(i) + " to postion: " + str(servoAngles[i]))
        p.setJointMotorControl2(id, i, p.POSITION_CONTROL, targetPosition = servoAngles[i])
    #time.sleep(0.001)
    #dofbot.send_joint_pos(transformAngles(servoAngles))
    print(p.getLinkState(id, 4)[0])
    stepCounter += 1



