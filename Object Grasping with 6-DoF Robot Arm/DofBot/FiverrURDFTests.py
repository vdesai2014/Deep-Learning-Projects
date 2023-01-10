import pybullet as p
import pybullet_data
import os 
import time
import numpy as np 

p.connect(p.GUI)
p.resetSimulation()
p.setPhysicsEngineParameter(numSolverIterations=200)
p.setPhysicsEngineParameter(solverResidualThreshold=1e-30)
p.setGravity(0,0,-10)
# get current path
urdfRootPath=pybullet_data.getDataPath()
# load plane URDF
planeUid = p.loadURDF(os.path.join(urdfRootPath,"plane.urdf"), basePosition=[0,0,0])
dofbot_path = os.path.join(os.path.dirname(__file__), 'fiverrArm.urdf')
baseOrn = p.getQuaternionFromEuler([0, 0, 0])
armUid = p.loadURDF(dofbot_path, basePosition=[0,0,0], baseOrientation = baseOrn, useFixedBase=True)
box = p.loadURDF('/home/vrushank/Downloads/pybullet_ur5_robotiq-robotflow/urdf/box.urdf', basePosition=[0, 0, 0.1], baseOrientation = baseOrn)

for i in range(p.getNumJoints(armUid)):
    print("Joint Index: ", p.getJointInfo(armUid, i)[0])
    print("Joint Name: ", p.getJointInfo(armUid, i)[1])

"""
ll1_5 = p.addUserDebugParameter("ll_1", -0.2, 1.5708, -0.2)
rl1_6 = p.addUserDebugParameter("ll_1", -0.2, 1.5708, -0.2)
g1_7 = p.addUserDebugParameter("ll_1", -0.25, 1.5708, -0.25)
f1_8 = p.addUserDebugParameter("ll_1", 0.2, 1.5708, 0.2)
g2_9 = p.addUserDebugParameter("ll_1", 0.25, 1.5708, 0.25)
f2_10 = p.addUserDebugParameter("ll_1", -0.2, 1.5708, -0.2)
"""

gripperAngle = p.addUserDebugParameter("gripper", -0.25, 1.5708, 0.8)
z = p.addUserDebugParameter("z", 0, 1.5, 0)
"""
c = p.createConstraint(armUid, 7,
                                   armUid, 8,
                                   jointType=p.JOINT_GEAR,
                                   jointAxis=[0, 1, 0],
                                   parentFramePosition=[0, 0, 0],
                                   childFramePosition=[0, 0, 0])
p.changeConstraint(c, gearRatio=1, maxForce=100, erp=1)
"""
"""
d = p.createConstraint(armUid, 7,
                                   armUid, 8,
                                   jointType=p.JOINT_GEAR,
                                   jointAxis=[1, 0, 0],
                                   parentFramePosition=[0, 0, 0],
                                   childFramePosition=[0, 0, 0])
p.changeConstraint(d, gearRatio=1, maxForce=100, erp=1)

e = p.createConstraint(armUid, 7,
                                   armUid, 9,
                                   jointType=p.JOINT_GEAR,
                                   jointAxis=[1, 0, 0],
                                   parentFramePosition=[0, 0, 0],
                                   childFramePosition=[0, 0, 0])
p.changeConstraint(e, gearRatio=1, maxForce=100, erp=0.8)
d = p.createConstraint(armUid, 6,
                                   armUid, 5,
                                   jointType=p.JOINT_GEAR,
                                   jointAxis=[0, 1, 0],
                                   parentFramePosition=[0, 0, 0],
                                   childFramePosition=[0, 0, 0])
p.changeConstraint(d, gearRatio=1, maxForce=100, erp=1)
"""
stepCounter = 0 
while(True):
    grip = p.readUserDebugParameter(gripperAngle)
    zed = p.readUserDebugParameter(z)
    #if(stepCounter%100 == 0):
        #print("g1 position: ", p.getJointState(armUid, 6)[0])
        #print("f1 position: ", p.getJointState(armUid, 7)[0])
    for i in range(7):
        p.setJointMotorControl2(armUid, i, p.POSITION_CONTROL, targetPosition = 0)
    
    p.setJointMotorControl2(armUid, 7, p.POSITION_CONTROL, targetPosition = -grip, force=50, maxVelocity=2)
    """
    p.setJointMotorControl2(armUid, 7+2, p.POSITION_CONTROL, targetPosition = -grip, force=50, maxVelocity=2)
    p.setJointMotorControl2(armUid, 6+2, p.POSITION_CONTROL, targetPosition = grip, force=50, maxVelocity=2)
    p.setJointMotorControl2(armUid, 8+2, p.POSITION_CONTROL, targetPosition = grip, force=50, maxVelocity=2)
    p.setJointMotorControl2(armUid, 9+2, p.POSITION_CONTROL, targetPosition = -grip, force=50, maxVelocity=2)
    p.setJointMotorControl2(armUid, 10+2, p.POSITION_CONTROL, targetPosition = grip, force=50, maxVelocity=2)
    """
    p.setJointMotorControl2(armUid, 0, p.POSITION_CONTROL, targetPosition = zed)
    p.stepSimulation()
