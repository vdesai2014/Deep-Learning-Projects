import os
import pybullet as p
import pybullet_data
import math
import numpy as np
import time
from DofbotComms import RealWorldDofbot
from array import *
import transform_utils
import transformations

p.connect(p.GUI)
p.resetSimulation()
p.setGravity(0,0,-9.8)
p.resetDebugVisualizerCamera(cameraDistance=0.5,
                                     cameraYaw=0,
                                     cameraPitch=-40,
                                     cameraTargetPosition=[0, 0, 0.4])

urdfRootPath=pybullet_data.getDataPath()
planeUid = p.loadURDF(os.path.join(urdfRootPath,"plane.urdf"), basePosition=[0,0,0])
dofbot_path = os.path.join(os.path.dirname(__file__), 'fiverrArm.urdf')
box_path = os.path.join(os.path.dirname(__file__), 'box.urdf')
baseOrn = p.getQuaternionFromEuler([0, -np.pi, 0])
id = p.loadURDF(dofbot_path, basePosition=[0,0,0.4], baseOrientation = baseOrn, useFixedBase=True)
box = p.loadURDF(box_path, basePosition=[0,0.125,0.01], baseOrientation = baseOrn)
p.changeDynamics(box, -1, mass=2.0, lateralFriction=1.5)
p.changeVisualShape(id, -1, rgbaColor=[0,0,0,1])
p.changeVisualShape(id, 0, rgbaColor=[0,1,0,1])
p.changeVisualShape(id, 1, rgbaColor=[1,1,0,1])
p.changeVisualShape(id, 2, rgbaColor=[0,1,0,1])
p.changeVisualShape(id, 3, rgbaColor=[1,1,0,1])
p.changeVisualShape(id, 4, rgbaColor=[0,0,0,1])

x_in = p.addUserDebugParameter("X", -0.5, 0.5, 0)
y_in = p.addUserDebugParameter("Y", -0.5, 0.5, 0.164)
z_in = p.addUserDebugParameter("Z", -1.57, 1.57, 0.2212)
g_in = p.addUserDebugParameter("Gripper", -1.57, 0.5, -1.57)
rollId = p.addUserDebugParameter("roll", -3.14, 3.14, 0)
pitchId = p.addUserDebugParameter("pitch", -3.14, 3.14, np.pi)
yawId = p.addUserDebugParameter("yaw", -np.pi, np.pi, 0)
rotX = p.addUserDebugParameter("Camera Rotation X", -3.14, 3.14, 0)
rotY = p.addUserDebugParameter("Camera Rotation Y", -3.14, 3.14, 0)
rotZ = p.addUserDebugParameter("Camera Rotation Z", -3.14, 3.14, 0)
transX = p.addUserDebugParameter("Camera Translation X", -3, 3, 0)
transY = p.addUserDebugParameter("Camera Translation Y", -3, 3, 0)
transZ = p.addUserDebugParameter("Camera Translation Z", -3, 3, 0)

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

    RotX = 0.5
    RotY = p.readUserDebugParameter(rotY)
    RotZ = p.readUserDebugParameter(rotZ)
    TransX = p.readUserDebugParameter(transX)
    TransY = 0.065
    TransZ = p.readUserDebugParameter(transZ)
    camRotation = p.getQuaternionFromEuler([RotX, RotY, RotZ])
    transform_dict = {'translation' : [TransX, TransY, TransZ], 'rotation' : [camRotation[0], camRotation[1], camRotation[2], camRotation[3]]} 
    transform = transform_utils.from_dict(transform_dict)
    """
    transform = np.copy(_transform)
    # Randomize translation
    magnitue = np.random.uniform(0., 0.002)
    direction = transform_utils.random_unit_vector()
    transform[:3, 3] += magnitue * direction
    # Randomize rotation
    angle = np.random.uniform(0., 0.0349)
    axis = transform_utils.random_unit_vector()
    q = transformations.quaternion_about_axis(angle, axis)
    transform = np.dot(transformations.quaternion_matrix(q), transform)
    """
    _h_robot_camera = transform
    pos, orn, _, _, _, _ = p.getLinkState(id, 4)
    h_world_robot = transform_utils.from_pose(pos, orn)
    h_camera_world = np.linalg.inv(np.dot(h_world_robot, _h_robot_camera))
    gl_view_matrix = h_camera_world.copy()
    gl_view_matrix[2, :] *= -1 # flip the Z axis to comply to OpenGL
    gl_view_matrix = gl_view_matrix.flatten(order='F')
    projectionMatrix = p.computeProjectionMatrixFOV(
            fov=45.0,
            aspect=1.0,
            nearVal=0.02,
            farVal=3.0)
    _, _, _, depthImg, _ = p.getCameraImage(
            width=64, 
            height=64,
            viewMatrix=gl_view_matrix,
            projectionMatrix=projectionMatrix)
    p.stepSimulation()

