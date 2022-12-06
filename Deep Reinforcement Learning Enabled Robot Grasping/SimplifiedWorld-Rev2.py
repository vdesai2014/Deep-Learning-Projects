import os, inspect
from turtle import speed
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)
import pybullet as p 
import pybullet_data
import gym 
import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import time
import pybullet as p
#from . import kuka
import random
import pybullet_data
from pkg_resources import parse_version
import pybullet_utils.bullet_client as bc
import stable_baselines3
from stable_baselines3.sac.policies import CnnPolicy
from stable_baselines3 import SAC
from matplotlib import pyplot as plt
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvWrapper
#from stable_baselines.common.cmd_util import make_vec_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env
import transform_utils
import transformations
from sklearn.preprocessing import MinMaxScaler
from collections import namedtuple
from attrdict import AttrDict

class SimplifiedWorld(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 50}

    def __init__(self, renders = False):
        self._maxSteps = 200
        self._timeStep = 1./240
        self._urdfRoot = pybullet_data.getDataPath()
        self._observation = []
        self._envStepCounter = 0
        self._renders = renders
        self._width = 64
        self._height = 64
        self.terminated = False 
        self.extent = 0.01
        self._max_translation = 0.05
        self._max_yaw_rotation = 0.15
        self.main_joints = [0, 1, 2, 3] 
        self._initial_height = 0
        transform_dict = {'translation' : [0.0, 0.0573, 0.0451], 'rotation' : [0.0, -0.1305, 0.9914, 0.0]} #mess with this to put the camera in a place that makes sense
        self._transform = transform_utils.from_dict(transform_dict) 

        if self._renders:
            self._physicsClient = bc.BulletClient(connection_mode = p.GUI)
        else: 
            self._physicsClient = bc.BulletClient(connection_mode = p.DIRECT)
        p.resetDebugVisualizerCamera(1.3, 180, -41, [0.52, -0.2, -0.33])

        self.reset() 

        self.observation_space = spaces.Box(low=0,
                                        high=255,
                                        shape=(2, self._height, self._width), dtype = np.uint8)
        self.action_space = spaces.Box(-1.,1., shape=(5,), dtype=np.float32)
    
    def reset(self):
        self.terminated = False
        self._gripper_open = True
        self.endEffectorAngle = 0.
        self._physicsClient.resetSimulation()
        self._physicsClient.setPhysicsEngineParameter(numSolverIterations=150, fixedTimeStep = 1./240., enableConeFriction = 1)
        self._physicsClient.setGravity(0, 0, -10)
        self._envStepCounter = 0
        self._lifting = False
        self.lift_dist = 0.01

        self._physicsClient.setRealTimeSimulation(1)

        #Load white floor
        plane = self._physicsClient.loadURDF("plane/planeRL.urdf", [0., 0., -0.1], [0., 0., 0., 1.])

        numBlocks = 3

        self._blocks = []
        for i in range (numBlocks):
            randBlock = np.random.randint(100, 999)
            path = self._urdfRoot + '/random_urdfs/' + str(randBlock) + '/' + str(randBlock) + '.urdf'
            #path = [os.path.join(self._urdfRoot, 'random_urdfs',(str(randBlock)), str(randBlock) + '.urdf')]
            position = np.r_[np.random.uniform(-self.extent, self.extent), np.random.uniform(-self.extent, self.extent), 0]
            orientation = transformations.random_quaternion()
            block = self._physicsClient.loadURDF(path, position, orientation, globalScaling = 0.5)
            self._blocks.append(block)
            for i in range(1000):
                self._physicsClient.stepSimulation()

        robotUrdfPath = './urdf/robotiq_85_gripper_simple.urdf'
        robotStartPos = [0, 0, self._initial_height]
        robotStartOrn = p.getQuaternionFromEuler([0, 1.57, 0])
        self.robotID = p.loadURDF(robotUrdfPath, robotStartPos, robotStartOrn,
                     flags=p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT)
        self._physicsClient.stepSimulation()
        self.currentX = 0
        self.currentY = 0
        self.currentZ = self._initial_height
        self.currentYaw = 0

        transform = np.copy(self._transform)
        # Randomize translation
        magnitude = np.random.uniform(0., 0.002)
        direction = transform_utils.random_unit_vector()
        transform[:3, 3] += magnitude * direction
        # Randomize rotation
        angle = np.random.uniform(0., 0.0349)
        axis = transform_utils.random_unit_vector()
        q = transformations.quaternion_about_axis(angle, axis)
        transform = np.dot(transformations.quaternion_matrix(q), transform)
        self._h_robot_camera = transform
        self._target_joint_pos = 0.0
        rgb = self.getObservation()

        controlJoints = ["robotiq_85_left_knuckle_joint",
                 "robotiq_85_right_knuckle_joint",
                 "robotiq_85_left_inner_knuckle_joint",
                 "robotiq_85_right_inner_knuckle_joint",
                 "robotiq_85_left_finger_tip_joint",
                 "robotiq_85_right_finger_tip_joint"]
        
        numJoints = p.getNumJoints(self.robotID)
        jointInfo = namedtuple("jointInfo",
                            ["id","name","type","lowerLimit","upperLimit","maxForce","maxVelocity"])
        jointTypeList = ["REVOLUTE", "PRISMATIC", "SPHERICAL", "PLANAR", "FIXED"]
        self.joints = AttrDict()

        for i in range(numJoints):
            info = p.getJointInfo(self.robotID, i)
            jointID = info[0]
            jointName = info[1].decode("utf-8")
            jointType = jointTypeList[info[2]]
            jointLowerLimit = info[8]
            jointUpperLimit = info[9]
            jointMaxForce = info[10]
            jointMaxVelocity = info[11]
            singleInfo = jointInfo(jointID, jointName, jointType, jointLowerLimit, jointUpperLimit, jointMaxForce, jointMaxVelocity)
            self.joints[singleInfo.name] = singleInfo #creates dictionary with tuples as specified in jointInfo 
            # register index of dummy center link
            if jointName == "gripper_roll": #sets dummy_center_indicator_link_index as gripper roll
                dummy_center_indicator_link_index = i
        
        self.position_control_joint_name = ["center_x",
                                                "center_y",
                                                "center_z",
                                                "gripper_roll",
                                                "gripper_pitch",
                                                "gripper_yaw"]

        self.gripper_main_control_joint_name = "robotiq_85_left_knuckle_joint"
        self.mimic_joint_name = ["robotiq_85_right_knuckle_joint",
                            "robotiq_85_left_inner_knuckle_joint",
                            "robotiq_85_right_inner_knuckle_joint",
                            "robotiq_85_left_finger_tip_joint",
                            "robotiq_85_right_finger_tip_joint"]
        self.dummy_center_indicator_link_index = 0 
                            
        self.mimic_multiplier = [1, 1, 1, -1, -1]

        mimic_parent_id = 7

        c1 = self._physicsClient.createConstraint(self.robotID, mimic_parent_id,
                                        self.robotID, 9,
                                        jointType=p.JOINT_GEAR,
                                        jointAxis=[0, 1, 0],
                                        parentFramePosition=[0, 0, 0],
                                        childFramePosition=[0, 0, 0])
        self._physicsClient.changeConstraint(c1, gearRatio=-1, maxForce=100, erp=1)

        c2 = self._physicsClient.createConstraint(self.robotID, mimic_parent_id,
                                        self.robotID, 11,
                                        jointType=p.JOINT_GEAR,
                                        jointAxis=[0, 1, 0],
                                        parentFramePosition=[0, 0, 0],
                                        childFramePosition=[0, 0, 0])
        self._physicsClient.changeConstraint(c2, gearRatio=-1, maxForce=100, erp=1)

        c3 = self._physicsClient.createConstraint(self.robotID, mimic_parent_id,
                                        self.robotID, 13,
                                        jointType=p.JOINT_GEAR,
                                        jointAxis=[0, 1, 0],
                                        parentFramePosition=[0, 0, 0],
                                        childFramePosition=[0, 0, 0])
        self._physicsClient.changeConstraint(c3, gearRatio=-1, maxForce=100, erp=1)

        c4 = self._physicsClient.createConstraint(self.robotID, mimic_parent_id,
                                        self.robotID, 12,
                                        jointType=p.JOINT_GEAR,
                                        jointAxis=[0, 1, 0],
                                        parentFramePosition=[0, 0, 0],
                                        childFramePosition=[0, 0, 0])
        self._physicsClient.changeConstraint(c4, gearRatio=1, maxForce=100, erp=1)

        c5 = self._physicsClient.createConstraint(self.robotID, mimic_parent_id,
                                        self.robotID, 14,
                                        jointType=p.JOINT_GEAR,
                                        jointAxis=[0, 1, 0],
                                        parentFramePosition=[0, 0, 0],
                                        childFramePosition=[0, 0, 0])
        self._physicsClient.changeConstraint(c5, gearRatio=1, maxForce=100, erp=1)

        self._physicsClient.enableJointForceTorqueSensor(self.robotID, 14)

        return rgb

    def __del__(self):
        self._physicsClient.disconnect()
    
    def getObservation(self):
        pos, orn, _, _, _, _ = self._physicsClient.getLinkState(self.robotID, 6)
        h_world_robot = transform_utils.from_pose(pos, orn)
        h_camera_world = np.linalg.inv(np.dot(h_world_robot, self._h_robot_camera))
        gl_view_matrix = h_camera_world.copy()
        gl_view_matrix[2, :] *= -1 # flip the Z axis to comply to OpenGL
        gl_view_matrix = gl_view_matrix.flatten(order='F')
        projectionMatrix = p.computeProjectionMatrixFOV(
            fov=45.0,
            aspect=1.0,
            nearVal=0.02,
            farVal=3.0)
        _, _, _, depthImg, _ = self._physicsClient.getCameraImage(
            width=self._width, 
            height=self._height,
            viewMatrix=gl_view_matrix,
            projectionMatrix=projectionMatrix)

        obs = np.array(depthImg)
        near, far = 0.02, 2
        obs = obs[np.newaxis, :, :]
        depth_buffer = np.asarray(depthImg, np.float32).reshape(  #need to study what the best way to implement here is once the world is functional 
            (self._height, self._width))
        obs = 1. * far * near / (far - (far - near) * depth_buffer)

        holdOver = np.zeros((1, 64, 64))
        return holdOver

    def closeGripper(self):
        if(self._gripper_open == False):
            return 
        gripper_opening_length = 0.085
        for i in range (100):
            force_X = p.getJointState(self.robotID, 14)[2][0]
            force_Y = p.getJointState(self.robotID, 14)[2][1]
            force_Z = p.getJointState(self.robotID, 14)[2][2]
            if (np.linalg.norm([force_X, force_Y, force_Z]) > 1):
                print("Force limit hit when arm was at length: ", gripper_opening_length)
                break 
            gripper_opening_angle = 0.715 - math.asin((gripper_opening_length - 0.010) / 0.1143)    # angle calculation for gripper angle 

            p.setJointMotorControl2(self.robotID,
                                    self.joints[self.gripper_main_control_joint_name].id,
                                    p.POSITION_CONTROL,
                                    targetPosition=gripper_opening_angle,
                                    force=100.0,
                                    maxVelocity=self.joints[self.gripper_main_control_joint_name].maxVelocity) #sets main control join position to target
            for i in range(len(self.mimic_joint_name)):
                joint = self.joints[self.mimic_joint_name[i]]
                p.setJointMotorControl2(self.robotID, joint.id, p.POSITION_CONTROL,
                                        targetPosition=gripper_opening_angle * self.mimic_multiplier[i], #moves mimic joints as necessary for closing gripper 
                                        force=joint.maxForce,
                                        maxVelocity=joint.maxVelocity) 
            p.stepSimulation()
            gripper_opening_length -= 0.00085
        self._gripper_open = False

    def openGripper(self):
        if(self._gripper_open == True):
            return 
        gripper_opening_length = 0.085
        gripper_opening_angle = 0.715 - math.asin((gripper_opening_length - 0.010) / 0.1143)
        p.setJointMotorControl2(self.robotID,
                                    self.joints[self.gripper_main_control_joint_name].id,
                                    p.POSITION_CONTROL,
                                    targetPosition=gripper_opening_angle,
                                    force=100.0,
                                    maxVelocity=self.joints[self.gripper_main_control_joint_name].maxVelocity) #sets main control join position to target 
        for i in range(len(self.mimic_joint_name)):
            joint = self.joints[self.mimic_joint_name[i]]
            p.setJointMotorControl2(self.robotID, joint.id, p.POSITION_CONTROL,
                                    targetPosition=gripper_opening_angle * self.mimic_multiplier[i], #moves mimic joints as necessary for closing gripper 
                                    force=joint.maxForce,
                                    maxVelocity=joint.maxVelocity)
        p.stepSimulation()
        self._gripper_open = True

    def _clip_translation_vector(self, translation, yaw):
        """Clip the vector if its norm exceeds the maximal allowed length."""
        length = np.linalg.norm(translation)
        if length > self._max_translation:
            translation *= self._max_translation / length
        if yaw > self._max_yaw_rotation:
            yaw *= self._max_yaw_rotation / yaw
        return translation, yaw

    def step(self,action):
        
        high = np.r_[[self._max_translation]
                         * 3, self._max_yaw_rotation, 1.]
        self._action_scaler = MinMaxScaler((-1, 1))
        self._action_scaler.fit(np.vstack((-1. * high, high)))
        action = self._action_scaler.inverse_transform(np.array([action]))
        action = action.squeeze()
        """
        dx = action[0]
        dy = action[1]
        dz = action[2]
        yaw = action[3]
        gripper = action[4]

        self.currentX = self.currentX + dx
        self.currentY = self.currentX + dy
        self.currentZ = self.currentX + dz
        self.currentYaw = self.currentYaw + yaw

        print(self.currentX, self.currentY, self.currentZ, self.currentYaw)
        time.sleep(2)

        parameter_orientation = p.getQuaternionFromEuler([0, 1.57, self.currentYaw])
        jointPose = p.calculateInverseKinematics(self.robotID,
                                             self.dummy_center_indicator_link_index,
                                             [self.currentX, self.currentY, self.currentZ],
                                             parameter_orientation) #calculates joint pose corresponding to commanded position & rotation

        for jointName in self.joints:
            if jointName in self.position_control_joint_name:
                joint = self.joints[jointName]
                p.setJointMotorControl2(self.robotID, joint.id, p.POSITION_CONTROL,
                                    targetPosition=jointPose[joint.id], force=joint.maxForce,
                                    maxVelocity=joint.maxVelocity)
        
        if(gripper > 0): 
            self.closeGripper()
        else:
            self.openGripper()
        """
        translation, yaw_rotation = self._clip_translation_vector(action[:3], action[3])
        open_close = action[4]
        #time.sleep(100)
        if(open_close > 0): 
            self.closeGripper()
        else:
            self.openGripper()
        
        pos, orn, _, _, _, _ = self._physicsClient.getLinkState(self.robotID, 0)
        #print(pos)
        _, _, yaw = transformations.euler_from_quaternion(orn)
        T_world_old = transformations.compose_matrix(
            angles=[np.pi, 0., yaw], translate=pos)
        T_old_to_new = transformations.compose_matrix(
            angles=[0., 0., yaw_rotation], translate=translation)
        T_world_new = np.dot(T_world_old, T_old_to_new)
        self.endEffectorAngle += yaw_rotation
        target_pos, target_orn = transform_utils.to_pose(T_world_new)
        self.endEffectorAngle += yaw_rotation
        yaw = self.endEffectorAngle
        target_pos[1] *= -1
        target_pos[2] = -1 * (target_pos[2] - self._initial_height)
        #print("Commanded X: ", target_pos[0])
        #print("Commanded Y: ", target_pos[1])
        #parameter_orientation = self._physicsClient.getQuaternionFromEuler([0, 0, 0])
        #jointPose = self._physicsClient.calculateInverseKinematics(self.robotID,
        #                                     self.dummy_center_indicator_link_index,
        #                                     [1, 1, 0],
        #                                     parameter_orientation) #calculates joint pose corresponding to commanded position & rotation
        
        for jointName in self.joints:
            if jointName in self.position_control_joint_name:
                if(jointName == 'center_x'):
                    joint = self.joints[jointName]
                    self._physicsClient.setJointMotorControl2(self.robotID, joint.id, p.POSITION_CONTROL,
                                        targetPosition=target_pos[0], force=joint.maxForce,
                                        maxVelocity=joint.maxVelocity)
                    #print("Joint name: ", jointName)
                    #print("Commanded to position: ", target_pos[0])
                elif(jointName == 'center_y'):
                    joint = self.joints[jointName]
                    self._physicsClient.setJointMotorControl2(self.robotID, joint.id, p.POSITION_CONTROL,
                                        targetPosition=target_pos[1], force=joint.maxForce,
                                        maxVelocity=joint.maxVelocity)
                    #print("Joint name: ", jointName)
                    #print("Commanded to position: ", target_pos[1])
                elif(jointName == 'center_z'):
                    joint = self.joints[jointName]
                    self._physicsClient.setJointMotorControl2(self.robotID, joint.id, p.POSITION_CONTROL,
                                        targetPosition=target_pos[2], force=joint.maxForce,
                                        maxVelocity=joint.maxVelocity)
                    #print("Joint name: ", jointName)
                    #print("Commanded to position: ", target_pos[2])
                elif(jointName == 'gripper_roll'):
                    joint = self.joints[jointName]
                    self._physicsClient.setJointMotorControl2(self.robotID, joint.id, p.POSITION_CONTROL,
                                        targetPosition=self.endEffectorAngle, force=joint.maxForce,
                                        maxVelocity=joint.maxVelocity)
                    #print("Joint name: ", jointName)
                    #print("Commanded to position: ", self.endEffectorAngle)

        self._physicsClient.stepSimulation()
        self._envStepCounter += 1
        self._observation = self.getObservation()
        reward = self.getReward()
        if(self._envStepCounter > self._maxSteps):
            self.terminated = True
            info = {"TimeLimit.truncated" : True}
            print("No Reward.")
        else:
            info = {}
        done = self.terminated

        return self._observation, reward, done, info
    
    def getReward(self): 
        return -0.01

env = SimplifiedWorld(renders = True)
env = make_vec_env(lambda: env, n_envs=1)
env = VecNormalize(env)
model = SAC("CnnPolicy", env, verbose=2, seed = 0, buffer_size = 100000, batch_size = 64, device = 'cpu', train_freq = (1, "episode"))
model.learn(total_timesteps=1000000, log_interval=4)

#UNIT TEST - ensure multiple random initialization result in block position/orientation congruent with Baris thesis
#UNIT TEST - ability for gripper to translate rotate around the environment without camera becoming imporperly detached 
#UNIT TEST - ability for rotational values to make sense when beyond +pi or -pi
