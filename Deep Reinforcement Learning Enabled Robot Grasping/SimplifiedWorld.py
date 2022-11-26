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
import transform_utils
import transformations
from sklearn.preprocessing import MinMaxScaler

class SimplifiedWorld(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 50}

    def __init__(self, renders = False):
        self._maxSteps = 150
        self._timeStep = 1./240
        self._urdfRoot = pybullet_data.getDataPath()
        self._observation = []
        self._envStepCounter = 0
        self._renders = renders
        self._width = 64
        self._height = 64
        self.terminated = False 
        self.extent = 0.01
        self._max_translation = 0.03
        self._max_yaw_rotation = 0.15
        self.main_joints = [0, 1, 2, 3] 
        self._initial_height = 0.15
        transform_dict = {'translation' : [0.0, 0.0573, 0.0451], 'rotation' : [0.0, -0.1305, 0.9914, 0.0]}
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
        self.action_space = gym.spaces.Box(-1.,1., shape=(5,), dtype=np.float32)
    
    def reset(self):
        self.terminated = False
        self._gripper_open = False
        self.endEffectorAngle = 0.
        self._physicsClient.resetSimulation()
        self._physicsClient.setPhysicsEngineParameter(numSolverIterations=150, fixedTimeStep = 1./240., enableConeFriction = 1)
        self._physicsClient.setGravity(0, 0, -10)
        self._envStepCounter = 0
        self._lifting = False
        self.lift_dist = 0.01

        #Load white floor
        plane = self._physicsClient.loadURDF("plane/planeRL.urdf", [0., 0., -0.196], [0., 0., 0., 1.])

        numBlocks = 3

        self._blocks = []
        for i in range (numBlocks):
            randBlock = np.random.randint(100, 999)
            path = self._urdfRoot + '/random_urdfs/' + str(randBlock) + '/' + str(randBlock) + '.urdf'
            #path = [os.path.join(self._urdfRoot, 'random_urdfs',(str(randBlock)), str(randBlock) + '.urdf')]
            position = np.r_[np.random.uniform(-self.extent, self.extent), np.random.uniform(-self.extent, self.extent), 0.1]
            orientation = transformations.random_quaternion()
            block = self._physicsClient.loadURDF(path, position, orientation)
            self._blocks.append(block)
            for i in range(1000):
                self._physicsClient.stepSimulation() 
        
        #load simple gripper, use 0,1,2 to control X, Y, Z pos, 3 to control gripper yaw, & 7/9 to close gripper
        start_pos = [0, 0, self._initial_height]
        start_orn = [1, 0, 0, 0] #CONFIRM - random environment intialization is congruent with simple base environment outlined by Baris
        #start_orn = p.getQuaternionFromEuler([3.14, 0, 3.14])
        self.model_id = self._physicsClient.loadSDF("gripper/wsg50_one_motor_gripper_new.sdf", globalScaling = 1.)[0]
        self._physicsClient.resetBasePositionAndOrientation(self.model_id, start_pos, start_orn)
        self._physicsClient.stepSimulation()
        transform = np.copy(self._transform)

        # Randomize translation
        magnitue = np.random.uniform(0., 0.002)
        direction = transform_utils.random_unit_vector()
        transform[:3, 3] += magnitue * direction
        # Randomize rotation
        angle = np.random.uniform(0., 0.0349)
        axis = transform_utils.random_unit_vector()
        q = transformations.quaternion_about_axis(angle, axis)
        transform = np.dot(transformations.quaternion_matrix(q), transform)
        self._h_robot_camera = transform
        self._target_joint_pos = 0.0
        rgb = self.getObservation()
        return rgb

    def __del__(self):
        self._physicsClient.disconnect()
    
    def getObservation(self):
        #print(self._physicsClient.getLinkState(self.model_id, 3))
        pos, orn, _, _, _, _ = self._physicsClient.getLinkState(self.model_id, 3)
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
        """
        for i in range(64):
            for j in range(64):
                if(j == 0):
                    print("[")
                if(j == 32):
                    print(" ")
                print(round(obs[i][j], 3), end = ' ')
                if(j == 63):
                    print(" ")
                    print("]")
            print(" ")
        """
        near, far = 0.02, 2
        obs = obs[np.newaxis, :, :]
        depth_buffer = np.asarray(depthImg, np.float32).reshape(
            (self._height, self._width))
        obs = 1. * far * near / (far - (far - near) * depth_buffer)
        obs *= 255
        #obs = obs[np.newaxis, :, :]
        sensor_pad = np.zeros((64, 64))
        self._obs_scaler = 1. / 0.1
        state = self._obs_scaler * self.get_gripper_width() * 255
        sensor_pad[0][0] = state
        obs_stacked = np.dstack((obs, sensor_pad))
        obs_stackedd = np.reshape(obs_stacked, (2, 64, 64))
        for i in range(64):
            for j in range(64):
                if(j == 0):
                    print("[")
                if(j == 32):
                    print(" ")
                print(round(obs_stacked[i][j][0], 3), end = ' ')
                if(j == 63):
                    print(" ")
                    print("]")
            print(" ")
        for i in range(64):
            for j in range(64):
                if(j == 0):
                    print("[")
                if(j == 32):
                    print(" ")
                print(round(obs_stacked[i][j][1], 3), end = ' ')
                if(j == 63):
                    print(" ")
                    print("]")
            print(" ")
        return obs_stackedd

    def closeGripper(self):
        self._physicsClient.setJointMotorControl2(
            self.model_id, 7,
            controlMode=p.POSITION_CONTROL,
            targetPosition=0.05,
            force=100.)
        self._physicsClient.setJointMotorControl2(
            self.model_id, 9,
            controlMode=p.POSITION_CONTROL,
            targetPosition=0.05,
            force=100.)
        for i in range(100):
            self._physicsClient.stepSimulation() 
        self._target_joint_pos = 0.05

    def openGripper(self):
        self._physicsClient.setJointMotorControl2(
            self.model_id, 7,
            controlMode=p.POSITION_CONTROL,
            targetPosition=0,
            force=100.)
        self._physicsClient.setJointMotorControl2(
            self.model_id, 9,
            controlMode=p.POSITION_CONTROL,
            targetPosition=0,
            force=100.)
        for i in range(100):
            self._physicsClient.stepSimulation() 
        self._target_joint_pos = 0.0

    def _clip_translation_vector(self, translation, yaw):
        """Clip the vector if its norm exceeds the maximal allowed length."""
        length = np.linalg.norm(translation)
        if length > self._max_translation:
            translation *= self._max_translation / length
        if yaw > self._max_yaw_rotation:
            yaw *= self._max_yaw_rotation / yaw
        return translation, yaw

    def step(self,action):
        #print(action)
        #action = [ 0.06817436, -0.07152063,  0,  0.47535467, -0.5590924]
        high = np.r_[[self._max_translation]
                         * 3, self._max_yaw_rotation, 1.]
        self._action_scaler = MinMaxScaler((-1, 1))
        self._action_scaler.fit(np.vstack((-1. * high, high)))
        action = self._action_scaler.inverse_transform(np.array([action]))
        action = action.squeeze()
        translation, yaw_rotation = self._clip_translation_vector(action[:3], action[3])
        open_close = action[4]

        if open_close > 0. and not self._gripper_open:
            self.openGripper()
            self._gripper_open = True
        elif open_close < 0. and self._gripper_open:
            self.closeGripper()
            self._gripper_open = False
        
        #print("Commanded translation: ", translation)
        #print("Commanded translation magnitude: ", np.linalg.norm(translation))
        #print("Commanded yaw: ", yaw_rotation)
        #print(" ")
        
        pos, orn, _, _, _, _ = self._physicsClient.getLinkState(self.model_id, 3)
        #print("Current Pos: ", pos)
        #print("Current Orn: ", orn)
        #print(" ")
        _, _, yaw = transformations.euler_from_quaternion(orn)
        #print(yaw)
        #Calculate transformation matrices
        T_world_old = transformations.compose_matrix(
            angles=[np.pi, 0., yaw], translate=pos)
        T_old_to_new = transformations.compose_matrix(
            angles=[0., 0., yaw_rotation], translate=translation)
        T_world_new = np.dot(T_world_old, T_old_to_new)
        self.endEffectorAngle += yaw_rotation
        target_pos, target_orn = transform_utils.to_pose(T_world_new)
        #print("Target Pos: ", str(np.linalg.norm(target_pos)))
        #print("Target Pos: ", target_pos)
        #print("Target Orn: ", target_orn)
        target_pos[1] *= -1
        target_pos[2] = -1 * (target_pos[2] - self._initial_height)
        #print(self._initial_height)
        yaw = self.endEffectorAngle
        comp_pos = np.r_[target_pos, yaw]

        self._physicsClient.setJointMotorControl2(
            self.model_id, 0,
            controlMode=p.POSITION_CONTROL,
            targetPosition=comp_pos[0],
            force=100.)
        #print("Commanding joint number 0 to position", comp_pos[0]) 
        self._physicsClient.setJointMotorControl2(
            self.model_id, 1,
            controlMode=p.POSITION_CONTROL,
            targetPosition=comp_pos[1],
            force=100.)
        #print("Commanding joint number 1 to position", comp_pos[1]) 
        self._physicsClient.setJointMotorControl2(
            self.model_id, 2,
            controlMode=p.POSITION_CONTROL,
            targetPosition=comp_pos[2],
            force=100.)
        #print("Commanding joint number 2 to position", comp_pos[2]) 
        self._physicsClient.setJointMotorControl2(
            self.model_id, 3,
            controlMode=p.POSITION_CONTROL,
            targetPosition=comp_pos[3],
            force=100.)
        #print("Commanding joint number 3 to position", comp_pos[3]) 
        #print(" ")

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

    """
    def step(self, action):
        high = np.r_[[self._max_translation]
                         * 3, self._max_yaw_rotation, 1.]
        self._action_scaler = MinMaxScaler((-1, 1))
        self._action_scaler.fit(np.vstack((-1. * high, high)))
        action = self._action_scaler.inverse_transform(np.array([action]))
        action = action.squeeze()
        translation, yaw_rotation = self._clip_translation_vector(action[:3], action[3])
        open_close = action[4]

        if open_close > 0. and not self._gripper_open:
            self.openGripper()
            self._gripper_open = True
        elif open_close < 0. and self._gripper_open:
            self.closeGripper()
            self._gripper_open = False

        #take translations & add to existing pos of gripper

        gripperPosX = self._physicsClient.getJointState(self.model_id, 0)[0]
        gripperPosY = self._physicsClient.getJointState(self.model_id, 1)[0]
        gripperPosZ = self._physicsClient.getJointState(self.model_id, 2)[0]
        gripperYaw = self._physicsClient.getJointState(self.model_id, 3)[0]

        dx = translation[0]
        dy = translation[1]
        dz = translation[2]
        dYaw = yaw_rotation

        self._physicsClient.setJointMotorControl2(
            self.model_id, 0,
            controlMode=p.POSITION_CONTROL,
            targetPosition=gripperPosX+dx,
            force=100.)
        self._physicsClient.setJointMotorControl2(
            self.model_id, 1,
            controlMode=p.POSITION_CONTROL,
            targetPosition=gripperPosY+dy,
            force=100.)
        self._physicsClient.setJointMotorControl2(
            self.model_id, 2,
            controlMode=p.POSITION_CONTROL,
            targetPosition=gripperPosZ+dz,
            force=100.)
        self._physicsClient.setJointMotorControl2(
            self.model_id, 3,
            controlMode=p.POSITION_CONTROL,
            targetPosition=gripperYaw+dYaw,
            force=100.)

        self._physicsClient.stepSimulation()
        self._envStepCounter += 1
        self._observation = self.getObservation()
        done = self.getTerminated() 
        reward = self.getReward() 

        if(self._envStepCounter > self._maxSteps):
            info = {"TimeLimit.truncated" : True}
        else:
            info = {}

        if(done):
            print(reward)

        return self._observation, reward, done, info
    """
    def get_gripper_width(self):
        left_joint_state = self._physicsClient.getJointState(
            self.model_id, 7)[0]
        right_joint_state = self._physicsClient.getJointState(
            self.model_id, 9)[0]
        left_finger_pos = 0.05 - left_joint_state
        right_finger_pos = 0.05 - right_joint_state

        return left_finger_pos + right_finger_pos
    
    def getReward(self): 
        position, _, _, _, _ , _ = self._physicsClient.getLinkState(self.model_id, 3)
        robot_height = position[2]
        tol = 0.005
        if(self._target_joint_pos == 0.05 and self.get_gripper_width() > tol):
            if not self._lifting:
                self._start_height = robot_height
                self._lifting = True
            if robot_height - self._start_height > 0:
                self.terminated = True
                print("Reward!")
                return 1.
        else:
            self._lifting = False
        
        return -0.01

    """
        #evaluates block position & assigns reward if block height is correct
        rewardFlag = False
        for i in range(len(self._blocks)):
            if(self._physicsClient.getBasePositionAndOrientation(self._blocks[i])[0][2] > -0.15):
                #print("Block at " + str(i) + " is " + str(self._physicsClient.getBasePositionAndOrientation(self._blocks[i])[0][2]))
                rewardFlag = True
            else:
                #print("Block at " + str(i) + " is " + str(self._physicsClient.getBasePositionAndOrientation(self._blocks[i])[0][2]))
                rewardFlag = False
        if (rewardFlag):
            #print("REWARD!!")
            return 1
        else:
            #print("NO Reward :( !")
            return -0.1
    """

env = SimplifiedWorld(renders = False)
env = make_vec_env(lambda: env, n_envs=1)
env = VecNormalize(env)
model = SAC("CnnPolicy", env, verbose=2, seed = 0, buffer_size = 100000, batch_size = 64, device = 'cpu')
model.learn(total_timesteps=1000000, log_interval=4)

#UNIT TEST - ensure multiple random initialization result in block position/orientation congruent with Baris thesis
#UNIT TEST - ability for gripper to translate rotate around the environment without camera becoming imporperly detached 
#UNIT TEST - ability for rotational values to make sense when beyond +pi or -pi
