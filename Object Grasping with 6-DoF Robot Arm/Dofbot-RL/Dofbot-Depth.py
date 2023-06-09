import time
import math
import random
import numpy as np
import pybullet as p
import pybullet_data
import gym
from gym import spaces
from stable_baselines3.sac.policies import CnnPolicy
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3 import PPO
import pybullet_utils.bullet_client as bc
import os
import transform_utils
import transformations
import collections

class DofbotRL(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 50}

    def __init__(self, renders=False):
        self._renders = renders
        self.endofEpisodeStep = 100
        self.urdfRootPath = pybullet_data.getDataPath()
        self.action_space = spaces.Box(-1.,1., shape=(2,), dtype=np.float32)
        self.initial_height = 0.4
        self.height = 64
        self.width = 64
        self.observation_space = spaces.Box(low=0,
                                        high=255,
                                        shape=(1, self.height, self.width), dtype=np.float32)
        if self._renders:
            self.physicsClient = bc.BulletClient(connection_mode = p.GUI)
        else: 
            self.physicsClient = bc.BulletClient(connection_mode = p.DIRECT)
        RotX = 0.5
        RotY = 0
        RotZ = 0
        TransX = 0
        TransY = 0.065
        TransZ = 0
        camRotation = self.physicsClient.getQuaternionFromEuler([RotX, RotY, RotZ])
        transform_dict = {'translation' : [TransX, TransY, TransZ], 'rotation' : [camRotation[0], camRotation[1], camRotation[2], camRotation[3]]} 
        transform = transform_utils.from_dict(transform_dict)
        self._h_robot_camera = transform
        self.episode_queue = collections.deque(maxlen=10)	
        self.boxYlower = 0.125	
        self.boxYupper = 0.125	
        self.difficultyLevel = 0 	
        self.episodeCount = 0
        self.reset() 

    def reset(self):
        self.terminated = False
        self.envStepCounter = 0
        self.physicsClient.resetSimulation()
        self.physicsClient.setGravity(0, 0, -9.8)

        #Load white floor
        plane_path = os.path.join(os.path.dirname(__file__), 'plane/planeRL.urdf')
        plane = self.physicsClient.loadURDF(plane_path, [0., 0., 0], [0., 0., 0., 1.])

        dofbot_path = os.path.join(os.path.dirname(__file__), 'fiverrArm.urdf')
        box_path = os.path.join(os.path.dirname(__file__), 'box.urdf')
        baseOrn = self.physicsClient.getQuaternionFromEuler([0, -np.pi, 0])
        self.id = self.physicsClient.loadURDF(dofbot_path, basePosition=[0,0,self.initial_height], baseOrientation = baseOrn, useFixedBase=True)
        self.boxX = random.uniform(-0.05, 0.05)
        if(self.boxYlower > 0.08):	
            if(self.curriculumMonitor()):
                self.difficultyLevel += 1
        self.boxY = random.uniform(self.boxYlower, self.boxYupper)
        self.box = self.physicsClient.loadURDF(box_path, basePosition=[self.boxX, self.boxY, 0.05], baseOrientation = baseOrn)
        self.physicsClient.changeDynamics(self.box, -1, mass=1.0, lateralFriction=2.0)
        self.physicsClient.changeVisualShape(self.id, -1, rgbaColor=[0,0,0,1])
        self.physicsClient.changeVisualShape(self.id, 0, rgbaColor=[0,1,0,1])
        self.physicsClient.changeVisualShape(self.id, 1, rgbaColor=[1,1,0,1])
        self.physicsClient.changeVisualShape(self.id, 2, rgbaColor=[0,1,0,1])
        self.physicsClient.changeVisualShape(self.id, 3, rgbaColor=[1,1,0,1])
        self.physicsClient.changeVisualShape(self.id, 4, rgbaColor=[0,0,0,1])
        self.physicsClient.changeVisualShape(self.box, -1, rgbaColor=[0,0,0,1])

        for i in range(100):
            self.physicsClient.stepSimulation()

        initialX = 0 
        initialY = 0.164
        initialZ = 0.2212
        self.roll = 0
        self.pitch = np.pi
        self.yaw = np.pi
        initialGripper = -1.57
        orn = self.physicsClient.getQuaternionFromEuler([self.roll, self.pitch, self.yaw])
        jointPoses = self.physicsClient.calculateInverseKinematics(self.id, 4, [initialX, initialY, initialZ], orn, maxNumIterations=1000,
                                                    residualThreshold=.01)
        for i in range(5):
            self.physicsClient.resetJointState(self.id, i, jointPoses[i])
        self.physicsClient.resetJointState(self.id, 5, initialGripper)
        self.physicsClient.resetJointState(self.id, 6, -initialGripper)
        self.physicsClient.resetJointState(self.id, 7, initialGripper)
        self.physicsClient.resetJointState(self.id, 8, -initialGripper)
        self.physicsClient.resetJointState(self.id, 9, -initialGripper)
        self.physicsClient.resetJointState(self.id, 10, -initialGripper)
        self.physicsClient.resetJointState(self.id, 11, -initialGripper)
        self.physicsClient.resetJointState(self.id, 12, -initialGripper)
        self.physicsClient.stepSimulation()

        self.currentX = self.physicsClient.getLinkState(self.id, 4)[0][0]
        self.currentY = self.physicsClient.getLinkState(self.id, 4)[0][1]
        self.currentZ = self.physicsClient.getLinkState(self.id, 4)[0][2]

        obs = self.getObservation()
        return obs
	
    def curriculumMonitor(self):	
        if(len(self.episode_queue) == 10):	
            if(sum(self.episode_queue)>7):	
                self.boxYlower -= 0.0125	
                for i in range(10):	
                    self.episode_queue.appendleft(0)	
                return True	
        return False

    def __del__(self):
        self.physicsClient.disconnect()

    def getObservation(self):
        pos, orn, _, _, _, _ = self.physicsClient.getLinkState(self.id, 4)
        h_world_robot = transform_utils.from_pose(pos, orn)
        h_camera_world = np.linalg.inv(np.dot(h_world_robot, self._h_robot_camera))
        gl_view_matrix = h_camera_world.copy()
        gl_view_matrix[2, :] *= -1 # flip the Z axis to comply to OpenGL
        gl_view_matrix = gl_view_matrix.flatten(order='F')
        projectionMatrix = self.physicsClient.computeProjectionMatrixFOV(
            fov=45.0,
            aspect=1.0,
            nearVal=0.02,
            farVal=3.0)
        _, _, _, depthImg, _ = self.physicsClient.getCameraImage(
            width=64, 
            height=64,
            viewMatrix=gl_view_matrix,
            projectionMatrix=projectionMatrix)
        near, far = 0.02, 2
        depth_buffer = np.asarray(depthImg, np.float32).reshape(
                (64, 64))
        depth = (1. * far * near / (far - (far - near) * depth_buffer))
        expandedDepthMatrix = np.expand_dims(depth, axis = 0)
        return expandedDepthMatrix

    def endOfEpisode(self):
        gripperPosition = -0.3
        for i in range(100): #optimize length of for loop to minimize wasted cpu cycles
            self.physicsClient.setJointMotorControl2(self.id, 5, self.physicsClient.POSITION_CONTROL, targetPosition = gripperPosition)
            self.physicsClient.setJointMotorControl2(self.id, 6, self.physicsClient.POSITION_CONTROL, targetPosition = -gripperPosition)
            self.physicsClient.setJointMotorControl2(self.id, 7, self.physicsClient.POSITION_CONTROL, targetPosition = gripperPosition)
            self.physicsClient.setJointMotorControl2(self.id, 8, self.physicsClient.POSITION_CONTROL, targetPosition = -gripperPosition)
            self.physicsClient.setJointMotorControl2(self.id, 9, self.physicsClient.POSITION_CONTROL, targetPosition = -gripperPosition)
            self.physicsClient.setJointMotorControl2(self.id, 10, self.physicsClient.POSITION_CONTROL, targetPosition = -gripperPosition)
            self.physicsClient.setJointMotorControl2(self.id, 11, self.physicsClient.POSITION_CONTROL, targetPosition = -gripperPosition)
            self.physicsClient.setJointMotorControl2(self.id, 12, self.physicsClient.POSITION_CONTROL, targetPosition = -gripperPosition)
            self.physicsClient.stepSimulation()

        orn = self.physicsClient.getQuaternionFromEuler([self.roll, self.pitch, self.yaw])
        jointPoses = self.physicsClient.calculateInverseKinematics(self.id, 4, [self.newX, self.newY, (self.newZ+0.2)], orn, maxNumIterations=1000,
                                                  residualThreshold=.01)
        for i in range(35): #optimize length of for loop to minimize wasted cpu cycles
            for j in range(5):
                self.physicsClient.setJointMotorControl2(self.id, j, self.physicsClient.POSITION_CONTROL, targetPosition = jointPoses[j], maxVelocity = 3.0)
            self.physicsClient.stepSimulation()

        if(self.physicsClient.getBasePositionAndOrientation(self.box)[0][2] > 0.05):
            reward = 1
        else:
            finalXCoordinate = self.physicsClient.getLinkState(self.id, 4)[0][0]
            finalYCoordiante = self.physicsClient.getLinkState(self.id, 4)[0][1]
            desiredXCoordinate = self.boxX
            desiredYCoordinate = 0.125
            distanceError = math.sqrt(((finalXCoordinate-desiredXCoordinate)**2)+((finalYCoordiante-desiredYCoordinate)**2))
            reward = -distanceError
        for i in range(5):
            print("This episode was a " + str(reward) + " and the difficulty level was " + str(self.difficultyLevel) + " and the box lower was: " + str(self.boxYlower))
        if(reward == 1):	
            self.episode_queue.appendleft(1)	
        else:	
            self.episode_queue.appendleft(0)
        return reward

    def step(self, action):
        self.newX = self.currentX + (action[0]*0.01) 
        self.newY = self.currentY + (action[1]*0.01)
        self.newZ = self.currentZ - 0.001
        orn = self.physicsClient.getQuaternionFromEuler([self.roll, self.pitch, self.yaw])
        jointPoses = self.physicsClient.calculateInverseKinematics(self.id, 4, [self.newX, self.newY, self.newZ], orn, maxNumIterations=1000,
                                                  residualThreshold=.01)
        for i in range(5):
            self.physicsClient.setJointMotorControl2(self.id, i, self.physicsClient.POSITION_CONTROL, targetPosition = jointPoses[i])
        self.physicsClient.stepSimulation()

        self.currentX = self.newX
        self.currentY = self.newY
        self.currentZ = self.newZ

        self.envStepCounter += 1
        obs = self.getObservation()

        if(self.envStepCounter == self.endofEpisodeStep):
            reward = self.endOfEpisode()
            done = True
            info = {}
        else:
            reward = 0
            done = False 
            info = {}
        return obs, reward, done, {"is_success":reward==1, "episode_step": self.envStepCounter, "episode_rewards": reward}

env = DofbotRL(False)
env = make_vec_env(lambda:env, n_envs=1)
env = VecNormalize(env, norm_obs = False)
model = SAC("CnnPolicy", env, verbose=2, seed = 0, batch_size = 128, learning_rate = 0.0003, buffer_size = 1000000, tensorboard_log="./logs/", device = 'cuda', policy_kwargs=dict(normalize_images=False))
model.learn(total_timesteps=1000000, log_interval=1)