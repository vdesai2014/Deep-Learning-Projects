import time
import math
import random
import numpy as np
import pybullet as p
import pybullet_data
import gym
from gym import spaces
from collections import namedtuple
from attrdict import AttrDict
from tqdm import tqdm
from stable_baselines3.sac.policies import CnnPolicy
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import VecNormalize
#from stable_baselines.common.vec_env.base_vec_env import VecEnv, VecEnvWrapper
#from stable_baselines.common.cmd_util import make_vec_env
from stable_baselines3.common.env_util import make_vec_env
import transform_utils
import transformations
from robot import UR5Robotiq85, UR5Robotiq140
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from custom_obs_policy import CustomCNN
import torch

class FullArmRL(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 50}

    def __init__(self, renders = False, policy = "CnnPolicy"):
        self._near = 0.02
        self._far = 2
        self._endofEpisodeStep = 78
        self._timeStep = 1./240
        self._urdfRoot = pybullet_data.getDataPath()
        self._observation = []
        self._envStepCounter = 0
        self._renders = renders
        self._width = 64
        self._height = 64
        self.terminated = False 
        self.extent = 0.1
        self._max_translation = 0.004 
        self._max_yaw_rotation = 0.15 
        self._initial_height = 0.2
        self.policy = policy
        camRotation = p.getQuaternionFromEuler([1.157, 0, 1.571])
        transform_dict = {'translation' : [0, 0, -0.1], 'rotation' : [camRotation[0], camRotation[1], camRotation[2], camRotation[3]]} 
        self._transform = transform_utils.from_dict(transform_dict) 

        self.physicsClient = p.connect(p.GUI if self._renders else p.DIRECT)
        p.resetDebugVisualizerCamera(1.3, 180, -41, [0.52, -0.2, -0.33])
        if (policy == "CnnPolicy"):
            self.observation_space = spaces.Box(low=0,
                                            high=255,
                                            shape=(2, 64, 64))
            self.action_space = spaces.Box(-1.,1., shape=(3,), dtype=np.float32)
        elif (policy == "MlpPolicy"):
            self.observation_space = spaces.Box(low=-1.0,
                                            high=1.0,
                                            shape=(3,), dtype = np.float32)
            self.action_space = spaces.Box(-1.,1., shape=(3,), dtype=np.float32)
        self.reset() 

    def reset(self):
        self.terminated = False
        self._gripper_open = True
        self.endEffectorAngle = 0.
        p.resetSimulation()
        p.setPhysicsEngineParameter(numSolverIterations=150, fixedTimeStep = 1./240.)
        p.setGravity(0, 0, -10)
        self._envStepCounter = 0
        self._lifting = False
        self.lift_dist = 0.01
    
        #Load white floor
        plane = p.loadURDF("/home/vrushank/Documents/GitHub/Deep-Learning-Projects/Object Grasping with 6-DoF Robot Arm/FullArmRL/plane/planeRL.urdf", [0., 0., -0.1], [0., 0., 0., 1.])
        
        self._blocks = []
        if (self.policy == "CnnPolicy"):
            numBlocks = 3
            for i in range (numBlocks):
                randBlock = np.random.randint(100, 999)
                path = self._urdfRoot + '/random_urdfs/' + str(randBlock) + '/' + str(randBlock) + '.urdf'
                position = np.r_[np.random.uniform(-self.extent, self.extent), np.random.uniform(-self.extent, self.extent), 0]
                orientation = transformations.random_quaternion()
                block = p.loadURDF(path, position, orientation, globalScaling = 0.7)
                self._blocks.append(block)
        elif (self.policy == "MlpPolicy"):
            randBlock = 103
            path = self._urdfRoot + '/random_urdfs/' + str(randBlock) + '/' + str(randBlock) + '.urdf'
            position = np.r_[np.random.uniform(-self.extent, self.extent), np.random.uniform(-self.extent, self.extent), 0]
            orientation = p.getQuaternionFromEuler((0,np.pi/2,0))
            block = p.loadURDF(path, position, orientation, globalScaling = 0.7)
            self._blocks.append(block)
        
        for i in range(1000):
                p.stepSimulation()

        self.currentX = 0
        self.currentY = 0
        self.currentZ = self._initial_height
        self.currentYaw = 1.571
        self.robot = UR5Robotiq85((0, 0.5, 0), (0, 0, 0))
        self.robot.load()
        self.robot.reset([self.currentX, self.currentY, self._initial_height], self.currentYaw)


        if(self.policy == "CnnPolicy"):
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
        
        """

        gripper_opening_length_control = p.addUserDebugParameter("gripper_opening_length",
                                                    0,
                                                    0.085,
                                                    0.085)
        self.position_control_group = []
        self.position_control_group.append(p.addUserDebugParameter('x', -0.5, 0.5, 0))
        self.position_control_group.append(p.addUserDebugParameter('y', -0.5, 0.5, 0))
        self.position_control_group.append(p.addUserDebugParameter('z', -0.25, 1, 0.15))
        self.position_control_group.append(p.addUserDebugParameter('roll', -3.14, 3.14, 0))
        self.position_control_group.append(p.addUserDebugParameter('pitch', 0, 3.14, 1.57))
        self.position_control_group.append(p.addUserDebugParameter('yaw', -3.14, 3.14, 0))
        self.position_control_group.append(p.addUserDebugParameter('CamX', -0.5, 0.5, 0)) #6
        self.position_control_group.append(p.addUserDebugParameter('CamY', -0.5, 0.5, 0)) #7
        self.position_control_group.append(p.addUserDebugParameter('CamZ', -0.5, 0.5, 0)) #8
        self.position_control_group.append(p.addUserDebugParameter('CamRoll', -3.14, 3.14, 0)) #9
        self.position_control_group.append(p.addUserDebugParameter('CamPitch', -3.14, 3.14, 0)) #10
        self.position_control_group.append(p.addUserDebugParameter('CamYaw', -3.14, 3.14, 0)) #11

        p.addUserDebugLine((0.03, 0, 0.15), (0.03, 0, -0.1), lineWidth = 4.0, lifeTime = 0)
        """

        obs = self.getObservation()
        return obs

    def __del__(self):
        p.disconnect()
    
    def getObservation(self):
        if(self.policy == "CnnPolicy"):
            pos, orn, _, _, _, _ = p.getLinkState(self.robot.getRobotID(), 7)
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
            _, _, _, depthImg, _ = p.getCameraImage(
                width=self._width, 
                height=self._height,
                viewMatrix=gl_view_matrix,
                projectionMatrix=projectionMatrix)
            near, far = self._near, self._far
            depth_buffer = np.asarray(depthImg, np.float32).reshape(
                (64, 64))
            depth = (1. * far * near / (far - (far - near) * depth_buffer)) * 255
            obs = np.expand_dims(depth, 0)
            obs = obs.astype(np.uint8)
            obs = np.zeros((2, 64, 64))
        elif(self.policy == "MlpPolicy"):
            blockX, blockY, blockZ = p.getBasePositionAndOrientation(self._blocks[0])[0]
            obs = np.r_[blockX, blockY, blockZ]
        return obs
    
    def _clip_translation_vector(self, translation, yaw):
        """Clip the vector if its norm exceeds the maximal allowed length."""
        length = np.linalg.norm(translation)
        if length > self._max_translation:
            translation *= self._max_translation / length
        if yaw > self._max_yaw_rotation:
            yaw *= self._max_yaw_rotation / yaw
        return translation, yaw
    
    def trimActions(self, action):
        high = np.r_[[self._max_translation]
                         * 1, self._max_yaw_rotation, 1.]
        self._action_scaler = MinMaxScaler((-1, 1))
        self._action_scaler.fit(np.vstack((-1. * high, high)))
        action = self._action_scaler.inverse_transform(np.array([action]))
        action = action.squeeze()
        return action

    def endOfEpisode(self):
        self.robot.move_gripper(0)
        reward = -0.0128
        for i in range(75):
            self.currentZ += 0.00166666667
            self.robot.move_ee([self.currentX, self.currentY, self.currentZ], self.currentYaw)
            p.stepSimulation()
        for block in self._blocks:
            if (p.getBasePositionAndOrientation(block)[0][2] > 0):
                reward = 1
                print("Yay I did it!")
            else:
                reward = -0.1
        
        return reward
    
    def step(self,action):
        action = self.trimActions(action)
        translation, yaw_rotation = self._clip_translation_vector(action[:2], action[2])

        self.newX = self.currentX + translation[0]
        self.newY = self.currentY + translation[1]
        self.newZ = self.currentZ - (0.000833333*2)
        
        if(self.policy == "CnnPolicy"):
            self.newYaw = self.currentYaw - yaw_rotation
        elif(self.policy == "MlpPolicy"):
            self.newYaw = 1.571
        
        self.robot.move_ee([self.newX, self.newY, self.newZ], self.newYaw)

        self.currentX = self.newX
        self.currentY = self.newY
        self.currentZ = self.newZ
        self.currentYaw = self.newYaw

        p.stepSimulation()
        self._envStepCounter += 1
        obs = self.getObservation()

        if(self._envStepCounter == self._endofEpisodeStep):
            reward = self.endOfEpisode()
            done = True
            info = {}
        else:
            reward = 0
            done = False 
            info = {}

        return obs, reward, done, {"is_success":reward==1, "episode_step": self._envStepCounter, "episode_rewards": reward}
    

    def debugStep(self):
        parameter = []
        for i in range(12):
            parameter.append(p.readUserDebugParameter(self.position_control_group[i]))
        self.newX = parameter[0]
        self.newY = parameter[1]
        self.newZ = parameter[2]
        self.newYaw = parameter[5]
        translation = []
        translation.append(parameter[6])
        translation.append(parameter[7])
        translation.append(parameter[8])
        rotation = []
        rotation.append(parameter[9])
        rotation.append(parameter[10])
        rotation.append(parameter[11])
        quat = p.getQuaternionFromEuler(rotation)
        self.getObservation(translation, quat)
        self.robot.move_ee([self.newX, self.newY, self.newZ], self.newYaw)
        p.stepSimulation()


    def getReward(self):
        return 


policy_kwargs = dict(
    features_extractor_class=CustomCNN
)
env = FullArmRL(renders = True, policy = "CnnPolicy")
env = make_vec_env(lambda: env, n_envs=1)
env = VecNormalize(env)
for i in range(30):
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
model = SAC("CnnPolicy", env, verbose=2, seed = 0, policy_kwargs=policy_kwargs, buffer_size = 100000, batch_size = 64, learning_rate = 0.0003, tensorboard_log="./logs/", device = 'cuda')
model.learn(total_timesteps=1000000, log_interval=4)
"""
env = FullArmRL(renders = True)
while(True):
    env.step()
"""