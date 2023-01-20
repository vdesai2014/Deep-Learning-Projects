import pybullet as py
import random
import time 
import math 
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np 
from math import sqrt 
import pybullet_data
import os 
from PIL import Image 
import torch 
import stable_baselines3
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import VecCheckNan
from stable_baselines3.sac.policies import CnnPolicy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import SAC
from stable_baselines3 import PPO
from matplotlib import pyplot as plt
import collections
from typing import Any
from typing import Dict
from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecNormalize
import pybullet_utils.bullet_client as bc

class MLPEnv(gym.Env):
    metadata = {'render.modes': ['console']}
    max_steps_one_episode = 1000

    def __init__(self, is_render=False, is_good_view=False):
        super(MLPEnv, self).__init__()
        self.camera_parameters = {
            'width': 960.,
            'height': 720,
            'fov': 60,
            'near': 0.1,
            'far': 100.,
            'eye_position': [0.59, 0, 0.8],
            'target_position': [0.55, 0, 0.05],
            'camera_up_vector':
            [1, 0, 0],  # I really do not know the parameter's effect.
            'light_direction': [
                0.5, 0, 1
            ],  #the direction is from the light source position to the origin of the world frame.
        }
        self.device = torch.device(
            "cuda")

        self.view_matrix = py.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=[0.55, 0, 0.05],
            distance=.7,
            yaw=90,
            pitch=-70,
            roll=0,
            upAxisIndex=2)

        self.projection_matrix = py.computeProjectionMatrixFOV(
            fov=self.camera_parameters['fov'],
            aspect=self.camera_parameters['width'] /
            self.camera_parameters['height'],
            nearVal=self.camera_parameters['near'],
            farVal=self.camera_parameters['far'])

        self.is_render = is_render
        self.is_good_view = is_good_view

        if(is_render):
            self.p = bc.BulletClient(connection_mode=py.GUI)
        else:
            self.p = bc.BulletClient(connection_mode=py.DIRECT)

        self.x_low_obs = 0.2
        self.x_high_obs = 0.7
        self.y_low_obs = -0.3
        self.y_high_obs = 0.3
        self.z_low_obs = 0
        self.z_high_obs = 0.6

        self.x_low_action = -0.4
        self.x_high_action = 0.4
        self.y_low_action = -0.4
        self.y_high_action = 0.4
        self.z_low_action = -0.6
        self.z_high_action = 0.3

        self.p.configureDebugVisualizer(lightPosition=[5, 0, 5])
        self.p.resetDebugVisualizerCamera(cameraDistance=1.5,
                                     cameraYaw=0,
                                     cameraPitch=-40,
                                     cameraTargetPosition=[0.55, -0.35, 0.2])

        self.action_space = spaces.Box(low=np.array(
            [self.x_low_action, self.y_low_action, self.z_low_action]),
                                       high=np.array([
                                           self.x_high_action,
                                           self.y_high_action,
                                           self.z_high_action
                                       ]),
                                       dtype=np.float32)

        self.observation_space = spaces.Box(
            low=np.array([self.x_low_obs, self.y_low_obs, self.z_low_obs, self.x_low_obs, self.y_low_obs, self.z_low_obs]),
            high=np.array([self.x_high_obs, self.y_high_obs, self.z_high_obs, self.x_low_obs, self.y_low_obs, self.z_low_obs]),
            dtype=np.float32)

        self.step_counter = 0

        self.urdf_root_path = pybullet_data.getDataPath()
        # lower limits for null space
        self.lower_limits = [-.967, -2, -2.96, 0.19, -2.96, -2.09, -3.05]
        # upper limits for null space
        self.upper_limits = [.967, 2, 2.96, 2.29, 2.96, 2.09, 3.05]
        # joint ranges for null space
        self.joint_ranges = [5.8, 4, 5.8, 4, 5.8, 4, 6]
        # restposes for null space
        self.rest_poses = [0, 0, 0, 0.5 * math.pi, 0, -math.pi * 0.5 * 0.66, 0]
        # joint damping coefficents
        self.joint_damping = [
            0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001
        ]

        self.init_joint_positions = [
            0.006418, 0.413184, -0.011401, -1.589317, 0.005379, 1.137684,
            -0.006539
        ]

        self.orientation = py.getQuaternionFromEuler(
            [0., -math.pi, math.pi / 2.])

        self.seed()
        self.reset()
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.p.resetSimulation()
        self.step_counter = 0
        self.terminated = False
        self.p.setGravity(0, 0, -10)
        self.p.loadURDF(os.path.join(self.urdf_root_path, "plane.urdf"),
                   basePosition=[0, 0, -0.65])
        self.kuka_id = self.p.loadURDF(os.path.join(self.urdf_root_path,
                                               "kuka_iiwa/model.urdf"),
                                  useFixedBase=True)
        table_uid = self.p.loadURDF(os.path.join(self.urdf_root_path,
                                            "table/table.urdf"),
                               basePosition=[0.5, 0, -0.65])
        self.p.changeVisualShape(table_uid, -1, rgbaColor=[1, 1, 1, 1])

        self.object_id = self.p.loadURDF(os.path.join(self.urdf_root_path,
                                                 "random_urdfs/000/000.urdf"),
                                    basePosition=[
                                        random.uniform(self.x_low_obs,
                                                       self.x_high_obs),
                                        random.uniform(self.y_low_obs,
                                                       self.y_high_obs), 0.01
                                    ])

        self.num_joints = self.p.getNumJoints(self.kuka_id)

        for i in range(self.num_joints):
            self.p.resetJointState(
                bodyUniqueId=self.kuka_id,
                jointIndex=i,
                targetValue=self.init_joint_positions[i],
            )

        self.robot_pos_obs = self.p.getLinkState(self.kuka_id,
                                            self.num_joints - 1)[4]

        self.p.stepSimulation()

        self.robot_state = self.p.getLinkState(self.kuka_id, self.num_joints - 1)[4]
        self.object_state = np.array(self.p.getBasePositionAndOrientation(self.object_id)[0]).astype(np.float32)
        observation = [self.robot_state[0], self.robot_state[1], self.robot_state[2],  self.object_state[0], self.object_state[1], self.object_state[2]]
        observation = np.array(observation)

        return observation

    def close(self):
        self.p.__del__()

    def step(self, action):
        dv = 0.005
        dx = action[0] * dv
        dy = action[1] * dv
        dz = action[2] * dv

        self.current_pos = self.p.getLinkState(self.kuka_id, self.num_joints - 1)[4]
        self.new_robot_pos = [self.current_pos[0] + dx, self.current_pos[1] + dy, self.current_pos[2] + dz]
        self.robot_joint_positions = self.p.calculateInverseKinematics(
            bodyUniqueId=self.kuka_id,
            endEffectorLinkIndex=self.num_joints - 1,
            targetPosition=[
                self.new_robot_pos[0], self.new_robot_pos[1],
                self.new_robot_pos[2]
            ],
            targetOrientation=self.orientation,
            jointDamping=self.joint_damping,
        )
        for i in range(self.num_joints):
            self.p.resetJointState(
                bodyUniqueId=self.kuka_id,
                jointIndex=i,
                targetValue=self.robot_joint_positions[i],
            )
        self.p.stepSimulation()
        self.step_counter += 1

        return self._reward()
    
    def _reward(self):
        self.robot_state = self.p.getLinkState(self.kuka_id, self.num_joints - 1)[4]
        self.object_state = np.array(self.p.getBasePositionAndOrientation(self.object_id)[0]).astype(np.float32)
        observation = [self.robot_state[0], self.robot_state[1], self.robot_state[2],  self.object_state[0], self.object_state[1], self.object_state[2]]
        observation = np.array(observation)

        square_dx = (self.robot_state[0] - self.object_state[0])**2
        square_dy = (self.robot_state[1] - self.object_state[1])**2
        square_dz = (self.robot_state[2] - self.object_state[2])**2

        self.distance = sqrt(square_dx + square_dy + square_dz)

        x = self.robot_state[0]
        y = self.robot_state[1]
        z = self.robot_state[2]
        terminated = bool(x < self.x_low_obs or x > self.x_high_obs
                          or y < self.y_low_obs or y > self.y_high_obs
                          or z < self.z_low_obs or z > self.z_high_obs)

        endPremature = False 

        if terminated:
            reward = -0.1   
            self.terminated = True
            print("Episode ended due to end-effector violating play-space boundaries. Final distance was: ", self.distance)

        elif self.step_counter > self.max_steps_one_episode:
            reward = -0.1
            self.terminated = True
            endPremature = True
            print("Episode ended due to agent reaching 1000 timesteps without achieving goal. Final distance was: ", self.distance)

        elif self.distance < 0.1:
            reward = 1
            self.terminated = True
            print("Episode successful! Final distance was: ", self.distance)
            print("Object location was: ", self.object_state)
            #time.sleep(2)
        else:
            reward = 0
            self.terminated = False

        if(endPremature):
            info = {'distance:': self.distance, 'TimeLimit.truncated': True}
        else:
            info = {'distance:': self.distance}

        return observation, reward, self.terminated, info

env = MLPEnv(is_render=False)
model = SAC("MlpPolicy", env, buffer_size = 1000000, learning_rate = 0.0003, device = 'cuda', tensorboard_log="./logs/")
model.learn(total_timesteps=10000000, log_interval = 1)
