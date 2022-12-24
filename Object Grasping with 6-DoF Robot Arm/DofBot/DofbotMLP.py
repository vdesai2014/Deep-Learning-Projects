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
import stable_baselines3
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.sac.policies import CnnPolicy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import EvalCallback
import pybullet_utils.bullet_client as bc

class MLPEnv(gym.Env):
    metadata = {'render.modes': ['console']}

    def __init__(self, is_render=False):
        super(MLPEnv, self).__init__()
        if(is_render):
            self.p = bc.BulletClient(connection_mode=py.GUI)
        else:
            self.p = bc.BulletClient(connection_mode=py.DIRECT)
        self.p.configureDebugVisualizer(lightPosition=[5, 0, 5])
        self.p.resetDebugVisualizerCamera(cameraDistance=1.5,
                                     cameraYaw=0,
                                     cameraPitch=-40,
                                     cameraTargetPosition=[0.55, -0.35, 0.2])
        
        self.action_space = spaces.Box(low = -1, high = 1, shape = (3,), dtype=np.float32)

        self.x_low_obs = -0.125
        self.x_high_obs = 0.125
        self.y_low_obs = 0.05
        self.y_high_obs = 0.175
        self.maxSteps = 500

        self.observation_space = spaces.Box(low = np.array([self.x_low_obs , self.x_high_obs]), high = np.array([self.y_low_obs, self.y_high_obs]), dtype=np.float32)
        
        self.baseOrientation = self.p.getQuaternionFromEuler([0., -np.pi, 0])
        
        self.urdf_root_path = pybullet_data.getDataPath()
        self.reset()
        self.seed()
    
    def seed(self, seed=0):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def reset(self):
        self.p.resetSimulation()
        self.step_counter = 0
        self.terminated = False
        self.p.setGravity(0, 0, -10)
        self.p.loadURDF(os.path.join(self.urdf_root_path, "plane.urdf"), basePosition=[0, 0, 0])
        dofbot_path = os.path.join(os.path.dirname(__file__), 'arm.urdf')
        self.armUid = self.p.loadURDF(dofbot_path, basePosition=[0,0,0.4], baseOrientation = self.baseOrientation, useFixedBase=True)
        self.ee_index = 4
        self.num_joints = self.p.getNumJoints(self.armUid)

        self.p.changeVisualShape(self.armUid, -1, rgbaColor=[0,0,0,1])
        self.p.changeVisualShape(self.armUid, 0, rgbaColor=[0,1,0,1])
        self.p.changeVisualShape(self.armUid, 1, rgbaColor=[1,1,0,1])
        self.p.changeVisualShape(self.armUid, 2, rgbaColor=[0,1,0,1])
        self.p.changeVisualShape(self.armUid, 3, rgbaColor=[1,1,0,1])
        self.p.changeVisualShape(self.armUid, 4, rgbaColor=[0,0,0,1])
        self.p.changeVisualShape(self.armUid, 5, rgbaColor=[0,1,0,0])
        self.p.changeVisualShape(self.armUid, 6, rgbaColor=[0,0,1,0])
        self.p.changeVisualShape(self.armUid, 7, rgbaColor=[1,0,0,0.5])

        self.goal_pos = np.array([random.uniform(self.x_low_obs, self.x_high_obs), random.uniform(self.y_low_obs, self.y_high_obs)]).astype(np.float32)
        self.p.addUserDebugLine([self.goal_pos[0], self.goal_pos[1], 0], [self.goal_pos[0], self.goal_pos[1], 1], lineColorRGB=[1,0,0], lineWidth = 1.0)

        self.resetOrn = self.p.getQuaternionFromEuler([0., np.pi, np.pi/2])
        self.resetPose = self.p.calculateInverseKinematics(self.armUid, 4, [0, 0.17, 0.23], self.resetOrn, maxNumIterations=1000,
                                                  residualThreshold=.01)
        for i in range(5):
            self.p.resetJointState(
                bodyUniqueId=self.armUid,
                jointIndex=i,
                targetValue=self.resetPose[i],
            )
        self.p.stepSimulation()
        print(self.goal_pos)
        return self.goal_pos

    def close(self):
        self.p.__del__()

    def step(self, action):
        dv = 0.05
        dx = action[0] * dv
        dy = action[1] * dv
        dz = action[2] * dv

        self.current_pos = self.p.getLinkState(self.armUid, 4)[0]
        self.new_robot_pos = [self.current_pos[0] + dx, self.current_pos[1] + dy, self.current_pos[2] + dz]
        self.robot_joint_positions = self.p.calculateInverseKinematics(
            bodyUniqueId=self.armUid,
            endEffectorLinkIndex=self.ee_index,
            targetPosition=[
                self.new_robot_pos[0], self.new_robot_pos[1],
                self.new_robot_pos[2]
            ], targetOrientation = self.resetOrn, maxNumIterations=1000, residualThreshold=.01
        )
        
        for i in range(5):
            """
            self.p.resetJointState(
                bodyUniqueId=self.armUid,
                jointIndex=i,
                targetValue=self.robot_joint_positions[i],
            )
            """
            self.p.setJointMotorControl2(self.armUid, i, self.p.POSITION_CONTROL, targetPosition = self.robot_joint_positions[i])
        self.p.stepSimulation()
        self.step_counter += 1

        return self.reward()
    
    def reward(self):
        self.robot_state = self.p.getLinkState(self.armUid, 7)[0]
        square_dx = (self.robot_state[0] - self.goal_pos[0])**2
        square_dy = (self.robot_state[1] - self.goal_pos[1])**2
        square_dz = (self.robot_state[2])**2

        self.distance = sqrt(square_dx + square_dy + square_dz)

        x = self.robot_state[0]
        y = self.robot_state[1]
        z = self.robot_state[2]
        terminated = bool(x < self.x_low_obs or x > self.x_high_obs
                          or y < self.y_low_obs or y > self.y_high_obs
                          or z < 0 or z > 0.25)
        endPremature = False

        if terminated:
            reward = -0.1   
            self.terminated = True
            print("Episode ended due to end-effector violating play-space boundaries. Final distance was: ", self.distance)
            print("X: ", x)
            print("Y: ", y)
            print("Z: ", z)

        elif self.step_counter > self.maxSteps:
            reward = -0.1
            self.terminated = True
            endPremature = True
            print("Episode ended due to agent reaching 1000 timesteps without achieving goal. Final distance was: ", self.distance)

        elif self.distance < 0.05:
            reward = 1
            self.terminated = True
            print("Episode successful! Final distance was: ", self.distance)
            print("Object location was: ", self.goal_pos)
        else:
            reward = 0
            self.terminated = False
        if(endPremature):
            info = {"is_success":reward==1, "episode_step": self.step_counter, 
            "episode_rewards": reward, 'distance:': 
            self.distance, 'TimeLimit.truncated': True}
        else:
            info = {"is_success":reward==1, "episode_step": self.step_counter, 
            "episode_rewards": reward, 'distance:': 
            self.distance}

        return self.goal_pos, reward, self.terminated, info

env = MLPEnv(is_render=False)
dummyVecEnv = make_vec_env(lambda: env, n_envs=1)
vecNorm = VecNormalize(dummyVecEnv)
evalEnv = MLPEnv()
dummyEvalEnv = make_vec_env(lambda: evalEnv, n_envs=1)
evalNorm = VecNormalize(dummyEvalEnv)
eval_callback = EvalCallback(evalNorm, best_model_save_path="./logs/",
                             log_path="./logs/", eval_freq=20000,
                             deterministic=True, render=False, n_eval_episodes=10)
tensorboard_path = os.path.join(os.path.dirname(__file__))
model = SAC("MlpPolicy", vecNorm, buffer_size = 1000000, learning_rate = 0.0003, device = 'cuda', tensorboard_log=tensorboard_path)
model.learn(total_timesteps=10000000, log_interval = 1, callback=eval_callback)