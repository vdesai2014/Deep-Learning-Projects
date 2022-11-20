import os, inspect
from turtle import speed
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)
import pybullet as p 
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
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.sac.policies import CnnPolicy
from stable_baselines3 import SAC
from matplotlib import pyplot as plt
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvWrapper
from stable_baselines3.common.env_util import make_vec_env
import transform_utils
import transformations

class SimplifiedWorld(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 50}

    def __init__(self, renders = False):
        self._maxSteps = 700
        self._timeStep = 1./240
        self._urdfRoot = pybullet_data.getDataPath()
        self._observation = []
        self._envStepCounter = 0
        self._renders = renders
        self._width = 64
        self._height = 64
        self.terminated = False 

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
                                        shape=(1, self._height, self._width),
                                        dtype=np.uint8)
        self.action_space = gym.spaces.Box(-1.,1., shape=(3,), dtype=np.float32)
    
    def reset(self):
        self.terminated = False
        self._physicsClient.resetSimulation()
        self._physicsClient.setPhysicsEngineParameter(numSolverIterations=150)
        self._physicsClient.setGravity(0, 0, -10)
        self._envStepCounter = 0

        #Load white floor
        plane = self._physicsClient.loadURDF("plane/planeRL.urdf", [0, 0, 0])

        #Load custom object @ random position (within bounds of simple env)
        squareBounds = 0.01
        xpos = squareBounds * random.random()
        ypos = squareBounds * random.random()
        ang = 3.1415925438 * random.random()
        orn = p.getQuaternionFromEuler([0, 0, ang])
        self.blockUid = p.loadURDF(os.path.join(self._urdfRoot, "block.urdf"), xpos, ypos, 0,
                               orn[0], orn[1], orn[2], orn[3])

        #load simple gripper, use 0,1,2 to control X, Y, Z pos, 3 to control gripper yaw, & 7/9 to close gripper
        start_pos = [0, 0, 0.4]
        start_orn = p.getQuaternionFromEuler([3.14, 0, 3.14]) #CONFIRM - random environment intialization is congruent with simple base environment outlined by Baris
        self.model_id = self._physicsClient.loadSDF("gripper/wsg50_one_motor_gripper_new.sdf")[0]
        self._physicsClient.resetBasePositionAndOrientation(self.model_id, start_pos, start_orn)
        self._physicsClient.stepSimulation()
        """
        # Randomize focal lengths fx and fy
        camera_info['K'][0] += np.random.uniform(-f, f)
        camera_info['K'][4] += np.random.uniform(-f, f)

        fx = 69.76 + np.np.random.uniform(-4, 4)
        fy = 77.25 + np.np.random.uniform(-4, 4)
        
        # Randomize optical center cx and cy
        camera_info['K'][2] += np.random.uniform(-c, c)
        camera_info['K'][5] += np.random.uniform(-c, c)

        cx = 32.19 
        cy = 32.0 
        """

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

        rgb = self.getObservation()
        return rgb

    def __del__(self):
        self._physicsClient.disconnect()
    
    def getObservation(self):
        """
        #take gripper pose, attach camera to gripper, return 64x64x1 depth numpy array
        gripperPos = self._physicsClient.getBasePositionAndOrientation(self.model_id)[0]
        gripperPosX = self._physicsClient.getJointState(self.model_id, 0)[0]
        gripperPosY = self._physicsClient.getJointState(self.model_id, 1)[0]
        gripperPosZ = self._physicsClient.getJointState(self.model_id, 2)[0] - 0.2
        yawAngle_x = math.sin(self._physicsClient.getJointState(self.model_id, 3)[0])
        yawAngle_y = math.cos(self._physicsClient.getJointState(self.model_id, 3)[0])
        
        viewMatrix = p.computeViewMatrix(                               
            cameraEyePosition=[-gripperPosX, gripperPosY, -gripperPosZ],
            cameraTargetPosition=[-gripperPosX, gripperPosY, 0],
            cameraUpVector=[yawAngle_x, yawAngle_y, 0])
        projectionMatrix = p.computeProjectionMatrixFOV(
            fov=45.0,
            aspect=1.0,
            nearVal=0.02,
            farVal=3.0)
        _, _, _, depthImg, _ = self._physicsClient.getCameraImage(
            width=self._width, 
            height=self._height,
            viewMatrix=viewMatrix,
            projectionMatrix=projectionMatrix)

        obs = np.array(depthImg)
        obs = obs[np.newaxis, :, :]
        """
        print(self._physicsClient.getLinkState(self.model_id, 3))
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
        plt.imshow(obs)
        plt.savefig(str(self._envStepCounter) + ".png")
        near, far = 0.2, 2

        obs = obs[np.newaxis, :, :]
        depth_buffer = np.asarray(depthImg, np.float32).reshape(
            (self._height, self._width))
        obs = 1. * far * near / (far - (far - near) * depth_buffer)
        plt.imshow(obs)
        plt.savefig(str(self._envStepCounter) + "modified.png")
        obs = obs[np.newaxis, :, :]
        return obs

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

    def raiseGripper(self):
        for i in range(1000):
            gripperPosZ = self._physicsClient.getJointState(self.model_id, 2)[0]
            dz = -0.001
            self._physicsClient.setJointMotorControl2(
            self.model_id, 2,
            controlMode=p.POSITION_CONTROL,
            targetPosition=(gripperPosZ + dz),
            force=200.)
            self._physicsClient.stepSimulation()
            if (self._physicsClient.getBasePositionAndOrientation(self.blockUid)[0][2] > 0.23):
                break

    def step(self,action):
        #takes in 3 floating point values (x, y, angle), steps actuator in that direction
        print(self._envStepCounter)
        speedConstant = 0.005
        dx = action[0] * speedConstant
        dy = action[1] * speedConstant
        dz = 0.001
        dA = action[2] * 2 * speedConstant #CONFIRM - speed constant for translation is reasonable 
        gripperPosX = self._physicsClient.getJointState(self.model_id, 0)[0]
        gripperPosY = self._physicsClient.getJointState(self.model_id, 1)[0]
        gripperPosZ = self._physicsClient.getJointState(self.model_id, 2)[0]
        gripperYaw = self._physicsClient.getJointState(self.model_id, 3)[0]

        self._physicsClient.setJointMotorControl2(
            self.model_id, 0,
            controlMode=p.POSITION_CONTROL,
            targetPosition=(gripperPosX + dx),
            force=100.)
        self._physicsClient.setJointMotorControl2(
            self.model_id, 1,
            controlMode=p.POSITION_CONTROL,
            targetPosition=(gripperPosY + dy),
            force=100.)
        self._physicsClient.setJointMotorControl2(
            self.model_id, 2,
            controlMode=p.POSITION_CONTROL,
            targetPosition=(gripperPosZ + dz),
            force=100.)
        self._physicsClient.setJointMotorControl2( 
                self.model_id, 3,
                controlMode=p.POSITION_CONTROL,
                targetPosition=(gripperYaw + dA),
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

    def getTerminated(self):
        #pulls location of end-effector
        #kills sim if terminated is True or counter has exceeded limit 
        #executes grasp if end-effector Z height meets threshold
        
        if (self.terminated or self._envStepCounter > self._maxSteps):
            self.terminated = True
            return True
        
        gripperPosZ = self._physicsClient.getJointState(self.model_id, 2)[0]

        if(gripperPosZ > 0.13): 
            self.terminated = True
            self.closeGripper()
            self.raiseGripper()
            return True
        
        return False
    
    def getReward(self): 
        #evaluates block position & assigns reward if block height is correct
        if(self._physicsClient.getBasePositionAndOrientation(self.blockUid)[0][2] > 0.05):
            return 1
        else:
            return 0

env = SimplifiedWorld(renders = True)
env = make_vec_env(lambda: env, n_envs=1)
model = SAC("CnnPolicy", env, verbose=1, device = 'cuda')
env = VecNormalize(env)
model.learn(total_timesteps=1000000, log_interval=4)

#UNIT TEST - ensure multiple random initialization result in block position/orientation congruent with Baris thesis
#UNIT TEST - ability for gripper to translate rotate around the environment without camera becoming imporperly detached 
#UNIT TEST - ability for rotational values to make sense when beyond +pi or -pi
