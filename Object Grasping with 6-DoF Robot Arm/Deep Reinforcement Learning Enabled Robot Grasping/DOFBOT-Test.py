import pybullet as p 
import time

import pybullet_data
robot_file = "/home/vrushank/Downloads/dofbot_moveit/urdf/dofbot.urdf"
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
target = p.getDebugVisualizerCamera()[11]
p.resetDebugVisualizerCamera(
    cameraDistance=1.1,
    cameraYaw=90,
    cameraPitch=-25,
    cameraTargetPosition=[target[0], target[1], 0.7])

robot = p.loadURDF(robot_file, [0, -0.3, 0], [0, 0, 0, 1], useFixedBase=True)

p.loadURDF("plane.urdf", [0, 0, 0], useFixedBase=True)
p.setRealTimeSimulation(1)

time.sleep(20000)