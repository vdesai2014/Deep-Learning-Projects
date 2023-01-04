# Deep-Learning-Projects
A compilation of self-learned projects to showcase my ability to independently work on AI/Robotics problems.  

**Neural Network Visualizer**

<img src="https://github.com/vdesai2014/Deep-Learning-Projects/blob/main/Neural%20Network%20Visualizer/Network%20Training.gif" width="800" height="800" />



In order to better understand artificial neural networks, I implemented a feed-forward deep neural network in Python and built a visualizer to understand how layer activations changed as training progressed. The network shown here is learning a very simple "exclusive-or" function, and is expected to output a "1" if and only if the input arguments differ. The visualization is a bit inaccurate in that it is showing the result of the summation function as being divided equally over each of the inputs. A point for future improvement would be to adjust the width of the line & activation (or lack thereof) by the weight of that link. 

Regardless, it is interesting to visualize how layer activations change as training progresses over 20,000 epochs, and also how relatively few neurons make a meaningful impact on the networks prediction. This effect is well-known and prominent for small networks such as the one being visualized. This is the reason why network weights are typically "pruned" after being trained, as keeping only pieces of the network with larger weights allows for much faster performance at inference time. Interestingly, this [effect can also be used in reverse](https://ai.facebook.com/blog/understanding-the-generalization-of-lottery-tickets-in-neural-networks/), by starting with a very specific initialization & training only specific parts of the network. The difficulty with this approach when training large networks lies in identifying exactly which intersection of network structure and specific initiailization weights is the "lottery ticket".

*References*

[1] - https://github.com/fanghao6666/neural-networks-and-deep-learning/blob/master/py/Building%20your%20Deep%20Neural%20Network%20Step%20by%20Step%20v3.py

[2] - https://matplotlib.org/stable/index.html


**Real-Time Robot Arm Control using Pose Estimation**

<img src="https://github.com/vdesai2014/Deep-Learning-Projects/blob/main/Real-time%206DoF%20Robot%20Arm%20Control%20with%20Pose%20Estimation/Pose%20Estimate%20to%20Robot%20Control.gif" width="600" height="338" />

This project involved using a pre-trained pose estimation model (implemented via MediaPipe) to control a Kuka robot arm in a virtual PyBullet environment. While not particularly technically challenging beyond utilizng some simple 2D kinematics, this project was a fun way to implement a useful AI model and learn the PyBullet API in the process. Future improvements could involve using a 3D pose estimation model to allow for more complex manuvers with the robot arm. Perhaps these complex 3D manuvers could be turned into expert-trajectories that can help speed up training of an agent learning via deep reinforcement. 

*References*

[1] - https://gerardmaggiolino.medium.com/creating-openai-gym-environments-with-pybullet-part-1-13895a622b24

[2] - https://www.youtube.com/watch?v=06TE_U21FK4

**Object Grasping with 6-DoF Robot Arm**

**Dec 11th Update**

My goal for this project is to get a robot arm (https://www.amazon.com/dp/B08T8XG2J6) in the real world to pick up blocks. I started out with a deep reinforcement learning based approach and intended on using modern sample efficient DRL algorithms (SAC/PPO) to train a robot arm to do this in simulation. Then I planned on transferring this learned policy to the real arm and iteratively utilizing domain randomization to improve the learned policy in simulation based on real world results (or lack thereof). Unfortunately, even trying to get the learned policy to work in simulation has proven to be very challenging (see below, .gif of my simulated robot failing). The robot in the .gif is not the one I plan to use in real life. I don't have a .URDF file for it yet, but the UR5 model is similar and I can use that for now. I'll switch to simulating the Amazon Dof-Bot once I get the UR5 working.

<img src="https://github.com/vdesai2014/Deep-Learning-Projects/blob/main/Object%20Grasping%20with%206-DoF%20Robot%20Arm/FullArmRL/graspingFail.gif" width="600" height="338" />

My game-plan going forward is shown below. I will continue to update this section as I make progress. 

1) Simplify getting learned policy to converge by directly providing X,Y,Z coordinates of block as an observation for the multi-layer perceptron network. I have already had success with this with a previous simpler Pybullet environment and am confident I can make this work.  
2) Implement learned policy in real life by using a camera, OpenCV, and ArUco boards to extract ground truth X/Y/Z coordinates of object to grasp. Feed this observation into the learned policy to control Dof-Bot and grasp object. 
3) Once the sim2real gap has been crossed with a more simple observation space, go back and make the original end-to-end vision based policy work without fiducials or OpenCV. This will involve a fair bit of debugging to understand why using a CNN & normalized depth image observations makes learning so much more unstable. 

**Jan 4th Update** 

Over the holidays I was able to set up the Dofbot in real life and build a .URDF model for it in Pybullet with a functional gripper! I am able to send commands to a rasp-pi over TCP in real-time while also sending the same commands to the simulated robot in order to ensure that the simulation has identical kinematics to the real world. I included a laser-etched grid under the robot to extract ground-truth XY coordinates of blocks and use these to verify the congruency of grasps executed in simulation and reality. The next challenge to tackle is to get a working grasping policy that uses visual input from a depth camera and maneuvers the end-effector to execute grasps. I am having lots of trouble getting a policy to converge using Stable Baselines 3 (RL algorithm library) and plan to use my own implementation of PPO or SAC in Python to better understand what is happening with the training under the hood. Hope to report back with good news in a few weeks! 

<img src="https://github.com/vdesai2014/Deep-Learning-Projects/blob/main/Object%20Grasping%20with%206-DoF%20Robot%20Arm/DofBot/real_dofbot.jpg" width="600" height="338" />

<img src="https://github.com/vdesai2014/Deep-Learning-Projects/blob/main/Object%20Grasping%20with%206-DoF%20Robot%20Arm/DofBot/simulated_dofbot.png" width="600" height="338" />




*References*

[1] - https://github.com/chao0716/pybullet_ur5_robotiq/tree/robotflow

[2] - https://github.com/BarisYazici/deep-rl-grasping

[3] - https://stable-baselines3.readthedocs.io/en/master/modules/sac.html

[4] - https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html

[5] - https://spinningup.openai.com/en/latest/
