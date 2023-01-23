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

**Jan 22nd Update**

I got a successful policy working in simulation! I was actually able to make things work using Stable Baselines 3 with some modifications and a denser reward structure. 

<img src="https://github.com/vdesai2014/Deep-Learning-Projects/blob/main/Object%20Grasping%20with%206-DoF%20Robot%20Arm/DofBot/dofbotsuccessful.gif?raw=true" width="600" height="338" />

The episodes are fixed length (100 timesteps) and in each time step the policy can choose to move the end-effector in the XY plane by some amount. Z movement is forced to be a certain negative distance in each step in order to speed up convergence. The computed end-effector coordinates are then fed into an inverse kinematics solver which commands the robot servos to the proper positions. The learned policy takes in depth images from a camera fixed to the end-effector and has gotten very good at panning the end-effector until it sees the block, and then centering the end-effector on the block so that the grasp is successful when the 100th time step occurs and the grasp is executed. The RL algorithm I choose was Soft Actor Critic as I found it to converge fairly quickly. While PPO or other policy gradient methods could work, I believe the reward structure is not conducive for sample efficient learning. Policy gradient algorithms, while more stable than an offline actor-critic based algorithm like SAC, typically work better when there is a continuous reward signal in the RL environment. This is not really the case for intermittently successful grasps. Another useful trick to speed up training was using curriculum learning by slowly scaling up the area over which blocks could randomly spawn. The initial area was 10cm x 0m and scaled up to 10cm x 5cm over four discrete steps. This was important because blocks closer to the origin are initially out of visual range of the camera’s field of view. A randomly initialized policy exposed to the full area to start with would struggle to deal with episodes where blocks spawned out of view. 

<img src="https://github.com/vdesai2014/Deep-Learning-Projects/blob/main/Object%20Grasping%20with%206-DoF%20Robot%20Arm/DofBot/Dofbot.png" width="858" height="338" />

Based on the literature on Sim2Real, I feel pretty good about having the simulated policy work in the real world, barring implementation of some domain randomization (camera position, image noise, etc). At the moment however, I am more interested in going under the hood of the RL + Deep Learning aspect of the project as it is a bit of a black box due to my use of Stable Baselines 3. As I understand it, SAC’s main objective is to maximize expected rewards over the course of the episode, while simultaneously maximizing the entropy of the agent’s actions. Entropy in this context refers to how random the distribution of sampled actions from the agent’s policy is. The idea with entropy maximization is that the agent explores more of the state-space and is less likely to get stuck in a local minima, or end up in states where the critic has no training data on. 

<img src="https://github.com/vdesai2014/Deep-Learning-Projects/blob/main/Object%20Grasping%20with%206-DoF%20Robot%20Arm/DofBot/sac_mainobjective.png" width="626" height="64" />

The critic network’s parameters can be learned by minimizing the specific temporal difference error shown below. This temporal difference error in standard reinforcement learning algorithms is the square of the difference between the predicted value of the current state, and the immediate reward received for a given action and the predicted value of the next state. SAC modifies this general TD-error equation and introduces a term to account for entropy.

Something that is pretty fascinating is that the generic temperal difference error equation was shown by neuroscientists in the 90s to explain activity of certain dopamine neurons. These neurons in the brain spiked in activity when the primate was presented with an unexpected reward (high prediction error) but would taper activity towards baseline once the reward was provided with a predictable cue (low prediction error). This may also explain why intermittent rewards, such as those provided by social media or gambling, can be so addictive. The inability for our brain's critic model to predict future rewards in these activites keeps us hooked in an effort to drive prediction error to zero. 

<img src="https://github.com/vdesai2014/Deep-Learning-Projects/blob/main/Object%20Grasping%20with%206-DoF%20Robot%20Arm/DofBot/sac_criticobjective.png" width="1071" height="47" />

The policy network updates are performed to drive towards a set of parameters that maximize the soft Q-value (critic network’s prediction of the “goodness” of a state/action pair) of actions taken by the policy. This is done by updating the policy to minimize the KL-divergence between the current policy and one which outputs actions with large Q-values. This expression which is minimized is shown below. 

<img src="https://github.com/vdesai2014/Deep-Learning-Projects/blob/main/Object%20Grasping%20with%206-DoF%20Robot%20Arm/DofBot/sac_policyobjective2.png" width="463" height="62" />

In practice, gradients cannot be directly computed from this expression and so the expression below is what is actually implemented, as it is a surrogate to the one above and yields the same gradients. The proof for going from the one above to the one below is pretty technical and at the moment, over my head, but I trust the authors given the results I have gotten using SAC! 

<img src="https://github.com/vdesai2014/Deep-Learning-Projects/blob/main/Object%20Grasping%20with%206-DoF%20Robot%20Arm/DofBot/sac_policyobjectiveimplement2.png" width="334" height="29" />

In the original SAC paper the alpha variable, which shows up in objectives for both the policy and the critic, was a fixed variable but a more recent update proposed the equation below to allow for the entropy objective to be automatically tuned as training progresses. The H_bar variable in the equation below is the lower bound for policy entropy and defined at the start of training. 

<img src="https://github.com/vdesai2014/Deep-Learning-Projects/blob/main/Object%20Grasping%20with%206-DoF%20Robot%20Arm/DofBot/sac_alpha.png" width="451" height="59" />

For my next update I hope to have gotten to the same success rate & policy effectiveness as I showed above with Stable Baselines 3, but with my own implementation of the CNN/Policy Network/Critic Network and training algorithms. 


*References*

[1] - https://github.com/chao0716/pybullet_ur5_robotiq/tree/robotflow

[2] - https://github.com/BarisYazici/deep-rl-grasping

[3] - https://stable-baselines3.readthedocs.io/en/master/modules/sac.html

[4] - https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html

[5] - https://spinningup.openai.com/en/latest/

[6] - http://www.yahboom.net/study/Dofbot-Pi

[7] - https://arxiv.org/abs/1812.05905

[8] - https://github.com/thomashirtz/soft-actor-critic

[9] - https://www.youtube.com/watch?v=_nFXOZpo50U
