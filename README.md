# Deep-Learning-Projects
A compilation of self-learned projects to showcase my ability to independently develop complex AI/Robotics systems. 

**Neural Network Visualizer**

<img src="https://github.com/vdesai2014/Deep-Learning-Projects/blob/main/Neural%20Network%20Visualizer/Network%20Training.gif" width="800" height="800" />



In order to better understand artificial neural networks, I implemented a feed-forward deep neural network in Python and built a visualizer to understand how layer activations changed as training progressed. The network shown here is learning a very simple "exclusive-or" function, and is expected to output a "1" if and only if the input arguments differ. The visualization is a bit inaccurate in that it is showing the result of the summation function as being divided equally over each of the inputs. A point for future improvement would be to adjust the width of the line & activation (or lack thereof) by the weight of that link. 

Regardless, it is interesting to visualize how layer activations change as training progresses over 20,000 epochs, and also how relatively few neurons make a meaningful impact on the networks prediction. This effect is well-known and is the reason why network weights are typically "pruned" after being trained, as keeping only pieces of the network with larger weights allows for much faster performance at inference time. Interestingly, this [effect can also be used in reverse] (https://ai.facebook.com/blog/understanding-the-generalization-of-lottery-tickets-in-neural-networks/), by starting with a very specific initialization & training only specific parts of the network. 


