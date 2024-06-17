# reinforcement_learning_robotic_pouring

This projects takes in concepts such as machine learning and specifically reinforcement learning with the end goal of teaching a robot to pour a cup into another cup. The success of each pour is determined by if the ice cubes in the pouring cup reach the destination cup. As robotics become more integrated into everyday life, this project serves as one potential viewpoint into how robotics may be able to aid humans in the future. From pouring cafe drinks in vending machines, this technology is already being utilized but further applications may include medical caretaking and in general, more human like arm motion.

One technique utilized is Q tables with Q-learning. The best potential move is determined by the Q table and the robot takes actions to move its hand left or right in order to get closer to the destination cup as determined by a heurisitc function. As the table gets populated, the robot learns the correct pattern to acheive higher and higher rewards (with the ultimate goal of bringing the ice cubes into the desination cup).

The results of Q-table learning spurred curiosity into other techniques to accelerate robotic learning for pouring tasks including techniques of Deep Learning, specifically leveraging Deep Q Learning. In Deep Q Learning, it is possible to utilize a neural network of sorts in order to determine the optimal Q-value to populate the table.

This projects mainly makes use of the Python programming language and involves CoppeliaSim for robotics simulation software.
