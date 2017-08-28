# Sleep-Wake Cycles with environment-modelling robotics
This project consists of applying machine learning to a Braitenberg vehicle to simulate the modelling of its environment through motor commands and sensory information. The robot will go through "sleep-wake" cycles to accelerate the learning of the environment by doing "offline" processing with data gathered in real-time.

The simulated vehicle uses a NARX network (Non-linear AutoRegressive network with eXogenous inputs), which uses past sensory information with present motor information to predict the next sensory outcome.

This is the thesis for my Master in Intelligent Systems at the University of Sussex. Supervised by Chris Buckley and helped by Stathis Kagioulis.

## How to use this project
Please clone the repository and make sure you have the module PyRenn installed.
https://github.com/yabata/pyrenn

From the Main.py file, you can run tests or simply cycles.

## Abstract
Robotics has always faced the issue of adaptability to new environments and new situations. To create an adaptive behaviour, we try to make the agent create its own model of the environment. We design Sleep-Wake cycles to structure the learning and problem solving behaviour of the agent. By using a model of the environment to predict future movement, we can simulate all robotic trial-and-error, decreasing the amount of real time needed to get a solution. Results show one cycle performs nearly as well as the optimal answer, with a generalised behaviour generated from a short prediction, and solving the problem in less time than trial-and-error.

## Introduction
In the event of limb damage, humans and animals can correct their movements by compensating for the damaged mechanic.  Unlike animals, machines lack the ability to adapt their behaviour in the event of an unexpected change, or predict new outcomes in an unseen territory. Robots are given precise, unbreakable rules that do not change over time, and they do not try to predict the outcome of an action, as it should never change. To build a more intelligent machine, we try to simulate an understanding of the environment so it can generate new behaviours. 

Many modern machines are given increasingly complex models of their environment and the complicated equations of behaviours. This research will focus on the opposite approach: giving the agents a simpler model of their environment in the hope they can create their own simple behaviours. The model will not be humanly understandable, nor will it visually depict the environment. This model will be created through a NARX neural network. This recurrent dynamical neural network will serve as a model for the robot to predict future behaviour. 

To test a new behaviour, a robot would have to execute the new actions in the real world, and observe the result. This method would take an unreasonable amount of time to discover a new behaviour. We are therefore looking to shift all real-time actions to offline processing, where the robot can use its understanding of the world to predict its next behaviour.

We decide to name the active movement of the robot a “wake” stage. To create an opportunity to do processing, we incorporate a “sleep” stage that can be considered like “thinking”. The robot can then alternate between wake stage and sleep stage to minimise real movement.
