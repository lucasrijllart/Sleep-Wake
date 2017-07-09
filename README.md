# BTBSimulator

This project consists of applying machine learning to a Braitenberg vehicle to simulate the modelling of its environment through motor commands and sensory information.
The robot will go through "sleep-wake" cycles to accelerate the learning of the environment by doing "offline" processing with data gathered in real-time.

The simulated vehicle uses a NARX network (Non-linear AutoRegressive network with eXogenous inputs), which uses past sensory information with present motor information to predict the next sensory outcome.

This is the thesis for my Master in Intelligent Systems at the University of Sussex. Helped by Stathis Kagioulis and supervised by Chris Buckley.
