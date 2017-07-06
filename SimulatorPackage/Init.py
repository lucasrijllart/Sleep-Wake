from pyneurgen.recurrent import NARXRecurrent
from pyneurgen.neuralnet import NeuralNet
from Simulator import Simulator
from GA import GA
import random


GA(graphics=True).run([300, 300], random.randint(0, 360), [1100, 600])

iterations = 300  # number of iterations to run simulation for
show_graphics = True  # True shows graphical window, False doesn't

random_vehicle_pos = False
random_vehicle_angle = False
random_light_pos = False

# for x in range(0, 1):
sim = Simulator()
v = sim.init_simulation_random(iterations, show_graphics, random_vehicle_pos, random_vehicle_angle, random_light_pos)

# collect data for narx
inputs = []
targets = []
for i in range(0, len(v.motor_left)):
    inputs.append([v.sensor_left[i], v.sensor_right[i], v.motor_left[i], v.motor_right[i]])
    targets.append([v.sensor_left[i], v.sensor_right[i]])
print inputs
print targets

# Set number of nodes
input_nodes = 4
hidden_nodes = [5, 5]
output_nodes = 2

# Set params
output_order = 5 # The delay of the network
incoming_weight_from_output = .6
input_order = 2  # The number of time steps in the past
incoming_weight_from_input = .4


# Intialize the NARX network
net = NeuralNet()
net.init_layers(input_nodes, hidden_nodes, output_nodes,
    NARXRecurrent(
        output_order,
        incoming_weight_from_output,
        input_order,
        incoming_weight_from_input))

# random initialize the network weights
net.randomize_network()

# set inputs and targets
net.set_all_inputs(inputs)
net.set_all_targets(targets)

# set the percentage of data to learn on. 80% in this case
length = len(inputs)
learn_end_point = int(length * .8)

# set the test data
net.set_learn_range(0, learn_end_point)
net.set_test_range(learn_end_point + 1, length - 1)

# set activation function
net.layers[1].set_activation_type('tanh')
net.layers[2].set_activation_type('tanh')

# train network
net.learn(epochs=100, show_epoch_results=True, random_testing=False)

print 'Testing the MSE'
print net.test()

print 'Size of all the data= ', len(inputs)
print net.get_test_range()

net.set_all_inputs(inputs)
print net._feed_forward()