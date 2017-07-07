from pyneurgen.recurrent import NARXRecurrent
from pyneurgen.neuralnet import NeuralNet
from Simulator import Simulator
from GA import GA
import random
import numpy


def collect_random_data(vehicle_pos=None, vehicle_angle=None, light_pos=None, vehicle_runs=10, iterations=1000, graphics=False, gamma=0.2):
    if vehicle_pos is None:
        vehicle_pos = [300, 300]
    if vehicle_angle is None:
        vehicle_angle = random.randint(0, 360)
    if light_pos is None:
        light_pos = [1100, 600]
    # add 1 because the simulation returns iterations-1 as the first time step is the starting position (not recorded)
    data = []
    targets = []
    sim = Simulator()
    for run in range(0, vehicle_runs):

        v = sim.init_simulation(iterations+1, graphics, vehicle_pos, vehicle_angle, light_pos, gamma)
        vehicle_data_in_t = []
        target_data_in_t = []
        for t in range(0, iterations):
            vehicle_data_in_t.append([v.motor_left[t], v.motor_right[t], v.sensor_left[t], v.sensor_right[t]])
            target_data_in_t.append([v.sensor_left[t], v.sensor_right[t]])
        data.append(vehicle_data_in_t)
        targets.append(target_data_in_t)
    print 'Collected data from ' + str(vehicle_runs) + ' vehicles over ' + str(iterations) + ' iterations'
    return [data, targets]


# GA(graphics=True).run([300, 300], random.randint(0, 360), [1100, 600])


# collect data for narx
inputs, targets = collect_random_data(vehicle_runs=3)

'''
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
'''