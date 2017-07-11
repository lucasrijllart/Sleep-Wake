import numpy as np
import random
from Simulator import Simulator
from Narx import Narx
import matplotlib.pyplot as plt
from decimal import Decimal
from GA import GA


def pre_process(raw_data):
    """ Creates two arrays of training inputs and targets to feed to the NARX network """
    new_inputs = []
    new_targets = []
    for t in range(0, len(raw_data[0])):
        for vehicle in range(0, len(raw_data)):
            new_inputs.append(np.transpose(np.array(raw_data[vehicle][t])))
            # extracting targets from data and adding to new list (transposed)
            new_targets.append(np.transpose(np.array(raw_data[vehicle][t][-2:])))
    return [new_inputs, new_targets]


def collect_random_data(vehicle_pos=None, vehicle_angle=None, light_pos=None, runs=10, iterations=1000, graphics=False,
                        gamma=0.2, seed=None):
    """ Runs many vehicles in simulations and collects their sensory and motor information """
    if vehicle_pos is None:
        vehicle_pos = [300, 300]
    if vehicle_angle is None:
        vehicle_angle = random.randint(0, 360)
    if light_pos is None:
        light_pos = [1100, 600]
    # add 1 because the simulation returns iterations-1 as the first time step is the starting position (not recorded)
    data = []
    sim = Simulator()
    for run in range(0, runs):
        v = sim.init_simulation(iterations + 1, graphics, vehicle_pos, vehicle_angle, light_pos, gamma, seed)
        vehicle_data_in_t = []
        for t in range(0, iterations):
            vehicle_data_in_t.append([v.motor_left[t], v.motor_right[t], v.sensor_left[t], v.sensor_right[t]])
        data.append(vehicle_data_in_t)
    print 'Collected data from ' + str(runs) + ' vehicles over ' + str(iterations) + ' iterations'
    return data


def show_error_graph(vehicle_runs, vehicle_iter, input_delay, output_delay, net_max_iter, mse1, mse2, real_left,
                     real_right, predictions_left, predictions_right):
    """ Presents a graph with both sensors' real and predicted values and their mean squared error """
    i = np.array(range(0, len(real_left)))
    plt.figure(1)
    plt.suptitle('Results for: veh_runs=' + str(vehicle_runs) + ' veh_iter=' + str(vehicle_iter) + ' delays=' +
                 str(input_delay) + ':' + str(output_delay) + ' net_iter:' + str(net_max_iter))
    plt.subplot(221)
    plt.title('Left sensor MSE. Mean:' + '%.4E' % Decimal(str(np.mean(mse1))))
    plt.plot(range(0, len(mse1)), mse1)

    plt.subplot(222)
    plt.title('Right sensor MSE. Mean:' + '%.4E' % Decimal(str(np.mean(mse2))))
    plt.plot(range(0, len(mse2)), mse2)

    plt.subplot(223)
    plt.title('Left sensor values b=real, r=pred')
    plt.plot(i, real_left, 'b', i, predictions_left, 'r')

    plt.subplot(224)
    plt.title('Right sensor values b=real, r=pred')
    plt.plot(i, real_right, 'b', i, predictions_right, 'r')

    plt.show()


class Cycle:

    def __init__(self):

        self.net = None  # NARX network
        self.brain = [0, 0, 0, 0]  # Vehicle brain, 4 weights
        self.train_input = None

        self.count_cycles = 0

    def wake_learning(self, vehicle_runs=4, vehicle_iter=400, test_iter=400, input_delay=5, output_delay=5,
                      net_max_iter=50, test_seed=200, show_graph=False):
        """ Start with random commands to train the model then compares actual with predicted sensor readings"""

        # Random training commands
        test_runs = 1

        # collect data for NARX and testing and pre-process data
        data = collect_random_data(runs=vehicle_runs, iterations=vehicle_iter)
        inputs_list, targets_list = pre_process(data)
        data = collect_random_data(runs=test_runs, seed=test_seed, vehicle_angle=100, iterations=test_iter,
                                   graphics=False)
        test_input, test_target = pre_process(data)

        # separation into training and testing
        self.train_input = np.transpose(np.array(inputs_list))
        train_target = np.transpose(np.array(targets_list))
        test_input = np.transpose(np.array(test_input))
        test_target = np.transpose(np.array(test_target))

        # Training of network

        self.net = Narx(input_delay=input_delay, output_delay=output_delay)

        # train network
        self.net.train(self.train_input, train_target, verbose=True, max_iter=net_max_iter)

        # Predicting of sensory outputs

        # extract predictions and compare with test
        predictions = self.net.predict(test_input)
        predictions_left = predictions[0]
        predictions_right = predictions[1]
        real_left = np.array(test_target)[0]
        real_right = np.array(test_target)[1]

        # calculate mean squared error
        mse1 = [(predictions_left[it] - real_left[it]) ** 2 / len(predictions_left) for it in
                range(0, len(predictions_left))]
        mse2 = [(predictions_right[it] - real_right[it]) ** 2 / len(predictions_right) for it in
                range(0, len(predictions_right))]

        # Show results
        if show_graph:
            show_error_graph(vehicle_runs, vehicle_iter, input_delay, output_delay, net_max_iter, mse1, mse2, real_left,
                             real_right, predictions_left, predictions_right)
        self.count_cycles += 1

    def sleep(self):
        # run GA and find best brain to give to testing
        ga = GA()
        ga.run_offline(self.net, self.train_input)

    def wake_testing(self):
        """ This phase uses the control system to iterate through many motor commands by passing them to the controlled
        robot in the world and retrieving its sensory information """

        pass

    def get_controls(self):
        pass
