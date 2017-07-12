import numpy as np
import random
from Simulator import Simulator
from Narx import Narx
import Narx as narx
import matplotlib.pyplot as plt
from decimal import Decimal
from GA import GA
from Vehicles import RandomMotorVehicle


def pre_process(raw_data):
    """ Creates two arrays of training inputs and targets to feed to the NARX network """
    new_inputs = []
    new_targets = []
    for t in range(0, len(raw_data[0])):
        for vehicle in range(0, len(raw_data)):
            new_inputs.append(np.transpose(np.array(raw_data[vehicle][t])))
            # extracting targets from data and adding to new list (transposed)
            new_targets.append(np.transpose(np.array(raw_data[vehicle][t][-2:])))
    return [np.transpose(np.array(new_inputs)), np.transpose(np.array(new_targets))]


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
    return pre_process(data)


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
        self.vehicle = None
        self.brain = [0, 0, 0, 0]  # Vehicle brain, 4 weights
        self.vehicle_first_move = None

        self.sim = None

        self.count_cycles = 0

    def train_network(self, vehicle_runs, vehicle_iter, test_iter, input_delay, output_delay, net_max_iter, test_seed,
                      show_graph, test_runs=1):
        # collect data for NARX and testing and pre-process data
        train_input, train_target = collect_random_data(runs=vehicle_runs, iterations=vehicle_iter)
        test_input, test_target = collect_random_data(runs=test_runs, seed=test_seed, vehicle_angle=100,
                                                      iterations=test_iter)

        # Training of network

        self.net = Narx(input_delay=input_delay, output_delay=output_delay)

        # train network
        self.net.train(train_input, train_target, verbose=True, max_iter=net_max_iter)

        # save network to file
        self.net.save_to_file(filename='testNARX')
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

    def wake_learning(self, random_movements, train_network, learning_runs=4, learning_time=400, testing_time=400,
                      input_delay=5, output_delay=5, max_epochs=50, test_seed=200, show_graph=False):
        """ Start with random commands to train the model then compares actual with predicted sensor readings"""

        if train_network:
            self.train_network(learning_runs, learning_time, testing_time, input_delay, output_delay, max_epochs,
                               test_seed, show_graph)

        self.vehicle_first_move, targets = collect_random_data(runs=1, iterations=random_movements)
        print self.vehicle_first_move.shape

        self.count_cycles += 1

    def sleep(self, net_filename=None, lookAhaid=100, generations=5):
        if net_filename is not None:
            saved_net = narx.load_net(net_filename)
            self.net = Narx()
            self.net.set_net(saved_net)

        # run GA and find best brain to give to testing
        ga = GA()
        ga.run_offline(self.net, self.vehicle_first_move, timesteps=lookAhaid, generations=generations)

    def wake_testing(self):
        """ This phase uses the control system to iterate through many motor commands by passing them to the controlled
        robot in the world and retrieving its sensory information """

        pass

    def get_controls(self):
        pass
