import numpy as np
import random
from Simulator import Simulator
from Narx import Narx
import Narx as narx
import matplotlib.pyplot as plt
from decimal import Decimal
from GA import GA
from Sprites import BrainVehicle, Light


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
        v = sim.quick_simulation(iterations + 1, graphics, vehicle_pos, vehicle_angle, light_pos, gamma, seed)
        vehicle_data_in_t = []
        for t in range(0, iterations):
            vehicle_data_in_t.append([v.motor_left[t], v.motor_right[t], v.sensor_left[t], v.sensor_right[t]])
        data.append(vehicle_data_in_t)
    print 'Collected data from ' + str(runs) + ' vehicles over ' + str(iterations) + ' iterations'
    return pre_process(data)


def error_graph(net, testing_time, input_delay=None, output_delay=None, max_epochs=None, vehicle_runs=None,
                vehicle_iter=None, test_runs=1, network_name=None):
    """ Presents a graph with both sensors' real and predicted values and their mean squared error """

    test_input, test_target = collect_random_data(runs=test_runs, vehicle_angle=random.randint(0, 360),
                                                  iterations=testing_time)
    # extract predictions and compare with test
    predictions = net.predict(test_input)
    predictions_left = predictions[0]
    predictions_right = predictions[1]
    real_left = np.array(test_target)[0]
    real_right = np.array(test_target)[1]

    # calculate mean squared error
    mse1 = [(predictions_left[it] - real_left[it]) ** 2 / len(predictions_left) for it in
            range(0, len(real_left))]
    mse2 = [(predictions_right[it] - real_right[it]) ** 2 / len(predictions_right) for it in
            range(0, len(real_right))]

    i = np.array(range(0, len(real_left)))
    plt.figure(1)
    if network_name is None:
        plt.suptitle('Results for: veh_runs=%s veh_iter=%s delays=%s:%s epoch:%s') % (vehicle_runs, vehicle_iter, input_delay, output_delay, max_epochs)
    else:
        plt.suptitle('Prediction results for network: ' + str(network_name))
    plt.subplot(221)
    plt.title('Left sensor MSE. Mean:' + '%.4E' % Decimal(str(np.mean(mse1))))
    plt.plot(range(0, len(mse1)), mse1)

    plt.subplot(222)
    plt.title('Right sensor MSE. Mean:' + '%.4E' % Decimal(str(np.mean(mse2))))
    plt.plot(range(0, len(mse2)), mse2)

    plt.subplot(223)
    plt.title('Left sensor values b=real, r=pred')
    plt.plot(i, real_left, 'b', i, predictions_left[:len(real_left)], 'r')

    plt.subplot(224)
    plt.title('Right sensor values b=real, r=pred')
    plt.plot(i, real_right, 'b', i, predictions_right[:len(real_right)], 'r')

    plt.show()


class Cycle:

    def __init__(self):

        self.net = None  # NARX network
        self.vehicle = None
        self.brain = [0, 0, 0, 0, 0, 0]  # Vehicle brain, 6 weights
        self.vehicle_first_move = None
        self.sensors = None

        self.sim = None

        self.count_cycles = 0

    def train_network(self, learning_runs, learning_time, testing_time, input_delay, output_delay, max_epochs,
                      show_error_graph):
        # collect data for NARX and testing and pre-process data
        train_input, train_target = collect_random_data(runs=learning_runs, iterations=learning_time)

        # creation of network
        self.net = Narx(input_delay=input_delay, output_delay=output_delay)

        # train network
        self.net.train(train_input, train_target, verbose=True, max_iter=max_epochs)

        # save network to file
        self.net.save_to_file(filename='narx/testNARX2')

        # Show error graph for trained network
        if show_error_graph:
            show_error_graph(self.net, learning_runs, learning_time, testing_time, input_delay, output_delay,
                             max_epochs)

    def wake_learning(self, random_movements, train_network, learning_runs=4, learning_time=400, testing_time=500,
                      input_delay=5, output_delay=5, max_epochs=50, show_error_graph=False):
        """ Start with random commands to train the model then compares actual with predicted sensor readings"""
        # Train network or use network alreay saved
        if train_network:
            self.train_network(learning_runs, learning_time, testing_time, input_delay, output_delay, max_epochs,
                               show_error_graph)

        # Create vehicle in simulation
        self.sim = Simulator()
        self.vehicle = self.sim.init_simulation(random_movements + 1, graphics=True, veh_pos=[300, 300])
        vehicle_move = []
        for t in range(0, random_movements):
            vehicle_move.append([self.vehicle.motor_left[t], self.vehicle.motor_right[t], self.vehicle.sensor_left[t],
                                 self.vehicle.sensor_right[t]])
        vehicle_first_move = []
        for t in range(0, len(vehicle_move)):
            vehicle_first_move.append(np.transpose(np.array(vehicle_move[t])))
        self.vehicle_first_move = np.transpose(np.array(vehicle_first_move))

    def sleep(self, net_filename=None, look_ahead=100, individuals=25, generations=10, show_error_graph=False):
        if net_filename is not None:
            print 'Loading NARX from file "%s"' % net_filename
            saved_net = narx.load_net(net_filename)
            self.net = Narx()
            self.net.set_net(saved_net)

        if show_error_graph:
            error_graph(self.net, 1000, network_name=net_filename)

        # run GA and find best brain to give to testing
        ga = GA()
        self.brain, self.sensors = ga.run_offline(self.net, self.vehicle_first_move, veh_pos=self.vehicle.pos[-1],
                                                  veh_angle=self.vehicle.angle, look_ahead=look_ahead,
                                                  individuals=individuals, generations=generations, crossover_rate=0.4,
                                                  mutation_rate=0.6)
        print 'Got best brain: ' + str(self.brain)

    def wake_testing(self, iterations):
        """ This phase uses the control system to iterate through many motor commands by passing them to the controlled
        robot in the world and retrieving its sensory information """
        new_vehicle = BrainVehicle(self.vehicle.pos[-1], self.vehicle.angle)
        new_vehicle.set_values(self.brain)
        new_vehicle.previous_pos = self.vehicle.pos
        actual_vehicle = self.sim.run_simulation(iteration=iterations, graphics=True, vehicle=new_vehicle)

        '''
        # get sensory information of vehicle and compare with predicted
        plt.figure(1)
        i = np.array(range(0, len(actual_vehicle.sensor_left)))
        plt.subplot(221)
        plt.title('Left sensor values b=real, r=pred')
        plt.plot(i, actual_vehicle.sensor_left, 'b', i, self.sensors[0][:-1], 'r')

        plt.subplot(222)
        plt.title('Right sensor values b=real, r=pred')
        plt.plot(i, actual_vehicle.sensor_right, 'b', i, self.sensors[1][:-1], 'r')
        plt.show()
        '''
