import numpy as np
import random, time
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


class Cycle:

    def __init__(self, net_filename=None):

        self.net = None  # NARX network
        if net_filename is not None:
            print 'Loading NARX from file "%s"' % net_filename
            saved_net = narx.load_net(net_filename)
            self.net = Narx()
            self.net.set_net(saved_net)

        self.vehicle = None
        self.brain = [0, 0, 0, 0, 0, 0]  # Vehicle brain, 6 weights
        self.vehicle_first_move = None
        self.sensors = None

        self.sim = None

        self.count_cycles = 0

    def show_error_graph(self, testing_time=400, predict_after=100, brain=None):
        """ CHANGE THIS Presents a graph with both sensors' real and predicted values and their mean squared error """
        if self.net is None:
            print 'No network found'
            return
        # Lookahead is testing time - predict after
        look_ahead = testing_time - predict_after
        print 'Lookahead: ' + str(look_ahead)
        # Create random brain and give it to vehicle
        if brain is not None:
            random_brain = brain
        else:
            random_brain = [random.uniform(-8, 8) for x in range(0, 6)]
        print random_brain
        vehicle = BrainVehicle(start_pos=[300, 300], start_angle=200)
        vehicle.set_values(random_brain)
        # Create simulation, run vehicle in it, and collect its sensory and motor information
        sim = Simulator()
        vehicle = sim.init_simulation(testing_time + 1, True, veh_angle=200, brain=random_brain)
        sensor_motor = []
        for x in range(0, testing_time):
            sensor_motor.append(
                [vehicle.motor_left[x], vehicle.motor_right[x], vehicle.sensor_left[x], vehicle.sensor_right[x]])
        sensor_motor = np.transpose(np.array(sensor_motor))
        print sensor_motor.shape

        data = np.array(sensor_motor[:, :predict_after])  # data up until the initial run time
        sensor_log = np.array([[], []])
        wheel_log = []

        next_input = np.array(data[:, -1])
        next_input = np.array([[x] for x in next_input])
        for it in range(0, look_ahead):  # loop through the time steps
            # 1. predict next sensory output
            prediction = self.net.predict(next_input, pre_inputs=data, pre_outputs=data[2:])

            # concatenate to the full data
            data = np.concatenate((data, next_input), axis=1)

            # 2. log predicted sensory information to list (used for fitness)
            sensor_log = np.concatenate((sensor_log, prediction), axis=1)

            # 3. feed it to the brain to get motor information
            wheel_l, wheel_r = [
                (prediction[0] * random_brain[0]) + (prediction[1] * random_brain[3]) + random_brain[4] / 80,
                (prediction[1] * random_brain[2]) + (prediction[0] * random_brain[1]) + random_brain[5] / 80]
            wheel_log.append([wheel_l[0], wheel_r[0]])
            # 4. add this set of data to the input of the prediction
            next_input = np.array([wheel_l, wheel_r, prediction[0], prediction[1]])
            # loop back to 1 until reached timestep

        print wheel_log
        wheel_log = np.transpose(np.array(wheel_log))

        print sensor_motor[2][:predict_after].shape
        print sensor_log[0].shape

        sensor_left = np.concatenate((sensor_motor[2][:predict_after], sensor_log[0]))
        print sensor_left.shape

        sensor_right = np.concatenate((sensor_motor[3][:predict_after], sensor_log[1]))
        print sensor_right.shape

        motor_left = np.concatenate((sensor_motor[0][:predict_after], wheel_log[0]))
        print motor_left.shape

        motor_right = np.concatenate((sensor_motor[1][:predict_after], wheel_log[1]))
        print motor_right.shape

        # get sensory information of vehicle and compare with predicted
        plt.figure(1)
        i = np.array(range(0, len(vehicle.sensor_left)))
        plt.subplot(221)
        plt.title('Left sensor values b=real, r=pred')
        plt.plot(i, vehicle.sensor_left, 'b', i, sensor_left, 'r')

        plt.subplot(222)
        plt.title('Right sensor values b=real, r=pred')
        plt.plot(i, vehicle.sensor_right, 'b', i, sensor_right, 'r')

        plt.subplot(223)
        plt.title('Left motor values b=real, r=pred')
        plt.plot(i, vehicle.motor_left, 'b', i, motor_left, 'r')

        plt.subplot(224)
        plt.title('Right motor values b=real, r=pred')
        plt.plot(i, vehicle.motor_right, 'b', i, motor_right, 'r')
        plt.show()

    def train_network(self, learning_runs, learning_time, input_delay, output_delay, max_epochs, filename):
        # collect data for NARX and testing and pre-process data
        train_input, train_target = collect_random_data(runs=learning_runs, iterations=learning_time, graphics=False)

        # creation of network
        start_time = time.time()
        self.net = Narx(input_delay=input_delay, output_delay=output_delay)

        # train network
        self.net.train(train_input, train_target, verbose=True, max_iter=max_epochs)

        # save network to file
        self.net.save_to_file(filename=filename)
        print 'Finished training network "%s" in %ds' % (filename, time.time()-start_time)

    def wake_learning(self, random_movements, train_network=None, learning_runs=4, learning_time=400,
                      input_delay=5, output_delay=5, max_epochs=50):
        """ Start with random commands to train the model then compares actual with predicted sensor readings"""
        # Train network or use network alreay saved
        if train_network is not None:
            self.train_network(learning_runs, learning_time, input_delay, output_delay, max_epochs, train_network)

        # Create vehicle in simulation
        self.sim = Simulator()
        self.vehicle = self.sim.init_simulation(random_movements + 1, graphics=False, veh_pos=[300, 300])
        vehicle_move = []
        for t in range(0, random_movements):
            vehicle_move.append([self.vehicle.motor_left[t], self.vehicle.motor_right[t], self.vehicle.sensor_left[t],
                                 self.vehicle.sensor_right[t]])
        vehicle_first_move = []
        for t in range(0, len(vehicle_move)):
            vehicle_first_move.append(np.transpose(np.array(vehicle_move[t])))
        self.vehicle_first_move = np.transpose(np.array(vehicle_first_move))

    def sleep(self,look_ahead=100, individuals=25, generations=10):
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
        actual_vehicle = self.sim.run_simulation(iteration=iterations, graphics=False, vehicle=new_vehicle)

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
