import numpy as np
import random, time, datetime, re
from Simulator import Simulator
from Narx import PyrennNarx, NarxMLP
import Narx as narx
import matplotlib.pyplot as plt
import Genetic
from Genetic import GA as GA
from Sprites import *
import Sprites
import os.path


def pre_process_by_time(raw_data):
    """ Creates two arrays of training inputs and targets to feed to the NARX network """
    new_inputs = []
    new_targets = []
    for t in range(0, len(raw_data[0])):
        for vehicle in range(0, len(raw_data)):
            new_inputs.append(np.transpose(np.array(raw_data[vehicle][t])))
            # extracting targets from data and adding to new list (transposed)
            new_targets.append(np.transpose(np.array(raw_data[vehicle][t][-2:])))
    return [np.transpose(np.array(new_inputs)), np.transpose(np.array(new_targets))]


def pre_process_by_vehicle(raw_data):
    """ Returns the data in a list of vehicles. Every vehicle has a list of timesteps with each motors and sensors """
    new_inputs = []
    for vehicle in raw_data:
        vehicle_timesteps = []
        for timestep in vehicle:
            vehicle_timesteps.append(np.transpose(np.array(timestep)))
        new_inputs.append(np.transpose(np.array(vehicle_timesteps)))
    new_inputs = np.array(new_inputs)
    return new_inputs


class Cycles:

    def __init__(self, light_pos, type=None, net_filename=None):

        self.net = None  # NARX network
        self.net_filename = net_filename

        # network training
        self.training_runs, self.training_time = None, None
        self.random_movements, self.after_ga_movements = None, None

        if type == 'skmlp':
            self.net = narx.load_narx_mlp(net_filename)
        if net_filename is not None:
            start_time = time.time()
            saved_net = narx.load_pyrenn(net_filename)
            self.net = PyrennNarx()
            self.net.set_net(saved_net)
            self.training_runs = int(re.search('\d+', re.search('r\d+', net_filename).group(0)).group(0))
            self.training_time = int(re.search('\d+', re.search('t\d+', net_filename).group(0)).group(0))
            print 'Loaded NARX from file "%s" in %ds' % (net_filename, time.time() - start_time)

        self.random_vehicle = None
        self.brain = [0, 0, 0, 0, 0, 0]  # Vehicle brain assigned after GA, 6 weights
        self.vehicle_first_move = None
        self.predicted_pos = None

        self.ga_individuals = None
        self.ga_generations = None

        self.required_distance_from_light = 400

        self.sim = None
        if light_pos is None:
            light_pos = [1100, 600]
        self.light = Light(light_pos)

        self.actual_vehicle = None

        self.count_cycles = 0

    def find_random_pos(self):
        """ Returns a random position and angle where the position is far enough from the light

        :return: [[pos_x, pos_y], angle]
        """
        dist_to_light, rand_x, rand_y = 0, 0, 0
        while dist_to_light < self.required_distance_from_light:
            rand_x = random.randint(RandomMotorVehicle.radius, Simulator.window_width)
            rand_y = random.randint(RandomMotorVehicle.radius, Simulator.window_height)
            dist_to_light = np.sqrt((rand_x - self.light.pos[0]) ** 2 + (rand_y - self.light.pos[1]) ** 2)
        return [rand_x, rand_y], random.randint(0, 360)

    def collect_random_data(self, rand_vehicle_pos=True, runs=10, iterations=1000, data_collection_graphics=False,
                            seed=None, gamma=0.3):
        """ Runs many vehicles in simulations and collects their sensory and motor information

        :param rand_vehicle_pos: if vehicle should spawn randomly
        :param runs: number of vehicle runs to execute
        :param iterations: number of iterations of a run
        :param data_collection_graphics: show the vehicles while collecting data
        :param gamma:
        :param seed: seed to make the collection of data identically random
        :return: data collected and pre-processed
        """

        if seed is not None:
            random.seed(seed)

        data = []
        sim = Simulator(self.light)
        for run in range(0, runs):
            if rand_vehicle_pos:  # needs to be further than 500
                vehicle_pos, vehicle_angle = self.find_random_pos()
            else:
                vehicle_pos = [900, 600]
                vehicle_angle = 200
            v = sim.quick_simulation(iterations, data_collection_graphics, vehicle_pos, vehicle_angle, gamma, start_stop=True)
            vehicle_data_in_t = []
            for t in range(0, iterations):
                vehicle_data_in_t.append([v.motor_left[t], v.motor_right[t], v.sensor_left[t], v.sensor_right[t]])
            data.append(vehicle_data_in_t)
        print '\nCollected data from %d vehicles over %d iterations' % (runs, iterations)
        return pre_process_by_vehicle(data)

    def show_error_graph(self, testing_time=300, predict_after=50, brain=None, seed=None, graphics=True):
        """ Presents a graph with real and predicted sensor and motor values """
        if self.net is None:
            print 'show_error_graph() Exception: No network found'
            return
        if seed is not None:
            random.seed(seed)
        # Lookahead is testing time - predict after
        look_ahead = testing_time - predict_after

        # Create simulation, run vehicle in it, and collect its sensory and motor information
        sim = Simulator(self.light)
        # find random starting pos
        vehicle_pos, vehicle_angle = self.find_random_pos()
        # execute vehicle
        vehicle = sim.init_simulation(testing_time, graphics, veh_pos=vehicle_pos, veh_angle=vehicle_angle, brain=brain, start_stop=True)
        data = []
        for x in range(0, testing_time):
            data.append(
                [vehicle.motor_left[x], vehicle.motor_right[x], vehicle.sensor_left[x], vehicle.sensor_right[x]])
        data = np.transpose(np.array(data))

        # check if vehicle has a brain, if not just pass it the data
        if brain is not None:
            brain = [float(item) for item in brain]
            pass
        else:  # vehicle does not have a brain, just random movement
            # make predictions
            sensor_log, wheel_log = self.net.predict_error_graph(data, look_ahead, predict_after)

            v_iter = len(sensor_log[0])
            msel = [((sensor_log[0][i] - vehicle.sensor_left[i]) ** 2) / v_iter for i in range(0, v_iter)]
            mser = [((sensor_log[1][i] - vehicle.sensor_right[i]) ** 2) / v_iter for i in range(0, v_iter)]

            if graphics:
                # get sensory information of vehicle and compare with predicted
                plt.figure(1)
                brain = 'random'
                title = 'Graph showing predicted vs actual values for sensors and Mean Squared Error.\nNetwork:%s,' \
                        'with %d timesteps and predictions starting at t=%d, and with %s brain' %\
                        (self.net_filename, testing_time, predict_after, brain)
                plt.suptitle(title)
                i = np.array(range(0, len(vehicle.sensor_left)))
                i2 = np.array(range(predict_after, testing_time))
                plt.subplot(221)
                plt.title('Left sensor values')
                plt.plot(i, vehicle.sensor_left, 'b', label='real')
                plt.plot(i2, sensor_log[0], 'r', label='predicted')

                plt.subplot(222)
                plt.title('Right sensor values')
                plt.plot(i, vehicle.sensor_right, 'b', label='real')
                plt.plot(i2, sensor_log[1], 'r', label='predicted')
                plt.legend()

                plt.subplot(223)
                plt.title('MSE of left sensor')
                plt.plot(i2, msel)

                plt.subplot(224)
                plt.title('MSE of right sensor')
                plt.plot(i2, mser)
                print 'Error: ' + str(sum(msel) + sum(mser))
                plt.show()

            return sum(msel) + sum(mser)

    def benchmark_tests(self, random_brains=1000, ga_graphics=True):
        """
        Shows a graph with the fitness of the predicted vehicle, the evolved vehicle and a number of random brains
        :param random_brains: number of random brains to run to get a good benchmark curve
        :param ga_graphics: set to True to show online GA and
        :return:
        """
        start_pos, start_a = self.random_vehicle.pos[-1], self.random_vehicle.angle
        iterations = self.after_ga_movements
        fitnesses = []

        print '\nStarting benchmark test for %d random brains...' % random_brains
        start_time = time.time()
        for individual in range(0, random_brains):
            brain = Genetic.make_random_brain()
            fitnesses.append(Genetic.get_fitness(start_pos, start_a, brain, iterations, self.light))
        print 'Collected %d random brains in %ds' % (random_brains, time.time() - start_time)
        random_mean_fit = np.mean(fitnesses)

        ga = GA(self.light, ga_graphics)
        brain = ga.run(start_pos, start_a, self.ga_individuals, self.ga_generations, iterations)
        brain = brain[0]
        evolved_score = Genetic.get_fitness(start_pos, start_a, brain, iterations, self.light)
        fitnesses.append(evolved_score)

        brain = self.actual_vehicle.get_brain()
        predicted_score = Genetic.get_fitness(start_pos, start_a, brain, iterations, self.light)
        fitnesses.append(predicted_score)
        fitnesses.sort()

        plt.figure(1)
        plt.suptitle('Benchmark test for fitness of agents and time comparison between online and offline')
        plt.subplot(121)

        plt.title('Benchmark test for network predicted vehicle\n with evolved control and %d random brain vehicles' %
                  random_brains)
        evol_idx = np.where(fitnesses == evolved_score)
        pred_idx = np.where(fitnesses == predicted_score)
        plt.scatter(range(0, len(fitnesses)), fitnesses, s=1, c='grey', label='random')
        plt.scatter(evol_idx, evolved_score, s=15, c='green', label='evolved')
        plt.scatter(pred_idx, predicted_score, s=20, c='red', label='predicted')
        plt.plot([0, len(fitnesses)], [random_mean_fit, random_mean_fit], c='blue', label='random mean fitness')
        plt.xlabel('individuals')
        plt.ylabel('fitness')
        plt.legend()

        plt.subplot(122)
        ga_time = self.random_movements + self.ga_individuals * self.ga_generations * iterations
        pred_time = self.training_runs * self.training_time + self.random_movements + self.after_ga_movements
        plt.title('Time comparison between online and offline GA\n and their fitness')
        plt.scatter(ga_time, evolved_score, s=40, c='green', label='real-world: ' + str(ga_time))
        plt.scatter(pred_time, predicted_score, s=40, c='red', label='predicted: ' + str(pred_time))
        plt.xlabel('time steps')
        plt.ylabel('fitness')
        plt.legend()
        plt.show()

        # print 'Predicted time: %d\nReal-world time: %d' % (pred_time, ga_time)
        benchmark_test = predicted_score > random_mean_fit  # test if the predicted fitness is better than chance
        time_test = pred_time < ga_time  # test if the predicted time is smaller than real-world time

        print 'Predicted vehicle tests:\n\tRandom benchmark:\t%s\n\tOn/Offline time:\t%s' \
              % (benchmark_test, time_test)

        return benchmark_test, time_test

    def train_network(self, api, learning_runs, learning_time, layers, delay, max_epochs, use_mean=True,
                      seed=None, graphics=False):
        self.training_runs, self.training_time = learning_runs, learning_time
        self.net_filename = 'narx/r%dt%dd%de%d' % (learning_runs, learning_time, delay, max_epochs)

        # collect data for NARX and testing and pre-process data
        train_input = self.collect_random_data(True, learning_runs, learning_time, graphics, seed)

        # creation of network
        print '\nNetwork training started at ' + str(
            time.strftime('%H:%M:%S %d/%m', time.localtime()) + ' with params:')
        print '\t learning runs=%d, learning time=%d, delays=%d, epochs=%d' % (
        learning_runs, learning_time, delay, max_epochs)

        start_time = time.time()
        if api == 'pyrenn':
            self.net = PyrennNarx(layers=layers, delay=delay)
            # train network
            self.net.train(train_input, verbose=True, max_iter=max_epochs, use_mean=use_mean)

            # check if filename is already taken
            count = 1
            new_filename = self.net_filename
            while os.path.exists(new_filename):
                new_filename = self.net_filename + '_v' + str(count)
                count += 1
            filename = new_filename
            # save network to file
            self.net.save_to_file(filename=filename)
        elif api == 'skmlp':
            self.net = NarxMLP()

            self.net.fit(train_input, delay, use_mean)

            # check if filename is already taken
            count = 1
            new_filename = self.net_filename
            while os.path.exists(new_filename):
                new_filename = self.net_filename + '_v' + str(count)
                count += 1
            filename = new_filename

            self.net.to_file(filename)
        else:
            print 'Wrong network type given'

        print 'Finished training network "%s" in %s' % (self.net_filename, datetime.timedelta(seconds=time.time() - start_time))

    def test_network(self, tests=100, test_time=200, seed=1):
        """ Function that tests a network's error on many random tests
        :param tests: number of tests to run (100-200 takes the right amount of time)
        :param test_time: usually around 100-200 (should be the same as network has been trained on)
        :param seed: makes all test vehicles the same random set for every network tested
        :return: nothing
        """
        print '\nTesting network %s for %d times...' % (self.net_filename, tests)
        if seed is not None:
            random.seed(seed)
        combined_error = []
        for it in range(0, tests):
            if it % 10 == 0:
                print '\rTest %d/%d' % (it, tests),
            combined_error.append(self.show_error_graph(test_time, graphics=False))
        combined_error_mean = sum(combined_error) / tests
        print '\rFinished network test. Average error: %s' % combined_error_mean

        plt.title('Testing network %s for %d tests of %d timesteps.\nAverage error:%s' %
                  (self.net_filename, tests, test_time, combined_error_mean))
        plt.scatter(range(0, len(combined_error)), np.sort(combined_error), s=5, label='test error')
        plt.plot([0, len(combined_error)], [combined_error_mean, combined_error_mean], label='mean error')
        plt.xlabel('test number')
        plt.ylabel('Sum of MSE of both sensors')
        plt.legend()
        plt.show()

    def wake_learning(self, random_movements):
        """ Create a vehicle and run for some time-steps """
        self.random_movements = random_movements
        # Create vehicle in simulation
        self.sim = Simulator(self.light)
        vehicle_pos, vehicle_angle = self.find_random_pos()
        self.random_vehicle = self.sim.init_simulation(random_movements, graphics=True, cycle='wake (training)',
                                                       veh_pos=vehicle_pos, veh_angle=vehicle_angle)
        vehicle_move = [[self.random_vehicle.motor_left[0], self.random_vehicle.motor_right[0],
                         self.random_vehicle.sensor_left[0], self.random_vehicle.sensor_right[0]]]
        for t in range(1, random_movements):
            vehicle_move.append([self.random_vehicle.motor_left[t], self.random_vehicle.motor_right[t],
                                 self.random_vehicle.sensor_left[t], self.random_vehicle.sensor_right[t]])
        vehicle_first_move = []
        for t in range(0, len(vehicle_move)):
            vehicle_first_move.append(np.transpose(np.array(vehicle_move[t])))
        self.vehicle_first_move = np.transpose(np.array(vehicle_first_move))
        print self.vehicle_first_move

    def sleep(self, look_ahead=100, individuals=25, generations=10):
        self.ga_individuals = individuals
        self.ga_generations = generations
        # run GA and find best brain to give to testing
        ga = GA(self.light, graphics=True)
        ga_result = ga.run_offline(self.net, self.vehicle_first_move, look_ahead,
                                   veh_pos=self.random_vehicle.pos[-1],
                                   veh_angle=self.random_vehicle.angle, individuals=individuals,
                                   generations=generations, crossover_rate=0.6, mutation_rate=0.3)
        self.brain = ga_result[0]
        predicted_sensors = ga_result[1]
        predicted_wheels = ga_result[2]

        # Create vehicle and pass it the wheel data, previous random movement, and then run. Get the pos data
        ga_prediction_vehicle = ControllableVehicle(self.random_vehicle.pos[-1], self.random_vehicle.angle, self.light)
        ga_prediction_vehicle.set_wheels(predicted_wheels)
        ga_prediction_vehicle.random_movement = self.random_vehicle.pos
        ga_prediction_vehicle = self.sim.run_simulation(len(predicted_wheels), graphics=True, cycle='sleep',
                                                        vehicle=ga_prediction_vehicle)
        self.predicted_pos = ga_prediction_vehicle.pos

        # Error graph with MSE. Get sensory information of vehicle and compare with predicted
        v_iter = np.array(range(0, len(ga_prediction_vehicle.sensor_left)))  # GA predicted vehicle iterations

        plt.figure(1)
        plt.suptitle('Graph of predicted vs actual sensor values and Mean Squared Error. Lookahead:' + str(look_ahead))
        plt.subplot(221)
        plt.title('Left sensor values')
        plt.xlabel('iterations')
        plt.ylabel('sensor value')
        plt.plot(v_iter, ga_prediction_vehicle.sensor_left, 'b', label='actual')
        plt.plot(v_iter, predicted_sensors[0], 'g', label='predicted')
        plt.legend()

        plt.subplot(222)
        plt.title('Right sensor values')
        plt.xlabel('iterations')
        plt.ylabel('sensor value')
        plt.plot(v_iter, ga_prediction_vehicle.sensor_right, 'b', label='actual')
        plt.plot(v_iter, predicted_sensors[1], 'g', label='predicted')
        plt.legend()

        plt.subplot(223)
        plt.title('Left sensor MSE')
        plt.xlabel('iterations')
        plt.ylabel('mean squared error')
        msel = [((predicted_sensors[0][it] - ga_prediction_vehicle.sensor_left[it]) ** 2) / len(v_iter) for it in
                range(0, len(v_iter))]
        plt.plot(v_iter, msel)

        plt.subplot(224)
        plt.title('Right sensor MSE')
        plt.xlabel('iterations')
        plt.ylabel('mean squared error')
        mser = [((predicted_sensors[1][it] - ga_prediction_vehicle.sensor_right[it]) ** 2) / len(v_iter) for it in
                range(0, len(v_iter))]
        plt.plot(v_iter, mser)
        plt.show()

    def sleep_wake(self, random_movements=50, cycles=2,  look_ahead=100, individuals=25, generations=10):
        brains = []
        # Perform the initial random movement, first wake phase
        self.wake_learning(random_movements)

        past_sensor_motor = self.format_movement_data(self.random_vehicle)
        # set the individuals and the generations
        # for the ga of every cycle
        # TODO: This variables could decrease over time when getting closer and closer to the light
        self.ga_individuals = individuals
        self.ga_generations = generations

        # intialize vehicle position , movement and angle
        past_pos = self.random_vehicle.pos
        last_pos = self.random_vehicle.pos[-1]
        last_angle = self.random_vehicle.angle

        ga = GA(self.light, graphics=False)

        for _ in range(0, cycles):
            # SLEEP NOW
            # run GA and find best brain to give to testing
            ga_result = ga.run_offline(self.net, past_sensor_motor, look_ahead,
                                       veh_pos=last_pos,
                                       veh_angle=last_angle, individuals=individuals,
                                       generations=generations, crossover_rate=0.6, mutation_rate=0.3)
            self.brain = ga_result[0]

            # add on top of the old brain
            brains.append(self.brain)
            self.net.set_brains(brains)
            _predicted_sensors = ga_result[1]

            predicted_wheels = ga_result[2]

            # Create vehicle and pass it the wheel data, previous random movement, and then run. Get the pos data
            ga_prediction_vehicle = BrainVehicle(last_pos, last_angle, self.light)
            ga_prediction_vehicle.set_values(self.brain)
            ga_prediction_vehicle.set_brains(brains)
            ga_prediction_vehicle.random_movement = past_pos
            steps = len(predicted_wheels)
            # WAKE UP AND ACT
            last_wake_vehicle = self.sim.run_simulation(steps, graphics=True, cycle='sleep',
                                                            vehicle=ga_prediction_vehicle)

            #self.benchmark_tests(random_brains=1000, ga_graphics=True)
            # TODO: Train a new model when brain incorporation works
            Sprites.world_brains = brains
            self.train_network('pyrenn', 30, 200, [4, 20, 20, 2], 30, 30, graphics=False)


            # update parameters from where the vehicle stopped
            last_sensor_motor = self.format_movement_data(last_wake_vehicle)
            past_sensor_motor = np.concatenate((past_sensor_motor, last_sensor_motor), axis=1)
            past_pos.extend(last_wake_vehicle.pos)
            last_pos = last_wake_vehicle.pos[-1]
            last_angle = last_wake_vehicle.angle


    def format_movement_data(self, vehicle):
        steps = len(vehicle.motor_left)
        vehicle_move = [[vehicle.motor_left[0], vehicle.motor_right[0],
                         vehicle.sensor_left[0], vehicle.sensor_right[0]]]
        for t in range(1, steps):
            vehicle_move.append([vehicle.motor_left[t], vehicle.motor_right[t],
                                 vehicle.sensor_left[t], vehicle.sensor_right[t]])
        vehicle_latest_move = []
        for t in range(0, len(vehicle_move)):
            vehicle_latest_move.append(np.transpose(np.array(vehicle_move[t])))
        return np.transpose(np.array(vehicle_latest_move))


    def wake_testing(self, iterations, benchmark=True):
        """ This phase uses the control system to iterate through many motor commands by passing them to the controlled
        robot in the world and retrieving its sensory information """
        self.after_ga_movements = iterations
        new_vehicle = BrainVehicle(self.random_vehicle.pos[-1], self.random_vehicle.angle, self.light)
        new_vehicle.set_values(self.brain)
        new_vehicle.random_movement = self.random_vehicle.pos
        new_vehicle.predicted_movement = self.predicted_pos

        self.actual_vehicle = self.sim.run_simulation(iteration=iterations, graphics=True, cycle='wake (testing)',
                                                      vehicle=new_vehicle)

        # get positional information of vehicle and compare with predicted
        plt.figure(1)
        plt.suptitle('Predicted vs actual movement in wake (testing) cycle')
        plt.subplot(121)
        plt.title('Predicted vs actual movement in space')
        plt.xlabel('x coordinate')
        plt.ylabel('y coordinate')
        actual_move = np.transpose(self.actual_vehicle.pos)
        pred_move = np.transpose(self.predicted_pos)
        plt.scatter(actual_move[0], actual_move[1], s=7, c='r', label='actual')
        plt.scatter(pred_move[0], pred_move[1], s=10, c='g', label='predicted')
        plt.plot(actual_move[0], actual_move[1], 'r')
        plt.plot(pred_move[0], pred_move[1], 'g')
        plt.legend()

        plt.subplot(122)
        plt.title('Movement prediction error')
        plt.xlabel('iterations')
        plt.ylabel('mean squared error')
        mse = [(((pred_move[0][it] - actual_move[0][it] + pred_move[1][it] - actual_move[1][it]) / 2) ** 2) /
               len(pred_move[0]) for it in range(0, len(pred_move[0]))]
        plt.plot(range(0, len(mse)), mse)

        plt.show()

        if benchmark:
            self.benchmark_tests()
