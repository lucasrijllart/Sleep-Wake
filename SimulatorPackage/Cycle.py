import time, datetime, re
from Simulator import Simulator
from Narx import PyrennNarx, NarxMLP
import Narx as narx
import matplotlib.pyplot as plt
import Genetic
from Genetic import GA as GA
import Sprites
from Sprites import *
import os.path


def pre_process_by_vehicle(raw_data):
    """
    Returns the data in a list of vehicles. Every vehicle has a list of timesteps with each motor and sensor.
    :param raw_data: the raw data of the vehicles
    :return: formatted list of vehicles and their values at every timestep
    """
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

        self.vehicle_training_data = None  # this data will hold all the vehicles that the network will train with
        self.ga_test_data = None  # this data will hold all the vehicles that the GA will test the fitness from
        self.init_pos = None  # first position of vehicle before random collection
        self.init_angle = None  # first angle of vehicle before random collection
        self.network_delay = None  # for show_error_graph() to know when the network is predicting on predictions
        self.collection_vehicle_pos = []  # all pos of collection vehicle to pass to wakeTest vehicle to show on Sim
        self.starting_pos_after_collect = None  # last position of vehicle after collection (passed to GA)
        self.starting_ang_after_collect = None  # last angle of vehicle after collection (passed to GA)

        self.random_vehicle = None
        self.brain = None  # Vehicle brain assigned after GA, 6 weights
        self.vehicle_first_move = None
        self.predicted_pos = None

        self.ga_individuals = None
        self.ga_generations = None

        self.required_distance_from_light = 500

        if light_pos is None:
            light_pos = [1100, 600]
        self.light = Light(light_pos)
        self.sim = Simulator(self.light)

        self.actual_vehicle = None

        self.count_cycles = 0

    def find_random_pos(self):
        """
        Returns a random position and angle where the position is far enough from the light.
        :return: position and angle [[pos_x, pos_y], angle]
        """
        dist_to_light, rand_x, rand_y = 0, 0, 0
        while dist_to_light < self.required_distance_from_light:
            rand_x = random.randint(RandomMotorVehicle.radius*2, Simulator.window_width-RandomMotorVehicle.radius*2)
            rand_y = random.randint(RandomMotorVehicle.radius*2, Simulator.window_height-RandomMotorVehicle.radius*2)
            dist_to_light = np.sqrt((rand_x - self.light.pos[0]) ** 2 + (rand_y - self.light.pos[1]) ** 2)
        return [rand_x, rand_y], random.randint(0, 360)

    def collect_training_data(self, rand_vehicle_pos=False, runs=10, iterations=1000, data_collection_graphics=False,
                              seed=None):
        """ Runs many vehicles in simulations and collects their sensory and motor information

        :param rand_vehicle_pos: if the second vehicle doesn't take the previous one's position
        :param runs: number of vehicle runs to execute
        :param iterations: number of iterations of a run
        :param data_collection_graphics: show the vehicles while collecting data
        :param seed: seed to make the collection of data identically random
        :return: data collected and pre-processed
        """
        if seed is not None:
            random.seed(seed)
        data = []
        sim = Simulator(self.light)
        v = None
        # get new random position for initial starting position of vehicle
        vehicle_pos, vehicle_angle = self.find_random_pos()
        self.init_pos = vehicle_pos
        self.init_angle = vehicle_angle

        for run in range(0, runs):  # run simulation for number of runs we have
            # run simulation
            v = sim.quick_simulation(iterations, data_collection_graphics, vehicle_pos, vehicle_angle,
                                     self.collection_vehicle_pos)
            vehicle_data_in_t = []

            # get new position
            if rand_vehicle_pos:  # needs to be further than 500
                vehicle_pos, vehicle_angle = self.find_random_pos()
            else:
                vehicle_pos = v.pos[-1]
                vehicle_angle = v.angle

            for t in range(0, iterations):
                vehicle_data_in_t.append([v.motor_left[t], v.motor_right[t], v.sensor_left[t], v.sensor_right[t]])
            data.append(vehicle_data_in_t)
            self.collection_vehicle_pos.extend(v.pos)  # adds the pos to the previous pos
        veh_is_rand = 'random' if rand_vehicle_pos else 'continuous'
        print '\nCollected data from %d %s vehicles over %d iterations' % (runs, veh_is_rand, iterations)
        self.starting_pos_after_collect = v.pos[-1]  # keeps track of last position after collection
        self.starting_ang_after_collect = v.angle  # keeps track of last angle after collection
        return pre_process_by_vehicle(data)

    def show_error_graph(self, veh_pos=None, veh_angle=None, testing_time=300, predict_after=50, brain=None, seed=None, graphics=True):
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
        # vehicle_pos, vehicle_angle = self.find_random_pos()
        # use initial position of training data
        if veh_pos is None and veh_angle is None:
            veh_pos, veh_angle = self.find_random_pos()

        # execute vehicle
        vehicle = sim.init_simulation(testing_time, graphics, veh_pos=veh_pos, veh_angle=veh_angle, brain=brain)
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
                pred_with_data = sensor_log[0][:self.network_delay]
                i2_data = i2[:self.network_delay]
                pred_on_pred = sensor_log[0][self.network_delay-1:]
                i2_pred = i2[self.network_delay-1:]
                plt.plot(i, vehicle.sensor_left, 'b', label='real vehicle values')
                plt.plot(i2_data, pred_with_data, 'r', label='predictions with real data')
                plt.plot(i2_pred, pred_on_pred, 'r--', label='predictions without real data')
                plt.legend()

                plt.subplot(222)
                plt.title('Right sensor values')
                pred_with_data = sensor_log[1][:self.network_delay]
                i2_data = i2[:self.network_delay]
                pred_on_pred = sensor_log[1][self.network_delay-1:]
                i2_pred = i2[self.network_delay-1:]
                plt.plot(i, vehicle.sensor_right, 'b', label='real vehicle values')
                plt.plot(i2_data, pred_with_data, 'r', label='predictions with real data')
                plt.plot(i2_pred, pred_on_pred, 'r--', label='predictions without real data')
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

    def benchmark_tests(self, vehicle_pos, vehicle_angle, random_brains=1000, ga_graphics=True):
        """
        Shows a graph with the fitness of the predicted vehicle, the evolved vehicle and a number of random brains
        :param random_brains: number of random brains to run to get a good benchmark curve
        :param ga_graphics: set to True to show online GA and
        :return:
        """
        iterations = self.after_ga_movements
        fitnesses = []

        print '\nStarting benchmark test for %d random brains...' % random_brains
        start_time = time.time()
        for individual in range(0, random_brains):
            brain = Genetic.make_random_brain()
            fitnesses.append(Genetic.get_fitness(vehicle_pos, vehicle_angle, brain, iterations, self.light))
        print 'Collected %d random brains in %ds' % (random_brains, time.time() - start_time)
        random_mean_fit = np.mean(fitnesses)

        ga = GA(self.light, ga_graphics)
        brain = ga.run(vehicle_pos, vehicle_angle, self.ga_individuals, self.ga_generations, iterations)
        brain = brain[0]
        evolved_score = Genetic.get_fitness(vehicle_pos, vehicle_angle, brain, iterations, self.light)
        fitnesses.append(evolved_score)

        brain = self.actual_vehicle.get_brain()
        predicted_score = Genetic.get_fitness(vehicle_pos, vehicle_angle, brain, iterations, self.light)
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
        ga_time = len(self.collection_vehicle_pos) + self.ga_individuals * self.ga_generations * iterations
        pred_time = self.training_runs * self.training_time + len(self.collection_vehicle_pos) + self.after_ga_movements
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
        self.training_runs, self.training_time, self.network_delay = learning_runs, learning_time, delay
        self.net_filename = 'narx/r%dt%dd%de%d' % (learning_runs, learning_time, delay, max_epochs)

        # collect data for NARX and testing and pre-process, separate into training and testing
        train_input = self.collect_training_data(True, learning_runs, learning_time, graphics, seed)
        np.random.shuffle(train_input)
        self.vehicle_training_data = train_input[:len(train_input)/2]
        self.ga_test_data = train_input[len(train_input)/2:]

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

    def test_network(self, tests=50, test_time=100, seed=1, graphics=False):
        """ Function that tests a network's error on many random tests
        :param tests: number of tests to run (100-200 takes the right amount of time)
        :param test_time: usually around 100-200 (should be the same as network has been trained on)
        :param seed: makes all test vehicles the same random set for every network tested
        :param graphics: shows the graph of network accuracy
        :return: nothing
        """
        print '\nTesting network %s for %d times...' % (self.net_filename, tests)
        if seed is not None:
            random.seed(seed)
        combined_error = []
        for it in range(0, tests):
            if it % 10 == 0:
                print '\rTest %d/%d' % (it, tests),
            combined_error.append(self.show_error_graph(testing_time=test_time, graphics=False))
        combined_error_mean = sum(combined_error) / tests
        random.seed(None)  # reset seed to random
        print '\rFinished network test. Average error: %s' % combined_error_mean

        if graphics:
            plt.title('Testing network %s for %d tests of %d timesteps.\nAverage error:%s' %
                      (self.net_filename, tests, test_time, combined_error_mean))
            plt.scatter(range(0, len(combined_error)), np.sort(combined_error), s=2, label='test error')
            plt.plot([0, len(combined_error)], [combined_error_mean, combined_error_mean], c='black', label='mean error')
            plt.xlabel('test number')
            plt.ylabel('Sum of MSE of both sensors')
            plt.legend()
            plt.show()

    def wake_learning(self, random_movements):
        """ Create a vehicle and run for some time-steps """
        self.random_movements = random_movements
        # Create vehicle in simulation
        self.sim = Simulator(self.light)
        # give last position of training to vehicle pos
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

    def sleep(self, vehicle_pos=None, vehicle_ang=None, look_ahead=100, individuals=25, generations=10, td=None):
        self.ga_individuals = individuals
        self.ga_generations = generations
        if self.ga_test_data is None:
            data = self.vehicle_first_move
            test_data = td
            vehicle_pos = self.random_vehicle.pos[-1]
            vehicle_ang = self.random_vehicle.angle
        else:
            data = self.ga_test_data[-1]
            test_data = self.ga_test_data

        # run GA and find best brain to give to testing
        ga = GA(self.light, graphics=False)
        ga_result = ga.run_offline(self.net, data, test_data, look_ahead,
                                   veh_pos=vehicle_pos, veh_angle=vehicle_ang, individuals=individuals,
                                   generations=generations, crossover_rate=0.6, mutation_rate=0.3)
        self.brain = ga_result[0]
        predicted_sensors = ga_result[1]
        predicted_wheels = ga_result[2]

        # Create vehicle and pass it the wheel data, previous random movement, and then run. Get the pos data
        ga_prediction_vehicle = ControllableVehicle(vehicle_pos, vehicle_ang, self.light)
        ga_prediction_vehicle.set_wheels(predicted_wheels)
        ga_prediction_vehicle.random_movement = self.collection_vehicle_pos
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

    def wake_testing(self, vehicle_pos, vehicle_angle, iterations, benchmark=True):
        """ This phase uses the control system to iterate through many motor commands by passing them to the controlled
        robot in the world and retrieving its sensory information """
        self.after_ga_movements = iterations
        new_vehicle = BrainVehicle(vehicle_pos, vehicle_angle, self.light)
        new_vehicle.set_values(self.brain)
        new_vehicle.random_movement = self.collection_vehicle_pos
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
        # TODO: fix this so that it can have a lookahead bigger than wake test iter
        plt.title('Movement prediction error')
        plt.xlabel('iterations')
        plt.ylabel('mean squared error')
        mse = [(((pred_move[0][it] - actual_move[0][it] + pred_move[1][it] - actual_move[1][it]) / 2) ** 2) /
               len(pred_move[0]) for it in range(0, len(pred_move[0]))]
        plt.plot(range(0, len(mse)), mse)

        plt.show()

        if benchmark:
            self.benchmark_tests(vehicle_pos, vehicle_angle)

    def retrain_with_brain(self):
        print 'Got brain: ' + str(self.brain)
        Sprites.world_brain = self.brain
        self.train_network('pyrenn', 50, 100, [4, 20, 20, 2], 20, 20, graphics=False)

    def assign_testing_as_initial(self):
        vehicle_move = [[self.actual_vehicle.motor_left[0], self.actual_vehicle.motor_right[0],
                         self.actual_vehicle.sensor_left[0], self.actual_vehicle.sensor_right[0]]]
        for t in range(1, len(self.actual_vehicle.motor_left)):
            vehicle_move.append([self.actual_vehicle.motor_left[t], self.actual_vehicle.motor_right[t],
                                 self.actual_vehicle.sensor_left[t], self.actual_vehicle.sensor_right[t]])
        vehicle_first_move = []
        for t in range(0, len(vehicle_move)):
            vehicle_first_move.append(np.transpose(np.array(vehicle_move[t])))
        self.vehicle_first_move = np.transpose(np.array(vehicle_first_move))
        self.starting_pos_after_collect = self.actual_vehicle.pos[-1]
        self.starting_ang_after_collect = self.actual_vehicle.angle

    def run_2_cylces(self, look_ahead, individuals, generations, wake_test_iter):
        self.sleep(self.starting_pos_after_collect, self.starting_ang_after_collect, look_ahead, individuals,
                    generations)

        self.wake_testing(wake_test_iter, benchmark=False)

        self.retrain_with_brain()

        self.show_error_graph(graphics=False)

        self.test_network(graphics=False)

        self.sleep(self.actual_vehicle.pos[-1], self.actual_vehicle.angle, 100, individuals, generations)

        self.wake_testing(self.init_pos, self.init_angle, 400)
