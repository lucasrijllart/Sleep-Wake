import time, datetime, os, random
import numpy as np
import matplotlib.pyplot as plt
import Genetic
from Genetic import GA
import Cycle
from Cycle import Cycles
from Simulator import Simulator
from Sprites import Light, BrainVehicle


def _save_to_file(test_name):
    """
    Function that returns an unused filename to save a new figure.
    :param test_name: name of the test
    :return: unused name
    """
    count = 0
    new_filename = test_name
    extension = '.png'
    while os.path.exists(new_filename + extension):
        count += 1
        new_filename = new_filename + '_v' + str(count)
    return test_name + '_v' + str(count)


class Tests:

    def __init__(self):
        self.light = Light([Simulator.window_width/2, Simulator.window_height/2])

    def test_1(self, iterations=300, random_brains=500, evolved_brains=10):
        """
        Test 1: Evolution of a Braitenberg vehicle. This tests the evolution of a vehicle, the GA fitness over time,
        the fitness of the GA compared to random brains, and the average GA fitness compared to average random.
        :param iterations: number of iterations for GA and random
        :param random_brains: number of random brains
        :param evolved_brains: number of evolved brains
        """
        print 'Starting Test 1: Evolution of Braitenberg vehicles'
        # run 1 GA
        vehicle_pos, vehicle_angle = Cycle.find_random_pos(self.light)
        ga = GA(self.light, graphics=True)
        ga_brain = ga.run_with_simulation(vehicle_pos, vehicle_angle, None, iterations=iterations)

        # random brains from same position
        random_fitnesses = []
        temp_fitness = 0
        best_random_brain = []
        print '\nStarting benchmark test for %d random brains...' % random_brains
        start_time = time.time()
        for individual in range(0, random_brains):
            brain = Genetic.make_random_brain()
            fitness = Genetic.get_fitness(vehicle_pos, vehicle_angle, brain, iterations, self.light)
            if fitness > temp_fitness:
                best_random_brain.append(brain)
            random_fitnesses.append(fitness)
        print 'Collected %d random brains in %ds' % (random_brains, time.time() - start_time)
        random_mean_fit = np.mean(random_fitnesses)
        print 'best brains:\n' + str(best_random_brain)
        brain = ga_brain[0]
        evolved_score = Genetic.get_fitness(vehicle_pos, vehicle_angle, brain, iterations, self.light)
        random_fitnesses.append(evolved_score)
        # add best brain
        random_fitnesses.sort()

        # Evolved brains benchmark test with randoms
        evolved_fitnesses = []
        random_random_fitnesses = []
        ga = GA(self.light, graphics=False, verbose=False)
        print '\nStarting benchmark test for %d evolved brains...' % evolved_brains
        start_time = time.time()
        for individual in range(0, evolved_brains):
            print '\rProgress: %d/%d' % (individual, evolved_brains),
            vehicle_pos, vehicle_angle = Cycle.find_random_pos(self.light)
            ga_brain = ga.run_with_simulation(vehicle_pos, vehicle_angle, iterations=iterations)
            fitness = Genetic.get_fitness(vehicle_pos, vehicle_angle, ga_brain[0], iterations, self.light)
            for _ in range(0, 10):  # do 10 random brain tests from this position
                random_random_fitnesses.append(Genetic.get_fitness(vehicle_pos, vehicle_angle,
                                                                   Genetic.make_random_brain(), iterations, self.light))
            evolved_fitnesses.append(fitness)
        print '\rCollected %d evolved brains in %ds' % (evolved_brains, time.time() - start_time)
        evolved_mean_fit = np.mean(evolved_fitnesses)
        random_random_avg = np.mean(random_random_fitnesses)
        evolved_fitnesses.sort()

        # graph
        plt.figure(1)
        plt.subplot(121)
        plt.title('Fitness test of evolved vehicle and %d random brain vehicles\nMean: %s' % (random_brains,
                                                                                              format(random_mean_fit,
                                                                                                     '.3f')))
        evol_idx = np.where(random_fitnesses == evolved_score)
        plt.scatter(range(0, len(random_fitnesses)), random_fitnesses, s=1, c='grey', label='random')
        plt.scatter(evol_idx, evolved_score, s=15, c='green', label='evolved')
        plt.plot([0, len(random_fitnesses)], [random_mean_fit, random_mean_fit], c='blue', label='random mean fitness')
        plt.xlabel('individuals')
        plt.ylabel('fitness')
        plt.legend()

        plt.subplot(122)
        plt.title('Fitness test of %d evolved vehicles\nMean: %s' % (evolved_brains, format(evolved_mean_fit, '.3f')))
        plt.scatter(range(0, len(evolved_fitnesses)), evolved_fitnesses, s=3, c='green', label='evolved')
        plt.plot([0, len(evolved_fitnesses)], [evolved_mean_fit, evolved_mean_fit], c='orange',
                 label='evolved mean fitness')
        plt.plot([0, len(evolved_fitnesses)], [random_random_avg, random_random_avg], c='blue',
                 label='random mean fitness')
        plt.xlabel('evolved individuals')
        plt.ylabel('fitness')
        plt.legend()

        plt.show()

    def test_2_1(self, net1, net2, predict_after, iterations, net1delay, net2delay, seed=8):
        """
        Test 2 part 1: Environment modelling - identical initial conditions
        :param net1: first network trained with 10 delays
        :param net2: second network trained with 40 delays
        :param predict_after: timestep to predict after
        :param iterations: total number of iterations for predictions
        :param net1delay: delays of network 1: 10
        :param net2delay: delays of network 2: 40
        :param seed:
        :return:
        """
        print 'Starting Test 2 part 1'
        cycle = Cycles(self.light.pos, net_filename=net1)
        cycle.show_error_graph([300, 300], 200, iterations, predict_after, seed=seed, graphics=False)
        first_sensors = cycle.error_sensor_log[0]

        cycle = Cycles(self.light.pos, net_filename=net2)
        cycle.show_error_graph([300, 300], 200, iterations, predict_after, seed=seed, graphics=False)
        second_sensors = cycle.error_sensor_log[0]

        plt.figure(1)
        plt.suptitle('Graph showing predicted vs actual left sensor values for two different networks')

        i = np.array(range(0, len(cycle.real_sensor_log)))
        i2 = np.array(range(predict_after, iterations))

        pred_with_data = first_sensors[:net1delay]
        i2_data = i2[:net1delay]
        pred_on_pred = first_sensors[net1delay - 1:]
        i2_pred = i2[net1delay - 1:]

        pred_with_data2 = second_sensors[:net2delay]
        i2_data2 = i2[:net2delay]
        pred_on_pred2 = second_sensors[net2delay - 1:]
        i2_pred2 = i2[net2delay - 1:]

        plt.plot(i, cycle.real_sensor_log, 'b', label='real vehicle values')
        plt.plot(i2_data, pred_with_data, 'r', label='predictions with real data')
        plt.plot(i2_pred, pred_on_pred, 'r--', label='predictions without real data')
        plt.plot(i2_data2, pred_with_data2, 'g', label='predictions with real data')
        plt.plot(i2_pred2, pred_on_pred2, 'g--', label='predictions without real data')
        plt.legend()

        # test of network prediction into the unknown
        test_iter = range(60, 200, 10)
        cycle = Cycles(self.light.pos, net_filename=net1)
        first_net_avg = []
        for iteration in test_iter:
            fit1 = cycle.test_network(iteration, iteration, verbose=False)
            fit2 = cycle.test_network(iteration, iteration * 2, verbose=False)
            first_net_avg.append((fit1 + fit2) / 2)

        cycle = Cycles(self.light.pos, net_filename=net2)
        second_net_avg = []
        for iteration in test_iter:
            fit1 = cycle.test_network(iteration, iteration, verbose=False)
            fit2 = cycle.test_network(iteration, iteration * 2, verbose=False)
            second_net_avg.append((fit1 + fit2) / 2)

        difference = [abs(first_net_avg[i] - second_net_avg[i]) for i in range(0, len(first_net_avg))]

        plt.figure(2)
        plt.title('Average error of two networks over number of iterations tested')
        plt.scatter(test_iter, first_net_avg, c='r', s=6)
        plt.plot(test_iter, first_net_avg, 'r', label='10 delays')
        plt.scatter(test_iter, second_net_avg, c='g', s=6)
        plt.plot(test_iter, second_net_avg, 'g', label='40 delays')
        plt.plot(test_iter, difference, '--', c='grey', label='difference of accuracy')
        plt.xlabel('total vehicle iterations')
        plt.ylabel('average error over 100 tests')
        plt.legend()
        plt.show()

    def test_2_2(self, best_net, num_of_tests):
        print 'Starting Test 2 part 2'
        # test 1 - prediction of a vehicle
        if True:
            cycle = Cycles(self.light.pos, net_filename=best_net)
            cycle.show_error_graph(testing_time=500, predict_after=50, seed=2, graphics=True)

            # test 2
            cycle.test_network(2, 200, seed=8, graphics=True)  # 5,

        # test 3 - randomly trained network vs identically trained network tested on random vehicles
        if True:
            start_time = time.time()
            num_of_tests = range(0, num_of_tests)
            cycle = Cycles(self.light.pos)
            random_trained_net_error = []
            identic_trained_net_error = []

            for i in num_of_tests:
                cycle.train_network('pyrenn', 20, 50, [4, 20, 20, 2], 10, 50, seed=i, random_pos=True, save_net=False)
                random_trained_net_error.append(cycle.test_network(100, 50, seed=i + 50, predict_after=10, verbose=False))
                cycle.train_network('pyrenn', 20, 50, [4, 20, 20, 2], 10, 50, seed=i, random_pos=False, save_net=False)
                identic_trained_net_error.append(cycle.test_network(100, 50, seed=i + 50, predict_after=10, verbose=False))

            # get difference between the two network's error
            diff_array = [(i, abs(random_trained_net_error[i] - identic_trained_net_error[i])) for i in num_of_tests]
            dtype = [('idx', int), ('diff', float)]  # create new type (idx, diff)
            diff_array = np.array(diff_array, dtype=dtype)  # convert diff_array to numpy array structure
            diff_array = np.sort(diff_array, order='diff')  # sort array based on difference
            new_sorting = [i[0] for i in diff_array]  # get the new, sorted indexes
            # pick out the values from the networks based on the new indexes (based on difference)
            new_random = [random_trained_net_error[i] for i in new_sorting]
            random_mean = np.mean(new_random)
            new_identic = [identic_trained_net_error[i] for i in new_sorting]
            identic_mean = np.mean(new_identic)
            print '\nTest 2 part 2 graph 3 time taken: %s' % str(datetime.timedelta(seconds=time.time() - start_time))
            print 'Random   mean: %s' % random_mean
            print 'Identic  mean: %s' % identic_mean

            plt.title('Measured error of models trained with random initial conditions\n'
                      'and identical initial conditions tested on random vehicles')
            plt.scatter(num_of_tests, new_random, c='orange')
            plt.scatter(num_of_tests, new_identic, c='blue')
            plt.plot(num_of_tests, new_random, c='orange', label='random initial conditions training')
            plt.plot([0, len(num_of_tests)], [random_mean, random_mean], '--', color='orange',
                     label='random net average error')
            plt.plot(num_of_tests, new_identic, c='blue', label='same initial conditions training')
            plt.plot([0, len(num_of_tests)], [identic_mean, identic_mean], '--', color='blue',
                     label='identical net average error')
            plt.xlabel('test number')
            plt.ylabel('combined MSE for 100 random trajectories')
            plt.legend()
            plt.show()

    def test_2_3(self, num_of_tests):
        print '\nStarting Test 2 part 3'
        # test 1 - random, continuous and identical trained network tested on random vehicles
        start_time = time.time()
        num_of_tests = range(0, num_of_tests)
        cycle = Cycles(self.light.pos)
        random_trained_net_error = []
        identic_trained_net_error = []
        cont_trained_net_error = []

        for i in num_of_tests:
            print '\n- - Test %s/%s  Time elapsed: %s' % (i, len(num_of_tests),
                                                          str(datetime.timedelta(seconds=time.time() - start_time)))
            cycle.train_network('pyrenn', 30, 50, [4, 20, 20, 2], 10, 100, seed=i + 200, random_pos=True, save_net=False)
            random_trained_net_error.append(
                cycle.test_network(100, 40, seed=i + 50, predict_after=10, verbose=False))

            cycle.train_network('pyrenn', 30, 50, [4, 20, 20, 2], 10, 100, seed=i + 200, random_pos=False, save_net=False)
            identic_trained_net_error.append(
                cycle.test_network(100, 40, seed=i + 50, predict_after=10, verbose=False))

            cycle.train_network('pyrenn', 30, 50, [4, 20, 20, 2], 10, 100, seed=i + 200, save_net=False, continuous=True)
            cont_trained_net_error.append(
                cycle.test_network(100, 40, seed=i + 50, predict_after=10, verbose=False))

        # get difference between the two network's error
        diff_array = [(i, abs(random_trained_net_error[i] - cont_trained_net_error[i])) for i in num_of_tests]
        dtype = [('idx', int), ('diff', float)]  # create new type (idx, diff)
        diff_array = np.array(diff_array, dtype=dtype)  # convert diff_array to numpy array structure
        diff_array = np.sort(diff_array, order='diff')  # sort array based on difference
        new_sorting = [i[0] for i in diff_array]  # get the new, sorted indexes
        # pick out the values from the networks based on the new indexes (based on difference)
        new_random = [random_trained_net_error[i] for i in new_sorting]
        random_mean = np.mean(new_random)
        new_identic = [identic_trained_net_error[i] for i in new_sorting]
        identic_mean = np.mean(new_identic)
        new_cont = [cont_trained_net_error[i] for i in new_sorting]
        cont_mean = np.mean(new_cont)
        new_random = np.sort(new_random)
        new_identic = np.sort(new_identic)
        new_cont = np.sort(new_cont)
        print '\nTest 2 part 2 graph 3 time taken: %s' % str(datetime.timedelta(seconds=time.time() - start_time))
        print 'Random   mean: %s' % random_mean
        print 'Identic  mean: %s' % identic_mean
        print 'Continu  mean: %s' % cont_mean

        plt.title('Measured error of models trained with random, continuous, and identical initial conditions\n'
                  'tested on the same 100 random trajectories of 50 iterations')
        plt.plot(num_of_tests, new_random, c='orange', label='random initial conditions')
        plt.plot([0, len(num_of_tests)], [random_mean, random_mean], '--', color='orange',
                 label='random average error')
        plt.plot(num_of_tests, new_identic, c='blue', label='same initial conditions')
        plt.plot([0, len(num_of_tests)], [identic_mean, identic_mean], '--', color='blue',
                 label='identical average error')
        plt.plot(num_of_tests, new_cont, c='green', label='continuous initial conditions')
        plt.plot([0, len(num_of_tests)], [cont_mean, cont_mean], '--', color='green',
                 label='continuous average error')

        plt.xlabel('test number')
        plt.ylabel('combined MSE for 100 random trajectories')
        plt.legend()
        plt.savefig(_save_to_file('Test2p3'))

    def test_2_4(self):
        """
        Test 2 part 4. Compares a continuous-trained network's accuracy from its last training point and random
        positions
        """
        print 'Starting Test 2 part 4'
        cycle = Cycles(self.light.pos)

        cycle.train_network('pyrenn', 50, 50, [4, 20, 20, 2], 10, 200, continuous=True)

        cycle.test_network(500, 100, veh_pos_angle=[cycle.pos_after_collect, cycle.ang_after_collect])

        cycle.test_network(500, 100)

    def test_3_1(self, net_name):
        print 'Starting Test 3 part 1'
        # test 1 - GA result
        cycle = Cycles(self.light.pos, net_filename=net_name)
        cycle.wake_learning(random_movements=50)

        cycle.sleep(look_ahead=100, individuals=25, generations=20)

        cycle.wake_testing(cycle.random_vehicle.pos[-1], cycle.random_vehicle.angle, 200, False)

        ga = GA(self.light)
        ga_result = ga.run_with_simulation(cycle.random_vehicle.pos[-1], cycle.random_vehicle.angle,
                                           cycle.random_vehicle.pos, iterations=200)
        ga_brain = ga_result[0]

        ga_vehicle = BrainVehicle(cycle.random_vehicle.pos[-1], cycle.random_vehicle.angle, self.light)
        ga_vehicle.set_values(ga_brain)
        cycle.sim.run_simulation(200, True, ga_vehicle)
        plt.plot([0, 1], [0, 1])
        plt.show()

    def test_3_2(self, net_name, tests):
        """
        3 tests, one to show the trajectory of a generalised vehicle, benchmark test with evolved and random,
        fitness comparison between evolved GA and simulated GA.
        :param net_name:
        :param num_of_tests
        :return:
        """
        print '\nStarting Test 3 part 2: Evolution of generalised Control System'
        if False:
            cycle = Cycles(self.light.pos, net_filename=net_name)
            cycle.wake_learning(random_movements=40)
            vehicle_pos, vehicle_angle = cycle.random_vehicle.pos[-1], cycle.random_vehicle.angle

            rand_test = cycle.collect_training_data(True, 3, 50, verbose=False)

            cycle.sleep(look_ahead=100, individuals=40, generations=20, td=rand_test)

            cycle.wake_testing(vehicle_pos, vehicle_angle, 200, True)

            ga = GA(self.light)
            rand_test = [Cycle.find_random_pos(self.light) for _ in range(0, 3)]
            ga_result = ga.run_with_simulation(vehicle_pos, vehicle_angle, cycle.random_vehicle.pos, iterations=200,
                                               test_data=rand_test)
            ga_brain = ga_result[0]

            ga_vehicle = BrainVehicle(vehicle_pos, vehicle_angle, self.light)
            ga_vehicle.set_values(ga_brain)
            cycle.sim.run_simulation(200, True, ga_vehicle)
            plt.plot([0, 1], [0, 1])
            plt.show()

        if True:
            start_time = time.time()
            num_of_tests = range(0, tests)
            cycle = Cycles(self.light.pos, net_filename=net_name)
            ga = GA(self.light, verbose=False)
            wake1_iter, wake2_iter, individuals, generations, sim_ga_iter = 50, 200, 25, 20, 200
            evolved_fit, predicted_fit, random_fit, evolved_time, predicted_time = [], [], [], [], []

            for i in num_of_tests:
                print '\rTest %s/%s  Time elapsed: %s' % (i, len(num_of_tests),
                                                          str(datetime.timedelta(seconds=time.time() - start_time))),
                # prediction vehicle
                cycle.wake_learning(wake1_iter, graphics=False)
                vehicle_pos, vehicle_angle = cycle.random_vehicle.pos[-1], cycle.random_vehicle.angle
                rand_test = cycle.collect_training_data(True, 3, 50, verbose=False)
                ga_result = ga.run_offline(cycle.net, cycle.vehicle_first_move, rand_test, 150, vehicle_pos,
                                           vehicle_angle, self.light.pos, individuals, generations)
                brain = ga_result[0]
                rand_test = [Cycle.find_random_pos(self.light) for _ in range(0, 3)]
                predicted_fit.append(Genetic.get_fitness(vehicle_pos, vehicle_angle, brain, wake2_iter, self.light))
                pred_time = cycle.training_runs * cycle.training_time + wake2_iter
                predicted_time.append(pred_time)
                # GA vehicle
                ga_result = ga.run_with_simulation(vehicle_pos, vehicle_angle, individuals=individuals,
                                                   generations=generations,iterations=sim_ga_iter, test_data=rand_test)
                brain = ga_result[0]
                evolved_fit.append(Genetic.get_fitness(vehicle_pos, vehicle_angle, brain, wake2_iter, self.light))
                evol_time = individuals * generations * sim_ga_iter + wake2_iter
                evolved_time.append(evol_time)
                # random vehicles
                for _ in range(0, 100):
                    random_fit.append(Genetic.get_fitness(vehicle_pos, vehicle_angle, Genetic.make_random_brain(), 200,
                                                          self.light, rand_test))
                # fitness test
                evolved_mean = np.mean(evolved_fit)
                predicted_mean = np.mean(predicted_fit)
                random_mean = np.mean(random_fit)
                new_evolved = np.sort(evolved_fit)
                new_predict = np.sort(predicted_fit)
                print 'Predict mean: %s' % predicted_mean
                print 'Evolved mean: %s' % evolved_mean
                print 'Random  mean: %s' % random_mean
                print 'Predict time: %s' % predicted_time
                print 'Evolved time: %s' % evolved_time

                plt.figure(1)
                plt.title('Fitness of prediction-evolved, world-evolved,\nand random vehicles for multiple models')
                plt.plot(num_of_tests, new_predict, color='red', label='predicted')
                plt.plot([0, len(num_of_tests)], [predicted_mean, predicted_mean], '--', color='red', label='predicted mean')
                plt.plot(num_of_tests, new_evolved, color='green', label='evolved')
                plt.plot([0, len(num_of_tests)], [evolved_mean, evolved_mean], '--', color='green', label='evolved mean')
                plt.plot([0, len(num_of_tests)], [random_mean, random_mean], '--', color='grey', label='random mean')
                plt.xlabel('test number')
                plt.ylabel('fitness of individual')
                plt.legend()
                plt.show()
