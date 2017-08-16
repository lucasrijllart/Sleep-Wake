from Simulator import Simulator
from Sprites import BrainVehicle, Light
import numpy as np
import random
import math
import matplotlib.pyplot as plt


def make_random_brain():
    """
    Creates a random brain depending on the genome scale and genome length.
    :return: the random brain
    """
    # make this -GA.genome_scale, GA.genome_scale
    return [random.uniform(0, GA.genome_scale) for _ in range(0, GA.genome_length)]


def get_fitness(start_pos, start_a, brain, iterations, light, test_data=None):
    """
    Returns the fitness of a vehicle using real-world simulation
    :param start_pos: vehicle starting position
    :param start_a: vehicle starting angle
    :param brain: vehicle brain
    :param iterations: number of iterations to run simulation for
    :param light: the light of the simulation
    :return: fitness of vehicle
    """
    # create sprites
    vehicle = BrainVehicle(start_pos, start_a, light)
    vehicle.set_values(brain)
    # create Simulation
    vehicle = Simulator(light).run_simulation(iterations, False, vehicle)
    fitness = np.mean(vehicle.sensor_left) + np.mean(vehicle.sensor_right)

    # check for other initial conditions
    if test_data is not None:
        for test in test_data:
            vehicle = BrainVehicle(test[0], test[1], light)
            vehicle.set_values(brain)
            vehicle = Simulator(light).run_simulation(iterations, False, vehicle)
            fitness += np.mean(vehicle.sensor_left) + np.mean(vehicle.sensor_right)
        fitness /= len(test_data) + 1
    return fitness


class GA:
    """
    Genetic Algorithm class, creates a population then uses random tournament selection to replace the loser with a
    crossed-over and mutated version of the winner. The GA runs for the number of generations specified and shows a
    graph of the evolution of the fitness over time.
    """

    genome_scale = 10  # scale of values of genes (ex: -10, 10)
    genome_length = 4  # number of genes, can be 4 or 6

    sim = None  # simulator field
    print_iterations = 20  # this value controls how often to print the GA progress

    def __init__(self, light, crossover_rate=0.6, mutation_rate=0.3, graphics=False, verbose=True):
        """
        Constructor for GA class, takes a light and graphics. Graphics will show the winner in real-world GA and the
        evolution of the fitness in real-world and predicted cases.
        :param light: the light of the simulation
        :param graphics: boolean to show graphics or not
        """
        self.graphics = graphics
        self.light = light
        self.sim = Simulator(self.light)
        # init values as None, as they will be rewritten in run or run_random
        self.start_x, self.start_y, self.start_a = None, None, None
        self.iterations = None  # number of iterations to run real-world vehicles for
        self.individuals = None  # number of individuals
        self.generations = None  # number of generations
        self.crossover_rate = crossover_rate  # rate of crossover for a gene (0.0 to 1.0)
        self.mutation_rate = mutation_rate  # rate of mutation for a gene (0.0 to 1.0)

        self.net = None  # NARX network used to predict values for fitness
        self.data = None
        self.test_data = None
        self.look_ahead = None
        self.offline = None
        self.verbose = verbose  # bool to print Starting/Finishing text

        # To check the values (can be removed when GA works)
        self.mean_fit = []
        self.max_fit = []
        self.min_fit = []
        self.smax = []
        self.sav = []
        self.difff = []

    def _show_fitness_graph(self, best_ind, best_fit):
        """
        Graph that shows the best fitness over time and the max, average, min over time.
        :param best_ind: the best individual from the population
        :param best_fit: the best fitness from the population
        :return:
        """
        plt.figure(1)
        if self.genome_length == 4:
            brain = '[%s,%s,%s,%s]' % (format(best_ind[1][0], '.1f'), format(best_ind[1][1], '.1f'),
                                       format(best_ind[1][2], '.1f'), format(best_ind[1][3], '.1f'))
        else:
            brain = '[%s,%s,%s,%s,%s,%s]' % (format(best_ind[1][0], '.1f'), format(best_ind[1][1], '.1f'),
                                             format(best_ind[1][2], '.1f'), format(best_ind[1][3], '.1f'),
                                             format(best_ind[1][4], '.1f'), format(best_ind[1][5], '.1f'))
        plt.suptitle('Finished GA of %d individuals and %d generations,\nwith fitness %s of brain: %s' % (
            self.individuals, self.generations, format(best_ind[2], '.3f'), brain))

        plt.title('Individual fitness over time')
        plt.xlabel('iterations')
        plt.ylabel('fitness of individuals')
        i = range(0, self.individuals * self.generations, self.print_iterations)
        plt.scatter(i, self.max_fit, s=4, label='max')
        plt.plot(i, self.max_fit)
        plt.scatter(i, self.mean_fit, s=4, label='mean')
        plt.plot(i, self.mean_fit)
        plt.scatter(i, self.min_fit, s=4, label='min')
        plt.plot(i, self.min_fit)
        plt.legend()

        plt.show()

    def _mutate(self, ind):  # genome: [ll, lr, rr, rl, bl, br]
        """
        Method to mutate an individual. Has a mutation rate variable and modifies a gene to 1% of the scale.
        :param ind: individual
        :return: the mutated individual
        """
        for i in range(0, len(ind)):
            if random.random() < self.mutation_rate:
                ind[i] += (random.gauss(0, 1) * self.genome_scale*2) / 100
                if ind[i] > self.genome_scale:
                    ind[i] = -self.genome_scale + (self.genome_scale - ind[i])
                if ind[i] < -self.genome_scale:  # make it -self.genome_scale for wrapping around backwards movement
                    ind[i] = self.genome_scale - (-self.genome_scale - ind[i])
        return ind

    def _perform_crossover(self, indwin, indlos):
        """
        Performs crossover on loser then mutates, is given a winner, loser, crossover rate and mutation rate.
        :param indwin: individual with higher fitness
        :param indlos: individual with lower fitness
        :return: winner and loser in an array
        """
        for i in range(0, len(indwin)):
            if random.random() < self.crossover_rate:
                indlos[i] = indwin[i]
        # mutate
        indlos = self._mutate(indlos)
        return [indwin, indlos]

    def _init_pool(self, individuals):
        """
        Creates a population of random individuals, then calculates their fitnesses. Used when GA starts.
        :param individuals: number of individuals in the population
        :return: the initialised population pool
        """
        pool = []
        for i in range(0, individuals):
            ind = make_random_brain()
            pool.append(ind)

        i = 0
        all_individuals = []
        for individual in pool:
            fitness = self._get_fitness(individual)
            all_individuals.append([i, individual, fitness])
            i += 1
        return all_individuals

    def _get_fitness(self, brain):
        """
        Returns the fitness of an individual, checks first if GA is offline or online, then executes fitness test with
        test data or not. Calls get_fitness() if online GA
        :param brain: brain to test the fitness of
        :return: fitness of the brain
        """
        if self.offline is False:  # doing online fitness calculation
            if self.test_data is not None:  # if there is test data
                fitness = get_fitness([self.start_x, self.start_y], self.start_a, brain, self.iterations, self.light)
                for test_init in self.test_data:
                    fitness += get_fitness(test_init[0], test_init[1], brain, self.iterations, self.light)
                fitness /= len(self.test_data) + 1
                return fitness
            else:
                return get_fitness([self.start_x, self.start_y], self.start_a, brain, self.iterations, self.light)
        else:  # if offline, get fitness by using predictions
            if self.test_data is not None:
                fitness = 0
                sensor_log, wheel_log = self.net.predict_ahead(self.data, brain, self.look_ahead)
                sensor_left = sensor_log[0]
                sensor_right = sensor_log[1]
                fitness += np.mean(sensor_left) + np.mean(sensor_right)
                for test_init in self.test_data:
                    sensor_log, wheel_log = self.net.predict_ahead(test_init, brain, self.look_ahead)

                    sensor_left = sensor_log[0]
                    sensor_right = sensor_log[1]

                    # sim = Simulator(self.light)
                    # vehicle = ControllableVehicle([self.start_x, self.start_y], self.start_a, self.light)
                    # wheel_log1 = np.copy(wheel_log)
                    # vehicle.set_wheels(wheel_log)
                    # sim.run_simulation(len(wheel_log), graphics=False, vehicle=vehicle)

                    fitness += np.mean(sensor_left) + np.mean(sensor_right)
                fitness /= len(self.test_data) + 1  # +1 because of the real, initial condition not in the loop
            else:
                sensor_log, wheel_log = self.net.predict_ahead(self.data, brain, self.look_ahead)
                sensor_left = sensor_log[0]
                sensor_right = sensor_log[1]
                fitness = np.mean(sensor_left) + np.mean(sensor_right)
            return fitness

    def _tournament(self, individual1, individual2):
        """
        Executes a tournament for 2 individuals by performing crossover with winner and loser individuals.
        :param individual1: first individual
        :param individual2: second individual
        :return: the two individuals, an identifier of who is the loser, and the new fitness of the changed loser.
        """
        fitness1 = individual1[2]
        fitness2 = individual2[2]

        if fitness1 >= fitness2:
            ind1, ind2 = self._perform_crossover(individual1[1], individual2[1])
            new_fit = self._get_fitness(ind2)
            return [ind1, ind2, 2, new_fit]  # return 2 means the fitness is of individual 2 (loser)
        else:
            ind2, ind1 = self._perform_crossover(individual2[1], individual1[1])
            new_fit = self._get_fitness(ind1)
            return [ind1, ind2, 1, new_fit]  # return 1 means the fitness is of individual 1 (loser)

    def _run_winner(self, graphics, ind):
        """
        Executes winning brain at the end of GA. If offline just returns the sensor and wheel log that is predicted.
        If online, runs the vehicle in the simulation and returns the collected sensory and motor data.
        :param graphics: show the winner or not (only for online GA)
        :param ind: winning individual to show
        :return: sensor and wheel log
        """
        if self.offline:  # offline means in sleep cycle
            return self.net.predict_ahead(self.data, ind, self.look_ahead)

        else:  # this is for evolving real vehicles
            vehicle = BrainVehicle([self.start_x, self.start_y], self.start_a, self.light)
            vehicle.set_values(ind)
            vehicle.previous_movement = self.data
            # create Simulation
            vehicle = self.sim.run_simulation(self.iterations, graphics, vehicle)
            sensor_log = np.transpose([vehicle.sensor_left, vehicle.sensor_right])  # not tested
            wheel_log = []  # no need for these values, they are used in sleep for graphs
            return sensor_log, wheel_log

    def _start_ga(self):
        """
        GA main loop. Creates population, then iterates through the generations doing a tournament every iteration.
        Updates the data collected about fitness in GA, then runs the winner, shows graph and finishes.
        :return: brain of best individual, sensor log and wheel log
        """
        pool = self._init_pool(self.individuals)
        best_ind = [0, [0, 0, 0, 0, 0, 0], 0]  # initialize an individual
        for ind in pool:
            if ind[2] > best_ind[2]:
                best_ind = ind
        best_fit = [best_ind[2]]
        for generation in range(0, self.individuals * self.generations):
            if (generation % self.print_iterations) == 0:  # controls the print output to not spam console
                if self.verbose:
                    print '\riter: ' + str(generation) + '/' + str(self.individuals * self.generations),
                fits = [ind[2] for ind in pool]
                self.mean_fit.append(np.mean(fits))
                self.max_fit.append(max(fits))
                self.min_fit.append(min(fits))

            # find 2 random individuals
            rand_ind1 = random.randint(0, self.individuals - 1)
            rand_ind2 = random.randint(0, self.individuals - 1)
            if rand_ind1 == rand_ind2:
                rand_ind2 = random.randint(0, self.individuals - 1)
            # compare fitnesses
            ind1, ind2, loser, fit = self._tournament(pool[rand_ind1], pool[rand_ind2])

            # check who is winner and overwrite their stats
            if loser == 1:
                pool[rand_ind1][1] = ind1
                pool[rand_ind1][2] = fit
            else:
                pool[rand_ind2][1] = ind2
                pool[rand_ind2][2] = fit

            # update loser fitness and best fitness of the pool
            if pool[rand_ind1][2] > best_ind[2]:
                best_ind = pool[rand_ind1]
                best_fit.append(pool[rand_ind1][2])
            elif pool[rand_ind2][2] > best_ind[2]:
                best_ind = pool[rand_ind2]
                best_fit.append(pool[rand_ind2][2])
            else:
                best_fit.append(best_ind[2])

        if self.verbose:
            print '\rFinished GA: %s iter, best fit: %d, brain: %s' % (self.individuals * self.generations,
                                                                       best_fit[-1], best_ind[1])
        sensor_log, predicted_wheels = self._run_winner(self.graphics, best_ind[1])

        if self.graphics:
            self._show_fitness_graph(best_ind, best_fit)

        return [best_ind[1], sensor_log, predicted_wheels]

    def run_offline(self, narx, data, test_data, look_ahead, veh_pos=None, veh_angle=random.randint(0, 360),
                    light_pos=None, individuals=25, generations=10):
        """
        Method to run GA with predictions from network.
        :param narx: NARX network to predict with
        :param data: previous moment of the vehicle
        :param test_data: test points that the GA can test the vehicle's fitness on to generalise
        :param look_ahead: how far in the future to predict
        :param veh_pos: current vehicle position
        :param veh_angle: current vehicle angle
        :param light_pos: light position
        :param individuals: number of individuals for GA
        :param generations: number of generations for GA
        :return: best brain, sensor log, wheel log
        """
        if light_pos is None:
            light_pos = [1100, 600]
        if veh_pos is None:
            veh_pos = [300, 300]
        if veh_angle is None:
            veh_angle = random.randint(0, 360)
        self.net = narx
        self.data = data
        if test_data is not None and len(test_data) > 5:
            test_data = test_data[:5]
        self.test_data = test_data
        self.look_ahead = look_ahead
        self.start_x = veh_pos[0]
        self.start_y = veh_pos[1]
        self.start_a = veh_angle
        self.light = Light(light_pos)
        self.offline = True
        self.individuals = individuals
        self.generations = generations

        if self.verbose:
            print '\nStarting GA with model: individuals=%s generations=%s look_ahead=%s...' % (individuals,
                                                                                                generations, look_ahead)
        return self._start_ga()

    def run_with_simulation(self, veh_pos=None, veh_angle=random.randint(0, 360), previous_data=None, test_data=None,
                            individuals=40, generations=20, iterations=None):
        """
        Method to run GA with access to the real-world data. Evolves Braitenberg vehicles.
        :param veh_pos: current vehicle coordinates
        :param veh_angle: current vehicle angle
        :param previous_data: previous vehicle movement
        :param test_data: data points to test vehicle fitness on to generalise brain
        :param individuals: number of individuals
        :param generations: number of generations
        :param iterations: number of iterations to move vehicles for
        :return: best brain, sensor log, wheel log
        """
        if veh_pos is None:
            veh_pos = [random.randint(0, 1800), random.randint(0, 1000)]
        self.start_x = veh_pos[0]
        self.start_y = veh_pos[1]
        self.start_a = veh_angle
        self.individuals = individuals
        self.generations = generations
        if previous_data is None:
            previous_data = []
        self.data = previous_data
        self.test_data = test_data
        if iterations is None:
            start_light_dist = math.sqrt((self.light.pos[0] - self.start_x) ** 2 + (self.light.pos[1] - self.start_y) ** 2)
            # print 'vehicle available frames: ' + str(start_light_dist)
            self.iterations = int(start_light_dist / 2)  # Halved number to limit num of frames
        else:
            self.iterations = iterations
        self.offline = False

        if self.verbose:
            print 'Starting GA with world: individuals=%d generations=%d iterations=%d...' % (individuals, generations,
                                                                                              self.iterations)
        return self._start_ga()
