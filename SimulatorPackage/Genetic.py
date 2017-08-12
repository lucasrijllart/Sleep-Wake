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
    return [random.uniform(0, 2) for _ in range(0, GA.genome_length)]


def get_fitness(start_pos, start_a, brain, iterations, light):
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
    sensor_left = vehicle.sensor_left
    sensor_right = vehicle.sensor_right
    fitness = np.mean(sensor_left) + np.mean(sensor_right)
    return fitness


def _init_pool(individuals):
    """
    Creates a population of random individuals. Used when the GA starts.
    :param individuals: number of individuals in the population
    :return: the population as list of individuals
    """
    pool = []
    for i in range(0, individuals):
        ind = make_random_brain()
        pool.append(ind)
    return pool


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

    def __init__(self, light, graphics=False):
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
        self.iterations = None
        self.individuals = None
        self.generations = None

        self.net = None  # NARX network used to predict values for fitness
        self.data = None
        self.test_data = None
        self.look_ahead = None
        self.offline = None

        # To check the values (can be removed when GA works)
        self.mean_fit = []
        self.max_fit = []
        self.min_fit = []
        self.smax = []
        self.sav = []
        self.difff = []

    def show_fitness_graph(self, best_ind, best_fit):
        """
        Graph that shows the best fitness over time and the max, average, min over time.
        :param best_ind: the best individual from the population
        :param best_fit: the best fitness from the population
        :return:
        """
        plt.figure(1)
        if self.genome_length == 4:
            brain = '[%s,%s,%s,%s]' % (format(best_ind[1][0], '.2f'), format(best_ind[1][1], '.2f'),
                                       format(best_ind[1][2], '.2f'), format(best_ind[1][3], '.2f'))
        else:
            brain = '[%s,%s,%s,%s,%s,%s]' % (format(best_ind[1][0], '.2f'), format(best_ind[1][1], '.2f'),
                                             format(best_ind[1][2], '.2f'), format(best_ind[1][3], '.2f'),
                                             format(best_ind[1][4], '.2f'), format(best_ind[1][5], '.2f'))
        plt.suptitle('Finished GA of %d individuals and %d generations, with fitness %s of brain: %s' % (
            self.individuals, self.generations, format(best_ind[2], '.3f'), brain))
        plt.subplot(121)
        plt.title('Graph of best fitness by generation')
        plt.xlabel('iterations')
        plt.ylabel('max fitness')
        plt.plot(range(0, len(best_fit)), best_fit)

        plt.subplot(122)
        plt.title('Individual fitness over time')
        plt.xlabel('iterations')
        plt.ylabel('fitness of individuals')
        i = range(0, self.individuals * self.generations, self.print_iterations)
        plt.scatter(i, self.max_fit, s=4, label='max')
        plt.plot(i, self.max_fit)
        plt.scatter(i, self.mean_fit, s=6, label='mean')
        plt.plot(i, self.mean_fit)
        plt.scatter(i, self.min_fit, s=4, label='min')
        plt.plot(i, self.min_fit)
        plt.legend()

        plt.show()

    def _mutate(self, ind, mutation_rate):  # genome: [ll, lr, rr, rl, bl, br]
        for i in range(0, len(ind)):
            if random.random() < mutation_rate:
                ind[i] += (random.gauss(0, 1) * self.genome_scale*2) / 100
                if ind[i] > self.genome_scale:
                    ind[i] = -self.genome_scale + (self.genome_scale - ind[i])
                if ind[i] < -self.genome_scale:  # make it -self.genome_scale for wrapping around backwards movement
                    ind[i] = self.genome_scale - (-self.genome_scale - ind[i])
        return ind

    def _perform_crossover(self, indwin, indlos, crossover_rate, mutation_rate):
        for i in range(0, len(indwin)):
            if random.random() < crossover_rate:
                indlos[i] = indwin[i]
        # mutate
        indlos = self._mutate(indlos, mutation_rate)
        return [indwin, indlos]

    def _get_all_fitnesses(self, pool):
        i = 0
        all_individuals = []
        for individual in pool:
            fitness = self._get_fitness(individual)
            all_individuals.append([i, individual, fitness])
            i += 1
        return all_individuals

    def _get_fitness(self, brain):
        if self.offline is False:
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
                fitness /= len(self.test_data)
            else:
                sensor_log, wheel_log = self.net.predict_ahead(self.data, brain, self.look_ahead)
                sensor_left = sensor_log[0]
                sensor_right = sensor_log[1]
                fitness = np.mean(sensor_left) + np.mean(sensor_right)
            return fitness

    def _tournament(self, individual1, individual2, crossover_rate, mutation_rate):
        fitness1 = individual1[2]
        fitness2 = individual2[2]

        if fitness1 >= fitness2:
            ind1, ind2 = self._perform_crossover(individual1[1], individual2[1], crossover_rate, mutation_rate)
            new_fit = self._get_fitness(ind2)
            return [ind1, ind2, 2, new_fit]  # return 2 means the fitness is of individual 2 (loser)
        else:
            ind2, ind1 = self._perform_crossover(individual2[1], individual1[1], crossover_rate, mutation_rate)
            new_fit = self._get_fitness(ind1)
            return [ind1, ind2, 1, new_fit]  # return 1 means the fitness is of individual 1 (loser)

    def _run_winner(self, graphics, ind):
        if self.offline:  # offline means in sleep cycle
            return self.net.predict_ahead(self.data, ind, self.look_ahead)

        else:  # this is for evolving real vehicles
            vehicle = BrainVehicle([self.start_x, self.start_y], self.start_a, self.light)
            vehicle.set_values(ind)
            # create Simulation
            vehicle = self.sim.run_simulation(self.iterations, graphics, vehicle)
            sensor_log = np.transpose([vehicle.sensor_left, vehicle.sensor_right])  # not tested
            wheel_log = []  # no need for these values, they are used in sleep for graphs
            return sensor_log, wheel_log

    def _start_ga(self, crossover_rate, mutation_rate):
        pool = self._get_all_fitnesses(_init_pool(self.individuals))
        best_ind = [0, [0, 0, 0, 0, 0, 0], 0]  # initialize an individual
        for ind in pool:
            if ind[2] > best_ind[2]:
                best_ind = ind
        best_fit = [best_ind[2]]
        for generation in range(0, self.individuals * self.generations):
            if (generation % self.print_iterations) == 0:  # controls the print output to not spam console
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
            ind1, ind2, loser, fit = self._tournament(pool[rand_ind1], pool[rand_ind2], crossover_rate, mutation_rate)

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

        print '\rFinished GA: %s iter, best fit: %d, brain: %s' % (self.individuals * self.generations, best_fit[-1],
                                                                   best_ind[1])
        sensor_log, predicted_wheels = self._run_winner(self.graphics, best_ind[1])

        if self.graphics:
            self.show_fitness_graph(best_ind, best_fit)

        return [best_ind[1], sensor_log, predicted_wheels]

    def run_offline(self, narx, data, test_data, look_ahead, veh_pos=None, veh_angle=random.randint(0, 360),
                    light_pos=None, individuals=25, generations=10, crossover_rate=0.6, mutation_rate=0.3):
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

        print '\nStarting GA with model: individuals=%s generations=%s look_ahead=%s...' % (individuals, generations, look_ahead)
        return self._start_ga(crossover_rate, mutation_rate)

    def run(self, veh_pos=None, veh_angle=random.randint(0, 360), individuals=30, generations=20,
            iterations=None, crossover_rate=0.6, mutation_rate=0.3):
        if veh_pos is None:
            veh_pos = [random.randint(0, 1800), random.randint(0, 1000)]
        self.start_x = veh_pos[0]
        self.start_y = veh_pos[1]
        self.start_a = veh_angle
        self.individuals = individuals
        self.generations = generations
        if iterations is None:
            start_light_dist = math.sqrt((self.light.pos[0] - self.start_x) ** 2 + (self.light.pos[1] - self.start_y) ** 2)
            print 'vehicle available frames: ' + str(start_light_dist)
            self.iterations = int(start_light_dist / 2)  # Halved number to limit num of frames
        else:
            self.iterations = iterations
        self.offline = False

        print 'Starting GA with world: individuals=%d generations=%d iterations=%d...' % (individuals, generations, iterations)
        return self._start_ga(crossover_rate, mutation_rate)
