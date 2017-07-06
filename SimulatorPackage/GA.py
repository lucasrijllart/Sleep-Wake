from Simulator import Simulator
from Vehicles import Attacker
from Light import Light
import random
import math
import matplotlib.pyplot as plt


class GA:

    def __init__(self, graphics=False):
        self.graphics = graphics

        self.individual_reached_light = [False]
        self.sim = Simulator()
        self.genome_scale = 8

        # init values as None, as they will be rewritten in run or run_random
        self.start_x, self.start_y, self.start_a = None, None, None
        self.light = None
        self.iterations = None

    def _init_pool(self, individuals):
        pool = []
        for i in range(0, individuals):
            ind = [random.uniform(-self.genome_scale, self.genome_scale) for x in range(0, 6)]
            pool.append(ind)
        return pool

    def _mutate(self, ind, mutation_rate):  # genome: [ll, lr, rr, rl, bl, br]
        for i in range(0, len(ind)):
            if random.random() < mutation_rate:
                ind[i] += (random.gauss(0, 1) * self.genome_scale*2) / 100
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
        fitnesses = []
        for individual in pool:
            fitness = self._get_fitness(individual)
            fitnesses.append([i, individual, fitness])
            i += 1
        return fitnesses

    def _get_fitness(self, ind):
        # create sprites
        vehicle = Attacker([self.start_x, self.start_y], self.start_a)
        vehicle.set_values(ind[0], ind[1], ind[2], ind[3], ind[4], ind[5])

        # create Simulation
        vehicle = self.sim.run_simulation(self.iterations, False, vehicle, self.light)

        if vehicle.reached_light:
            self.individual_reached_light.append(True)

        '''
        # calculate fitness with 1/distance
        distance = 0
        for step in vehicle.pos:
            distance += math.sqrt((step[0] - self.light.pos[0]) ** 2 + (step[1] - self.light.pos[1]) ** 2)
        fitness = 1 / distance
        '''

        # calculate fitness with average distance
        distances = []
        for step in vehicle.pos:
            distances.append(math.sqrt((step[0] - self.light.pos[0]) ** 2 + (step[1] - self.light.pos[1]) ** 2))
        fitness = 0
        for distance in distances:
            fitness += distance
        fitness /= len(distances)
        fitness = 1000 / fitness

        return fitness

    def _tournament(self, individual1, individual2, crossover_rate, mutation_rate):
        fitness1 = self._get_fitness(individual1)
        fitness2 = self._get_fitness(individual2)

        if fitness1 >= fitness2:
            return self._perform_crossover(individual1, individual2, crossover_rate, mutation_rate)
        else:
            ind2, ind1 = self._perform_crossover(individual2, individual1, crossover_rate, mutation_rate)
            return [ind1, ind2]

    def _run_winner(self, graphics, ind):
        print 'Running: ' + str(ind) + str(self._get_fitness(ind[1]))
        ind = ind[1]
        vehicle = Attacker([self.start_x, self.start_y], self.start_a)
        vehicle.set_values(ind[0], ind[1], ind[2], ind[3], ind[4], ind[5])
        # create Simulation
        self.sim.run_simulation(self.iterations, graphics, vehicle, self.light)

    def _start_ga(self, individuals, generations, crossover_rate, mutation_rate):
        pool = self._init_pool(individuals)
        all_fitness = self._get_all_fitnesses(pool)
        print(all_fitness)
        best_ind = [0, 0, 0]
        for ind in all_fitness:
            if ind[2] > best_ind[2]:
                best_ind = ind
        print 'best: ' + str(best_ind)
        best_fit = [best_ind[2]]
        for generation in range(0, individuals * generations):
            print '\rgen: ' + str(generation) + '/' + str(individuals * generations),

            # find 2 random individuals
            rand_ind1 = random.randint(0, individuals - 1)
            rand_ind2 = random.randint(0, individuals - 1)
            if rand_ind1 == rand_ind2:
                rand_ind2 = random.randint(0, individuals - 1)
            # compare fitnesses
            ind1, ind2 = self._tournament(pool[rand_ind1], pool[rand_ind2], crossover_rate, mutation_rate)

            # winner overwrites loser with crossover
            pool[rand_ind1] = ind1
            pool[rand_ind2] = ind2
            all_fitness[rand_ind1][2] = self._get_fitness(ind1)
            all_fitness[rand_ind2][2] = self._get_fitness(ind2)
            if all_fitness[rand_ind1][2] > best_ind[2]:
                best_ind = all_fitness[rand_ind1]
                best_fit.append(all_fitness[rand_ind1][2])
            elif all_fitness[rand_ind2][2] > best_ind[2]:
                best_ind = all_fitness[rand_ind2]
                best_fit.append(all_fitness[rand_ind2][2])
            else:
                best_fit.append(best_ind[2])

        print '\nBest fitness: ' + str(best_fit[-1]) + str(best_ind)
        self._run_winner(self.graphics, best_ind)
        plt.plot(range(0, len(best_fit)), best_fit)
        plt.show()

    def run_random(self, individuals=25, generations=8, crossover_rate=0.7, mutation_rate=0.3):
        # create random values
        self.start_x = random.randint(25, self.sim.window_width - 25)
        self.start_y = random.randint(25, self.sim.window_height - 25)
        self.start_a = random.randint(0, 360)
        light_x = random.randint(25, self.sim.window_width - 25)
        light_y = random.randint(25, self.sim.window_height - 25)
        self.light = Light([light_x, light_y])

        start_light_dist = math.sqrt((light_x - self.start_x) ** 2 + (light_y - self.start_y) ** 2)
        print 'vehicle available frames: ' + str(start_light_dist)
        self.iterations = int(start_light_dist / 2)  # Halved number to limit num of frames and tested with a big dist

        self._start_ga(individuals, generations, crossover_rate, mutation_rate)

    def run(self, veh_pos, veh_angle, light_pos, individuals=25, generations=8, crossover_rate=0.7, mutation_rate=0.3):
        self.start_x = veh_pos[0]
        self.start_y = veh_pos[1]
        self.start_a = veh_angle
        self.light = Light(light_pos)

        start_light_dist = math.sqrt((light_pos[0] - self.start_x) ** 2 + (light_pos[1] - self.start_y) ** 2)
        print 'vehicle available frames: ' + str(start_light_dist)
        self.iterations = int(start_light_dist / 2)  # Halved number to limit num of frames and tested with a big dist

        self._start_ga(individuals, generations, crossover_rate, mutation_rate)
