from Simulator import Simulator
from Sprites import BrainVehicle, ControllableVehicle, Light
import numpy as np
import random
import math
import matplotlib.pyplot as plt


def make_random_brain():
    return [random.uniform(-GA.genome_scale, GA.genome_scale) for _ in range(0, GA.genome_length)]


def _init_pool(individuals):
    pool = []
    for i in range(0, individuals):
        ind = make_random_brain()
        pool.append(ind)
    return pool


class GA:

    genome_scale = 10
    genome_length = 6

    def __init__(self, graphics=True):
        self.graphics = graphics

        self.individual_reached_light = [False]
        self.sim = None

        # init values as None, as they will be rewritten in run or run_random
        self.start_x, self.start_y, self.start_a = None, None, None
        self.light = None
        self.iterations = None

        self.net = None  # NARX network used to predict values for fitness
        self.data = None
        self.look_ahead = None
        self.offline = None

        self.sim = Simulator()

    def _mutate(self, ind, mutation_rate):  # genome: [ll, lr, rr, rl, bl, br]
        for i in range(0, len(ind)):
            if random.random() < mutation_rate:
                ind[i] += (random.gauss(0, 1) * self.genome_scale*2) / 100
                if ind[i] > self.genome_scale:
                    ind[i] = -self.genome_scale + (self.genome_scale - ind[i])
                if ind[i] < -self.genome_scale:
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

    def _get_fitness(self, ind):
        if self.offline is False:
            # create sprites
            vehicle = BrainVehicle([self.start_x, self.start_y], self.start_a)
            vehicle.set_values(ind)
            # create Simulation
            vehicle = self.sim.run_simulation(self.iterations, False, vehicle, self.light)

            if vehicle.reached_light:
                self.individual_reached_light.append(True)

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

        else:  # if offline, get fitness by using predictions
            sensor_log = np.array([[], []])
            wheel_log = []
            data = self.data

            next_input = np.array(data[:, -1])
            next_input = np.array([[x] for x in next_input])

            # Execute the first prediction without adding it to the data, as the first prediction comes from actual data
            # 1. predict next sensory output
            prediction = self.net.predict(next_input, pre_inputs=data[:, :-1], pre_outputs=data[2:, :-1])
            # 2. log predicted sensory information to list (used for fitness)
            sensor_log = np.concatenate((sensor_log, prediction), axis=1)

            # 3. feed it to the brain to get motor information
            wheel_l, wheel_r = [
                (prediction[0] * ind[0]) + (prediction[1] * ind[3]) + ind[4] / BrainVehicle.bias_constant,
                (prediction[1] * ind[2]) + (prediction[0] * ind[1]) + ind[5] / BrainVehicle.bias_constant]
            wheel_log.append([wheel_l[0], wheel_r[0]])

            # 4. add this set of data to the input of the prediction
            next_input = np.array([wheel_l, wheel_r, prediction[0], prediction[1]])

            for it in range(1, self.look_ahead):  # loop through the time steps
                # 1. predict next sensory output
                prediction = self.net.predict(next_input, pre_inputs=data, pre_outputs=data[2:])

                # 2. log predicted sensory information to list (used for fitness)
                sensor_log = np.concatenate((sensor_log, prediction), axis=1)

                # 3. feed it to the brain to get motor information
                wheel_l, wheel_r = [(prediction[0] * ind[0]) + (prediction[1] * ind[3]) + ind[4] / BrainVehicle.bias_constant,
                                    (prediction[1] * ind[2]) + (prediction[0] * ind[1]) + ind[5] / BrainVehicle.bias_constant]
                wheel_log.append([wheel_l[0], wheel_r[0]])

                # 4. concatenate previous step to the full data
                data = np.concatenate((data, next_input), axis=1)

                # 5. set the predicted data to the next input of the prediction
                next_input = np.array([wheel_l, wheel_r, prediction[0], prediction[1]])
                # loop back to 1 until reached timestep

            # calculate fitness by taking average of sensory predictions
            # fitness = (sum(sensor_log[0]) + sum(sensor_log[1]))# / len(sensor_log[0])

            sim = Simulator()
            vehicle = ControllableVehicle([self.start_x, self.start_y], self.start_a)
            wheel_log1 = np.copy(wheel_log)
            vehicle.set_wheels(wheel_log)
            sim.light = Light([1100, 600])
            sim.run_simulation(len(wheel_log), graphics=False, vehicle=vehicle)

            # wheel_log = np.transpose(np.array(wheel_log))
            # fitness_after = 0
            # for i in range(0, len(sensor_log[0])):
            #     v = abs(wheel_log[0][i] + wheel_log[1][i])
            #     diff = math.sqrt(abs(wheel_log[0][i] - wheel_log[1][i]))
            #     max_ = max(sensor_log[0][i], sensor_log[1][i])
            #     avg = (sensor_log[0][i] + sensor_log[1][i]) / 2
            #     fitness = ((v * (1 - diff)) / max_)
            #     fitness_after += fitness
            # fitness_after /= len(sensor_log[0])
            #
            # return fitness_after

            Sl = sensor_log[0]
            Sr = sensor_log[1]
            wheel_log = np.transpose(wheel_log1)

            # sum = 0
            # for idx in range(1, len(Sl)):
            #     sum += (Sl[idx] + Sr[idx] )/2 - (mt.fabs(Sl[idx] - Sl[idx-1]) + mt.fabs(Sr[idx] - Sl[idx-1]))
            #

            # fit = 0
            # for idx in range(1, len(Sl)):
            #     if Sl[idx] > Sl[idx - 1] and Sr[idx] > Sr[idx - 1]:
            #         fit += 1
            #     elif Sl[idx] < Sl[idx - 1] or Sr[idx] < Sr[idx - 1]:
            #         fit += -1.1
            #     else:
            #         fit += -1.2

            fitness = 0
            for i in range(0, len(Sl)):
                vA = (wheel_log[0][i] + wheel_log[1][i])/2
                diff = math.sqrt(abs(wheel_log[0][i] - wheel_log[1][i]))
                sA = (sensor_log[0][i] + sensor_log[1][i])
                smax = max(sensor_log[0][i] ,sensor_log[1][i])
                fitness += vA/2 * (5 - diff) * (10 + smax) + sA
            fitness /= len(Sl)

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
        if self.offline:
            sensor_log = np.array([[], []])
            wheel_log = []
            data = self.data

            next_input = np.array(data[:, -1])
            next_input = np.array([[x] for x in next_input])

            # Execute the first prediction without adding it to the data, as the first prediction comes from actual data
            # 1. predict next sensory output
            prediction = self.net.predict(next_input, pre_inputs=data[:, :-1], pre_outputs=data[2:, :-1])
            # 2. log predicted sensory information to list (used for fitness)
            sensor_log = np.concatenate((sensor_log, prediction), axis=1)

            # 3. feed it to the brain to get motor information
            wheel_l, wheel_r = [
                (prediction[0] * ind[0]) + (prediction[1] * ind[3]) + ind[4] / BrainVehicle.bias_constant,
                (prediction[1] * ind[2]) + (prediction[0] * ind[1]) + ind[5] / BrainVehicle.bias_constant]
            wheel_log.append([wheel_l[0], wheel_r[0]])

            # 4. add this set of data to the input of the prediction
            next_input = np.array([wheel_l, wheel_r, prediction[0], prediction[1]])

            for it in range(1, self.look_ahead):  # loop through the time steps
                # 1. predict next sensory output
                prediction = self.net.predict(next_input, pre_inputs=data, pre_outputs=data[2:])

                # 2. log predicted sensory information to list (used for fitness)
                sensor_log = np.concatenate((sensor_log, prediction), axis=1)

                # 3. feed it to the brain to get motor information
                wheel_l, wheel_r = [
                    (prediction[0] * ind[0]) + (prediction[1] * ind[3]) + ind[4] / BrainVehicle.bias_constant,
                    (prediction[1] * ind[2]) + (prediction[0] * ind[1]) + ind[5] / BrainVehicle.bias_constant]
                wheel_log.append([wheel_l[0], wheel_r[0]])

                # 4. concatenate previous step to the full data
                data = np.concatenate((data, next_input), axis=1)

                # 5. set the predicted data to the next input of the prediction
                next_input = np.array([wheel_l, wheel_r, prediction[0], prediction[1]])
                # loop back to 1 until reached timestep
            return [sensor_log, wheel_log]

        else:
            print 'Running: ' + str(ind) + str(self._get_fitness(ind))
            vehicle = BrainVehicle([self.start_x, self.start_y], self.start_a)
            vehicle.set_values(ind)
            # create Simulation
            vehicle = self.sim.run_simulation(self.iterations, graphics, vehicle, self.light)
            sensor_log = np.transpose([vehicle.sensor_left, vehicle.sensor_right])  # not tested
            return sensor_log

    def _start_ga(self, individuals, generations, crossover_rate, mutation_rate):
        pool = self._get_all_fitnesses(_init_pool(individuals))
        best_ind = [0, [0, 0, 0, 0, 0, 0], 0] # initialize an individual
        for ind in pool:
            if ind[2] > best_ind[2]:
                best_ind = ind
        best_fit = [best_ind[2]]
        for generation in range(0, individuals * generations):
            print '\riter: ' + str(generation) + '/' + str(individuals * generations),

            # find 2 random individuals
            rand_ind1 = random.randint(0, individuals - 1)
            rand_ind2 = random.randint(0, individuals - 1)
            if rand_ind1 == rand_ind2:
                rand_ind2 = random.randint(0, individuals - 1)
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

        print '\rFinished GA: %s iter, best fit: %d, brain: %s' % (individuals * generations, best_fit[-1], best_ind[1])
        sensor_log, predicted_wheels = self._run_winner(self.graphics, best_ind[1])

        plt.plot(range(0, len(best_fit)), best_fit)
        plt.show()

        return [best_ind[1], sensor_log, predicted_wheels]

    def run_offline(self, narx, data, look_ahead=100, veh_pos=None, veh_angle=random.randint(0, 360), light_pos=None,
                    individuals=25, generations=10, crossover_rate=0.6, mutation_rate=0.2):
        if light_pos is None:
            light_pos = [1100, 600]
        if veh_pos is None:
            veh_pos = [300, 300]
        if veh_angle is None:
            veh_angle = random.randint(0, 360)
        self.net = narx
        self.data = data
        self.look_ahead = look_ahead
        self.start_x = veh_pos[0]
        self.start_y = veh_pos[1]
        self.start_a = veh_angle
        self.light = Light(light_pos)
        self.offline = True

        # Uncomment this to run GA with simulation
        # print 'Starting GA with simulation'
        # self.run(veh_pos, veh_angle, light_pos)
        # self.offline = True

        print 'Starting GA: individuals=%s generations=%s look_ahead=%s' % (individuals, generations, look_ahead)
        return self._start_ga(individuals, generations, crossover_rate, mutation_rate)

    def run(self, veh_pos, veh_angle, light_pos, individuals=25, generations=8, crossover_rate=0.7, mutation_rate=0.3):
        self.start_x = veh_pos[0]
        self.start_y = veh_pos[1]
        self.start_a = veh_angle
        self.light = Light(light_pos)
        self.offline = False

        start_light_dist = math.sqrt((light_pos[0] - self.start_x) ** 2 + (light_pos[1] - self.start_y) ** 2)
        print 'vehicle available frames: ' + str(start_light_dist)
        self.iterations = int(start_light_dist / 4)  # Halved number to limit num of frames and tested with a big dist

        self._start_ga(individuals, generations, crossover_rate, mutation_rate)
