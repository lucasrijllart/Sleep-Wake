from Simulator import Simulator
from Vehicles import Attacker
from Light import Light
import random
import math
import matplotlib.pyplot as plt


def init_pool():
    pool = []
    for i in range(0, individuals):
        ind = [random.randint(0, 20) for x in range(0, 6)]
        pool.append(ind)
    return pool


def mutate(ind):
    for i in range(0, len(ind)):
        if random.random() < mutation_rate:
            ind[i] = random.randint(1, 20)
    return ind


def perform_crossover(indwin, indlos, indwinfit):
    for i in range(0, len(indwin)):
        if random.random() < crossover_rate:
            indlos[i] = indwin[i]
    # mutate until fitter or counter
    counter = 0
    while indwinfit > get_fitness(indlos) and counter > 2000:
        indlos = mutate(indlos)
        counter += 1
        print(counter)
    return [indwin, indlos]


def get_all_fitnesses(pool):
    i = 0
    fitnesses = []
    for individual in pool:
        fitness = get_fitness(individual)
        fitnesses.append([i, individual, fitness])
        i += 1
    return fitnesses


def get_fitness(ind):
    # create sprites
    vehicle = Attacker([start_x, start_y], start_a)
    vehicle.set_values(ind[0], ind[1], ind[2], ind[3], ind[4], ind[5])

    # create Simulation
    vehicle = sim.run_simulation(iterations, False, vehicle, light)

    if vehicle.reached_light:
        individual_reached_light.append(True)
        winner.append(ind)

    # calculate fitness with 1/distance
    distance = 0
    for step in vehicle.pos:
        distance += math.sqrt((step[0] - light_x) ** 2 + (step[1] - light_y) ** 2)
    fitness = 1 / distance

    '''
    # calculate fitness with average distance
    distances = []
    for step in vehicle.pos:
        distances.append(math.sqrt((step[0] - light_x) ** 2 + (step[1] - light_y) ** 2))
    fitness = 0
    for distance in distances:
        fitness += distance
    fitness = fitness / len(distances)
    fitness = 1 / fitness
    '''

    return fitness*100000


def tournament(individual1, individual2):
    fitness1 = get_fitness(individual1)
    fitness2 = get_fitness(individual2)

    if fitness1 >= fitness2:
        return perform_crossover(individual1, individual2, fitness1)
    else:
        ind2, ind1 = perform_crossover(individual2, individual1, fitness2)
        return [ind1, ind2]


def run_winner(ind):
    vehicle = Attacker([start_x, start_y], start_a)
    vehicle.set_values(ind[0], ind[1], ind[2], ind[3], ind[4], ind[5])

    # create Simulation
    sim.run_simulation(iterations, True, vehicle, light)


def run_ga():
    pool = init_pool()
    all_fitness = get_all_fitnesses(pool)
    print(all_fitness)
    best_ind = [0, 0, 0]
    for ind in all_fitness:
        if ind[2] > best_ind[2]:
            best_ind = ind
    print 'best: ' + str(best_ind)
    best_fit = [best_ind[2]]
    for generation in range(0, individuals * generations):
        print '\rgen: ' + str(generation) + '/' + str(individuals*generations),

        # find 2 random individuals
        rand_ind1 = random.randint(0, individuals-1)
        rand_ind2 = random.randint(0, individuals-1)
        if rand_ind1 == rand_ind2:
            rand_ind2 = random.randint(0, individuals-1)
        # compare fitnesses
        ind1, ind2 = tournament(pool[rand_ind1], pool[rand_ind2])
        if individual_reached_light[-1]:
            ind = [x for x in all_fitness if winner[-1] in x][0]
            best_ind = ind
            best_fit.append(ind[2])
            break
        else:
            # winner overwrites loser with crossover
            pool[rand_ind1] = ind1
            pool[rand_ind2] = ind2
            all_fitness[rand_ind1][2] = get_fitness(ind1)
            all_fitness[rand_ind2][2] = get_fitness(ind2)
            if all_fitness[rand_ind1][2] > best_ind[2]:
                best_ind = all_fitness[rand_ind1]
                best_fit.append(all_fitness[rand_ind1][2])
            elif all_fitness[rand_ind2][2] > best_ind[2]:
                best_ind = all_fitness[rand_ind2]
                best_fit.append(all_fitness[rand_ind2][2])
            else:
                best_fit.append(best_ind[2])

    print '\nBest fitness: ' + str(best_fit[-1]) + str(best_ind)
    run_winner(best_ind[1])

    plt.plot(range(0, len(best_fit)), best_fit)
    plt.show()


individuals = 20
generations = 4
crossover_rate = 0.7
mutation_rate = 0.3
iterations = 300
individual_reached_light = [False]
winner = []

sim = Simulator()

# create random values
start_x = random.randint(25, sim.window_width - 25)
start_y = random.randint(25, sim.window_height - 25)
start_a = random.randint(0, 360)
light_x = random.randint(25, sim.window_width - 25)
light_y = random.randint(25, sim.window_height - 25)
light = Light([light_x, light_y])

run_ga()
