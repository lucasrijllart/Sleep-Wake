import numpy as np
import random, time, datetime
from Simulator import Simulator
from Narx import PyrennNarx
import Narx as narx
import matplotlib.pyplot as plt
import Genetic
from Genetic import GA as GA
from Sprites import BrainVehicle, ControllableVehicle, Light
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
    new_targets = []
    for vehicle in raw_data:
        vehicle_timesteps = []
        for timestep in vehicle:
            vehicle_timesteps.append(np.transpose(np.array(timestep)))
        new_inputs.append(np.transpose(np.array(vehicle_timesteps)))
        new_targets.append(np.transpose(np.array(vehicle_timesteps))[-2:, :])
    new_inputs = np.array(new_inputs)
    new_targets = np.array(new_targets)
    return [new_inputs, new_targets]


def collect_random_data(vehicle_pos=None, vehicle_angle_rand=True, light_pos=None, runs=10, iterations=1000,
                        graphics=False, gamma=0.3, seed=None):
    """ Runs many vehicles in simulations and collects their sensory and motor information """
    random_brains = 0  # maybe pass this into the method at some point
    if vehicle_pos is None:
        vehicle_pos = [300, 300]  # [random.randint(25, 1215), random.randint(25, 695)]
    if vehicle_angle_rand is False:
        vehicle_angle = 200
    else:
        vehicle_angle = random.randint(0, 360)
    if light_pos is None:
        light_pos = [1100, 600]

    data = []
    sim = Simulator()
    for run in range(0, runs):
        if vehicle_angle_rand:  # update angle for each vehicle
            vehicle_angle = random.randint(0, 360)
        brain = None
        if random.random() < 0.0:  # 30% chance of vehicle being a random brain
            brain = Genetic.make_random_brain()
            random_brains += 1
        # add 1 because the simulation returns iterations-1 as the first timestep is the starting pos (not recorded)
        v = sim.quick_simulation(iterations, graphics, vehicle_pos, vehicle_angle, light_pos, gamma, seed, brain=brain)
        vehicle_data_in_t = []
        for t in range(0, iterations):
            vehicle_data_in_t.append([v.motor_left[t], v.motor_right[t], v.sensor_left[t], v.sensor_right[t]])
        data.append(vehicle_data_in_t)
    print '\nCollected data from %d vehicles over %d iterations with %d randoms' % (runs, iterations, random_brains)
    return pre_process_by_vehicle(data)


def random_brain_benchmark(random_brains=1000, iterations=100, start_pos=None, start_a=random.randint(0, 360),
                           light=None, graphics=False, actual=None):
    """ Shows a graph of the fitness of a number of random brains """
    print '\nStarting benchmark test for %d random brains' % random_brains
    if start_pos is None:
        start_pos = [random.randint(0, Simulator.window_width), random.randint(0, Simulator.window_height)]
    if light is None:
        light = Light([1100, 600])
    fitnesses = []

    start_time = time.time()
    for individual in range(0, random_brains):
        brain = Genetic.make_random_brain()
        fitnesses.append(Genetic.get_fitness(start_pos, start_a, brain, iterations, light))
    print 'Collected %d random brains in %ds' % (random_brains, time.time()-start_time)
    random_mean_fit = np.mean(fitnesses)

    ga = GA(graphics=graphics)
    brain = ga.run(start_pos, start_a, light, iterations=iterations)
    brain = brain[0]
    evol_score = Genetic.get_fitness(start_pos, start_a, brain, iterations, light)
    fitnesses.append(evol_score)

    if actual is not None:
        brain = actual.get_brain()
        pred_score = Genetic.get_fitness(start_pos, start_a, brain, iterations, light)
        fitnesses.append(pred_score)

    plt.title('Benchmark test for random vehicle fitness')

    fitnesses.sort()
    evol_idx = np.where(fitnesses == evol_score)
    pred_idx = np.where(fitnesses == pred_score)
    plt.scatter(range(0, len(fitnesses)), fitnesses, s=1, c='grey', label='random')
    plt.scatter(evol_idx, evol_score, s=15, c='green', label='evolved')
    plt.scatter(pred_idx, pred_score, s=20, c='red', label='predicted')
    plt.plot([0, len(fitnesses)], [random_mean_fit, random_mean_fit], c='blue', label='random mean')
    plt.xlabel('individuals')
    plt.ylabel('fitness')
    plt.legend()
    plt.show()


class Cycles:

    def __init__(self, net_filename=None):

        self.net = None  # NARX network
        self.net_filename = net_filename
        if net_filename is not None:
            start_time = time.time()
            saved_net = narx.load_net(net_filename)
            self.net = PyrennNarx()
            self.net.set_net(saved_net)
            print 'Loaded NARX from file "%s" in %ds' % (net_filename, time.time()-start_time)

        self.random_vehicle = None
        self.brain = [0, 0, 0, 0, 0, 0]  # Vehicle brain assigned after GA, 6 weights
        self.vehicle_first_move = None
        self.predicted_pos = None

        self.sim = None

        self.count_cycles = 0

    def show_error_graph(self, testing_time=400, predict_after=100, brain=None, gamma=0.2, use_narx=True):
        """ Presents a graph with real and predicted sensor and motor values """
        if self.net is None:
            print 'show_error_graph() Exception: No network found'
            return
        # Lookahead is testing time - predict after
        look_ahead = testing_time - predict_after

        # Create random brain and give it to vehicle
        if brain is not None:
            brain = [float(item) for item in brain]

        # Create simulation, run vehicle in it, and collect its sensory and motor information
        sim = Simulator()
        vehicle = sim.init_simulation(testing_time, True, veh_angle=200, brain=brain, gamma=gamma)
        sensor_motor = []
        for x in range(0, testing_time):
            sensor_motor.append(
                [vehicle.motor_left[x], vehicle.motor_right[x], vehicle.sensor_left[x], vehicle.sensor_right[x]])
        sensor_motor = np.transpose(np.array(sensor_motor))
        # print sensor_motor.shape

        data = np.array(sensor_motor[:, predict_after:])  # data up until the initial run time
        sensor_log = np.array([[], []])
        wheel_log = []

        # check if vehicle has a brain, if not just pass it the data
        if brain is not None:
            next_input = np.array(data[:, -1])
            next_input = np.array([[x] for x in next_input])
            # Execute the first prediction without adding it to the data, as the first prediction comes from actual data
            # 1. predict next sensory output
            prediction = self.net.predict(next_input, pre_inputs=data[:, :-1], pre_outputs=data[2:, :-1])
            # 2. log predicted sensory information to list (used for fitness)
            sensor_log = np.concatenate((sensor_log, prediction), axis=1)
            # 3. feed it to the brain to get motor information
            wheel_l, wheel_r = [
                (prediction[0] * brain[0]) + (prediction[1] * brain[3]) + brain[4] / BrainVehicle.bias_constant,
                (prediction[1] * brain[2]) + (prediction[0] * brain[1]) + brain[5] / BrainVehicle.bias_constant]
            wheel_log.append([wheel_l[0], wheel_r[0]])
            # 4. add this set of data to the input of the prediction
            next_input = np.array([wheel_l, wheel_r, prediction[0], prediction[1]])

            for it in range(1, look_ahead):  # loop through the time steps
                # 1. predict next sensory output
                prediction = self.net.predict(next_input, pre_inputs=data, pre_outputs=data[2:])
                # 2. log predicted sensory information to list (used for fitness)
                sensor_log = np.concatenate((sensor_log, prediction), axis=1)
                # 3. feed it to the brain to get motor information
                wheel_l, wheel_r = [(prediction[0] * brain[0]) + (prediction[1] * brain[3]) + brain[4] / BrainVehicle.bias_constant,
                                    (prediction[1] * brain[2]) + (prediction[0] * brain[1]) + brain[5] / BrainVehicle.bias_constant]
                wheel_log.append([wheel_l[0], wheel_r[0]])
                # 4. concatenate previous step to the full data
                data = np.concatenate((data, next_input), axis=1)
                # 5. set the predicted data to the next input of the prediction
                next_input = np.array([wheel_l, wheel_r, prediction[0], prediction[1]])
                # loop back to 1 until reached timestep

        else:  # vehicle does not have a brain, just random movement
            sensor_log, wheel_log = self.net.predict_error_graph(data, look_ahead, predict_after, use_narx)

            brain = 'random'

            # add previous and predicted values for sensors to display in graph
            sensor_left = np.concatenate((sensor_motor[2][:predict_after], sensor_log[0]))
            sensor_right = np.concatenate((sensor_motor[3][:predict_after], sensor_log[1]))

            # get sensory information of vehicle and compare with predicted
            plt.figure(1)
            title = 'Graph showing predicted vs actual values for sensors and Mean Squared Error.\nNetwork:%s, with' \
                    ' %d timesteps and predictions starting at t=%d, and with %s brain' % (self.net_filename, testing_time,
                                                                                           predict_after, brain)
            plt.suptitle(title)
            i = np.array(range(0, len(sensor_left)))
            i2 = np.array(range(predict_after, testing_time))
            plt.subplot(221)
            plt.title('Left sensor values')
            plt.plot(i, vehicle.sensor_left, 'b', i2, sensor_log[0], 'r')

            plt.subplot(222)
            plt.title('Right sensor values')
            plt.plot(i, vehicle.sensor_right, 'b', label='real')
            plt.plot(i2, sensor_log[1], 'r', label='predicted')
            plt.legend()

            msel = [((sensor_left[i] - vehicle.sensor_left[i]) ** 2) / len(sensor_left) for i in range(0, len(sensor_left))]
            plt.subplot(223)
            plt.title('MSE of left sensor')
            plt.plot(range(0, len(msel)), msel)

            mser = [((sensor_right[i]-vehicle.sensor_right[i])**2) / len(sensor_right) for i in range(0, len(sensor_right))]
            plt.subplot(224)
            plt.title('MSE of left sensor')
            plt.plot(range(0, len(mser)), mser)
            plt.show()

    def train_network(self, learning_runs, learning_time, layers, delay, max_epochs, gamma=0.3, use_mean=True):
        filename = 'narx/r%dt%dd%de%d' % (learning_runs, learning_time, delay, max_epochs)
        # check if filename is already taken
        count = 1
        new_filename = filename
        while os.path.exists(new_filename):
            new_filename = filename + '_v' + str(count)
            count += 1
        filename = new_filename

        # collect data for NARX and testing and pre-process data
        train_input, train_target = collect_random_data(runs=learning_runs, iterations=learning_time,
                                                        graphics=False, gamma=gamma)

        # creation of network
        print '\nNetwork training started at ' + str(time.strftime('%H:%M:%S %d/%m', time.localtime()) + ' with params:')
        print '\t learning runs=%d, learning time=%d, delays=%d, epochs=%d' % (learning_runs, learning_time,
                                                                                  delay, max_epochs)
        start_time = time.time()
        self.net = PyrennNarx(layers=layers, delay=delay)

        # train network
        self.net.train(train_input, verbose=True, max_iter=max_epochs, use_mean=use_mean)

        # save network to file
        self.net.save_to_file(filename=filename)
        print 'Finished training network "%s" in %s' % (filename, datetime.timedelta(seconds=time.time()-start_time))

    def wake_learning(self, random_movements):
        """ Create a vehicle and run for some time-steps """
        # Create vehicle in simulation
        self.sim = Simulator()
        self.random_vehicle = self.sim.init_simulation(random_movements, graphics=True, cycle='wake (training)',
                                                       veh_pos=[300, 300], veh_angle=200)
        vehicle_move = []
        for t in range(0, random_movements):
            vehicle_move.append([self.random_vehicle.motor_left[t], self.random_vehicle.motor_right[t],
                                 self.random_vehicle.sensor_left[t], self.random_vehicle.sensor_right[t]])
        vehicle_first_move = []
        for t in range(0, len(vehicle_move)):
            vehicle_first_move.append(np.transpose(np.array(vehicle_move[t])))
        self.vehicle_first_move = np.transpose(np.array(vehicle_first_move))

    def sleep(self, look_ahead=100, individuals=25, generations=10, use_narx=True):
        # run GA and find best brain to give to testing
        ga = GA(graphics=True)
        ga_result = ga.run_offline(self.net, self.vehicle_first_move, look_ahead, use_narx, veh_pos=self.random_vehicle.pos[-1],
                                   veh_angle=self.random_vehicle.angle, individuals=individuals,
                                   generations=generations, crossover_rate=0.6, mutation_rate=0.3)
        self.brain = ga_result[0]
        predicted_sensors = ga_result[1]
        predicted_wheels = ga_result[2]

        # Create vehicle and pass it the wheel data, previous random movement, and then run. Get the pos data
        ga_prediction_vehicle = ControllableVehicle(self.random_vehicle.pos[-1], self.random_vehicle.angle)
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
        plt.plot(v_iter,  mser)
        plt.show()

    def wake_testing(self, iterations, benchmark):
        """ This phase uses the control system to iterate through many motor commands by passing them to the controlled
        robot in the world and retrieving its sensory information """
        new_vehicle = BrainVehicle(self.random_vehicle.pos[-1], self.random_vehicle.angle)
        new_vehicle.set_values(self.brain)
        new_vehicle.random_movement = self.random_vehicle.pos
        new_vehicle.predicted_movement = self.predicted_pos

        actual_vehicle = self.sim.run_simulation(iteration=iterations, graphics=True, cycle='wake (testing)',
                                                 vehicle=new_vehicle)

        # get positional information of vehicle and compare with predicted
        plt.figure(1)
        plt.suptitle('Predicted vs actual movement in wake (testing) cycle')
        plt.subplot(121)
        plt.title('Predicted vs actual movement in space')
        plt.xlabel('x coordinate')
        plt.ylabel('y coordinate')
        actual_move = np.transpose(actual_vehicle.pos)
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
        mse = [(((pred_move[0][it] - actual_move[0][it] + pred_move[1][it] - actual_move[1][it])/2) ** 2) /
               len(pred_move[0]) for it in range(0, len(pred_move[0]))]
        plt.plot(range(0, len(mse)), mse)

        plt.show()

        if benchmark:
            random_brain_benchmark(random_brains=10000, iterations=iterations, start_pos=self.random_vehicle.pos[-1],
                                   start_a=self.random_vehicle.angle, light=self.sim.light, actual=actual_vehicle)
