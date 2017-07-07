import pyrenn as pr
import numpy as np
from narx import narx
from Simulator import Simulator
from GA import GA
import random


def collect_random_data(vehicle_pos=None, vehicle_angle=None, light_pos=None, vehicle_runs=10, iterations=1000, graphics=False, gamma=0.2):
    if vehicle_pos is None:
        vehicle_pos = [300, 300]
    if vehicle_angle is None:
        vehicle_angle = random.randint(0, 360)
    if light_pos is None:
        light_pos = [1100, 600]
    # add 1 because the simulation returns iterations-1 as the first time step is the starting position (not recorded)
    data = []
    targets = []
    sim = Simulator()
    for run in range(0, vehicle_runs):

        v = sim.init_simulation(iterations+1, graphics, vehicle_pos, vehicle_angle, light_pos, gamma)
        vehicle_data_in_t = []
        target_data_in_t = []
        for t in range(0, iterations):
            vehicle_data_in_t.append([v.motor_left[t], v.motor_right[t], v.sensor_left[t], v.sensor_right[t]])
            target_data_in_t.append([v.sensor_left[t], v.sensor_right[t]])
        data.append(vehicle_data_in_t)
        targets.append(target_data_in_t)
    print 'Collected data from ' + str(vehicle_runs) + ' vehicles over ' + str(iterations) + ' iterations'
    return [data, targets]


# GA(graphics=True).run([300, 300], random.randint(0, 360), [1100, 600])



net = narx()


# collect data for narx
inputs, targets = collect_random_data(vehicle_runs=3)


#rotate arrays, each row in an input element, each column is a sample
inputs_array = np.array(inputs_list)
inputs_array = np.rot90(inputs_array, 1)
targets_array = np.array(targets_list)
targets_array = np.rot90(targets_array, 1)
x = [inputs_array[i][-1] for i in range(0, 4)]
x = np.array([x])
x = np.rot90(x, 1)
r, c = inputs_array.shape
#remove last column to use as test
inputs_array = inputs_array[:, 1:c-1]
print inputs_array
y = [targets_array[i][-1] for i in range(0, 2)]
y = np.array(y)
r, c = targets_array.shape
targets_array = targets_array[:, 1:c-1]
net.train(inputs_array, targets_array, verbose=True)
print net.predict(x)
# layers = [4, 10, 10, 2]
# output_delay = [1, 2, 3, 4]
# input_delay = [1, 2, 3, 4]
# net = pr.CreateNN(layers, dIn=input_delay, dOut=output_delay)
#
# net = pr.train_LM(inputs_array, targets_array, net, 100, verbose=True)
# print 'test input ' + str(x)
# print 'test target ' + str(y)
#
# print pr.NNOut(x, net)

# check this methods
#pr.prepare_data()

