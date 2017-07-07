import pyrenn as pr
import numpy as np
from narx import narx
from Simulator import Simulator
from GA import GA
import random
import matplotlib.pyplot as plt


def pre_processing(data):
    new_inputs = []
    new_targets = []
    for t in range(0, len(data[0])):
        for vehicle in range(0, len(data)):
            new_inputs.append(np.transpose(np.array(data[vehicle][t])))
            new_targets.append(np.transpose(np.array(data[vehicle][t][-2:])))  # extracting targets from data and adding to new list (transposed)
    return [new_inputs, new_targets]


def collect_random_data(vehicle_pos=None, vehicle_angle=None, light_pos=None, vehicle_runs=10, iterations=1000, graphics=False, gamma=0.2):
    if vehicle_pos is None:
        vehicle_pos = [300, 300]
    if vehicle_angle is None:
        vehicle_angle = random.randint(0, 360)
    if light_pos is None:
        light_pos = [1100, 600]
    # add 1 because the simulation returns iterations-1 as the first time step is the starting position (not recorded)
    data = []
    sim = Simulator()
    for run in range(0, vehicle_runs):
        v = sim.init_simulation(iterations+1, graphics, vehicle_pos, vehicle_angle, light_pos, gamma)
        vehicle_data_in_t = []
        for t in range(0, iterations):
            vehicle_data_in_t.append([v.motor_left[t], v.motor_right[t], v.sensor_left[t], v.sensor_right[t]])
        data.append(vehicle_data_in_t)
    print 'Collected data from ' + str(vehicle_runs) + ' vehicles over ' + str(iterations) + ' iterations'
    return data


# GA(graphics=True).run([300, 300], random.randint(0, 360), [1100, 600])


# collect data for narx and pre-process data
data = collect_random_data(vehicle_runs=5, iterations=1000)
inputs_list, targets_list = pre_processing(data)

# separation into training and testing
train_input = np.transpose(np.array(inputs_list[:60]))
train_target = np.transpose(np.array(targets_list[:60]))
test_input = np.transpose(np.array(inputs_list[-40:]))
test_target = np.transpose(np.array(targets_list[-40:]))

# create narx network
net = narx(input_delay=10, output_delay=10)

# train network
net.train(train_input, train_target, verbose=True, max_iter=400)

# extract predictions and compare with test
predictions = net.predict(test_input)
print predictions.shape
predictions1 = predictions[0]
predictions2 = predictions[1]
test_target1 = np.array(test_target)[0]
print test_target1.shape
test_target2 = np.array(test_target)[1]
i = np.array(range(0, len(test_target1)))

MSE1 = []
for it in range(0, len(predictions1)):
    print (predictions1[it] - test_target1[it]) ** 2 / len(predictions1)
    MSE1.append((predictions1[it] - test_target1[it]) ** 2 / len(predictions1))
print MSE1

#MSE2 = [(predictions2[i] - test_target2[i])**2 / 2 for i in len(predictions1)]

plt.figure(1)
plt.plot(range(0, len(MSE1)), MSE1)
print i.shape
print test_target1.shape
print predictions1.shape

plt.figure(2)
plt.plot(i, test_target1, 'b', i, predictions1, 'r')
plt.show()


