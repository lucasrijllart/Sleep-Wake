import numpy as np
from Narx import Narx
from Simulator import Simulator
from GA import GA
import random
import matplotlib.pyplot as plt
from decimal import Decimal


def pre_processing(raw_data):
    new_inputs = []
    new_targets = []
    for t in range(0, len(raw_data[0])):
        for vehicle in range(0, len(raw_data)):
            new_inputs.append(np.transpose(np.array(raw_data[vehicle][t])))
            new_targets.append(np.transpose(np.array(raw_data[vehicle][t][-2:])))  # extracting targets from data and adding to new list (transposed)
    return [new_inputs, new_targets]


def collect_random_data(vehicle_pos=None, vehicle_angle=None, light_pos=None, runs=10, iterations=1000, graphics=False, gamma=0.2, seed=None):
    if vehicle_pos is None:
        vehicle_pos = [300, 300]
    if vehicle_angle is None:
        vehicle_angle = random.randint(0, 360)
    if light_pos is None:
        light_pos = [1100, 600]
    # add 1 because the simulation returns iterations-1 as the first time step is the starting position (not recorded)
    data = []
    sim = Simulator()
    for run in range(0, runs):
        v = sim.init_simulation(iterations + 1, graphics, vehicle_pos, vehicle_angle, light_pos, gamma, seed)
        vehicle_data_in_t = []
        for t in range(0, iterations):
            vehicle_data_in_t.append([v.motor_left[t], v.motor_right[t], v.sensor_left[t], v.sensor_right[t]])
        data.append(vehicle_data_in_t)
    print 'Collected data from ' + str(runs) + ' vehicles over ' + str(iterations) + ' iterations'
    return data


# GA(graphics=True).run([300, 300], random.randint(0, 360), [1100, 600])

# Parameters
vehicle_runs = 4
vehicle_iter = 400
test_runs = 1
test_iter = 400
test_seed = 200  # 100

input_delay = 15
output_delay = 15
net_max_iter = 200

# collect data for narx and pre-process data
data = collect_random_data(runs=vehicle_runs, iterations=vehicle_iter)
inputs_list, targets_list = pre_processing(data)

# test runs and preprocess data
data = collect_random_data(runs=test_runs, seed=test_seed, vehicle_angle=100, iterations=test_iter, graphics=False)
test_input, test_target = pre_processing(data)

# separation into training and testing
train_input = np.transpose(np.array(inputs_list))
train_target = np.transpose(np.array(targets_list))
test_input = np.transpose(np.array(test_input))
test_target = np.transpose(np.array(test_target))

# create narx network
net = Narx(input_delay=input_delay, output_delay=output_delay)

# train network
net.train(train_input, train_target, verbose=True, max_iter=net_max_iter)

# extract predictions and compare with test
predictions = net.predict(test_input)
predictions_left = predictions[0]
predictions_right = predictions[1]
real_left = np.array(test_target)[0]
real_right = np.array(test_target)[1]
i = np.array(range(0, len(real_left)))

# calculate mean squared error
MSE1 = [(predictions_left[it] - real_left[it]) ** 2 / len(predictions_left) for it in range(0, len(predictions_left))]
MSE2 = [(predictions_right[it] - real_right[it]) ** 2 / len(predictions_right) for it in range(0, len(predictions_right))]

plt.figure(1)
plt.suptitle('Results for: veh_runs=' + str(vehicle_runs) + ' veh_iter=' + str(vehicle_iter) + ' delays=' +
             str(input_delay) + ':' + str(output_delay) + ' net_iter:' + str(net_max_iter))
plt.subplot(221)
plt.title('Left sensor MSE. Mean:' + '%.4E' % Decimal(str(np.mean(MSE1))))
plt.plot(range(0, len(MSE1)), MSE1)

plt.subplot(222)
plt.title('Right sensor MSE. Mean:' + '%.4E' % Decimal(str(np.mean(MSE2))))
plt.plot(range(0, len(MSE2)), MSE2)

plt.subplot(223)
plt.title('Left sensor values b=real, r=pred')
plt.plot(i, real_left, 'b', i, predictions_left, 'r')

plt.subplot(224)
plt.title('Right sensor values b=real, r=pred')
plt.plot(i, real_right, 'b', i, predictions_right, 'r')

plt.show()


