import pyrenn as pr
import numpy as np
from narx import narx
from Simulator import Simulator
from GA import GA
import random


GA(graphics=True).run([300, 300], random.randint(0, 360), [1100, 600])

iterations = 300  # number of iterations to run simulation for
show_graphics = False  # True shows graphical window, False doesn't

random_vehicle_pos = False
random_vehicle_angle = False
random_light_pos = False

# for x in range(0, 1):
sim = Simulator()
v = sim.init_simulation_random(iterations, show_graphics, random_vehicle_pos, random_vehicle_angle, random_light_pos)


net = narx()


# collect data for narx
inputs_list = []
targets_list = []
for i in range(0, len(v.motor_left)):
    inputs_list.append([v.sensor_left[i], v.sensor_right[i], v.motor_left[i], v.motor_right[i]])
    targets_list.append([v.sensor_left[i], v.sensor_right[i]])
# print inputs_list
# print targets_list


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
#pr.RTRL()
#pr.prepare_data()

