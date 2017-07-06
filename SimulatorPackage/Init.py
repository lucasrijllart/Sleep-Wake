import pyrenn as pr
import numpy as np
from Simulator import Simulator

iterations = 300  # number of iterations to run simulation for
show_graphics = True  # True shows graphical window, False doesn't

random_vehicle_pos = True
random_vehicle_angle = True
random_light_pos = True

# for x in range(0, 1):
sim = Simulator()
v = sim.init_simulation_random(iterations, show_graphics, random_vehicle_pos, random_vehicle_angle, random_light_pos)

# collect data for narx
inputs = []
targets = []
for i in range(0, len(v.motor_left)):
    inputs.append([v.sensor_left[i], v.sensor_right[i], v.motor_left[i], v.motor_right[i]])
    targets.append([v.sensor_left[i], v.sensor_right[i]])
print inputs
print targets



layers = [4, 10, 10, 2]
output_delay = [[1, 2, 3, 4]]
internal_delay = [1]
net = pr.CreateNN(layers, dIntern=internal_delay, dOut=output_delay)

