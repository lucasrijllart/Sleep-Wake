from Simulator import Simulator

iterations = 300  # number of iterations to run simulation for
show_graphics = True  # True shows graphical window, False doesn't

random_vehicle_pos = True
random_vehicle_angle = True
random_light_pos = True

# for x in range(0, 1):
sim = Simulator()
vehicle = sim.init_simulation(iterations, show_graphics, random_vehicle_pos, random_vehicle_angle, random_light_pos)

