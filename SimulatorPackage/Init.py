from SimulatorPackage.Simulator import Simulator
import pygame



iterations = 300  # number of iterations to run simulation for
show_graphics = True  # True shows graphical window, False doesn't

random_vehicle_pos = False
random_vehicle_angle = False
random_light_pos = False

#for x in range(0, 1):
sim = Simulator()
sim.init_simulation(iterations, show_graphics, random_vehicle_pos, random_vehicle_angle, random_light_pos)

