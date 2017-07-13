import sys
import time
import pygame
import math
import random
import matplotlib.pyplot as plt
from Vehicles import BrainVehicle, RandomMotorVehicle
from Light import Light


def show_sensors_motors(screen, vehicle):
    bearing = vehicle.bearing[-1]
    radius = vehicle.radius + 5
    my_font = pygame.font.SysFont('monospace', 12)

    direction_x = (vehicle.pos[-1][0] - math.cos(bearing) * radius) + math.sin(bearing) * radius / 2
    direction_y = (vehicle.pos[-1][1] + math.sin(bearing) * radius) + math.cos(bearing) * radius / 2
    left_wheel = my_font.render(format(vehicle.wheel_l, '.1f'), 1, (0, 0, 0))
    screen.blit(left_wheel, [int(direction_x), int(direction_y)])

    direction_x = (vehicle.pos[-1][0] + math.cos(bearing) * radius) + math.sin(bearing) * radius / 2
    direction_y = (vehicle.pos[-1][1] - math.sin(bearing) * radius) + math.cos(bearing) * radius / 2
    right_wheel = my_font.render(format(vehicle.wheel_r, '.1f'), 1, (0, 0, 0))
    screen.blit(right_wheel, [int(direction_x), int(direction_y)])

    direction_x = (vehicle.pos[-1][0] - math.cos(bearing) * radius) - math.sin(bearing) * radius
    direction_y = (vehicle.pos[-1][1] + math.sin(bearing) * radius) - math.cos(bearing) * radius
    left_sensor = my_font.render(format(vehicle.sensor_left[-1], '.2f'), 1, (100, 100, 0))
    screen.blit(left_sensor, [int(direction_x), int(direction_y)])

    direction_x = (vehicle.pos[-1][0] + math.cos(bearing) * radius) - math.sin(bearing) * radius
    direction_y = (vehicle.pos[-1][1] - math.sin(bearing) * radius) - math.cos(bearing) * radius
    right_sensor = my_font.render(format(vehicle.sensor_right[-1], '.2f'), 1, (100, 100, 0))
    screen.blit(right_sensor, [int(direction_x), int(direction_y)])

    [pygame.draw.circle(screen, (240, 0, 0), (int(p[0]), int(p[1])), 2) for p in vehicle.pos]
    [pygame.draw.circle(screen, (0, 0, 240), (int(p[0]), int(p[1])), 2) for p in vehicle.previous_pos]


def show_graph(vehicle):
    i = range(0, len(vehicle.sensor_left))

    plt.plot(i, vehicle.sensor_left, 'r', i, vehicle.sensor_right, 'y', i, vehicle.motor_left, 'g',
             i, vehicle.motor_right, 'b')
    # red_line = mlines.Line2D([], [], color='red', label='left sensor')
    # yellow_line = mlines.Line2D([], [], color='yellow', label='right sensor')
    # green_line = mlines.Line2D([], [], color='green', label='left motor')
    # blue_line = mlines.Line2D([], [], color='blue', label='right motor')
    # plt.legend(handles=[red_line, yellow_line, green_line, blue_line])
    plt.show()


class Simulator:

    def __init__(self):
        pygame.init()
        self.window_width = 1240
        self.window_height = 720
        self.number_of_iterations = 0
        self.light = None

    def run_simulation(self, iteration, graphics, vehicle, show_sen_mot_graph=False):
        clock = pygame.time.Clock()  # clock to count ticks and fps
        all_sprites = pygame.sprite.RenderPlain(vehicle, self.light)
        if graphics:
            screen = pygame.display.set_mode((self.window_width, self.window_height))
            background = pygame.Surface(screen.get_size())
            background = background.convert(background)
            background.fill([240, 240, 240])
            self.number_of_iterations = 0

        for t in range(1, iteration):
            clock.tick()
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    sys.exit()

            # conditions for simulation stop: light and maybe out of bounds
            # if int(vehicle.pos[-1][0]) == light.pos[0] and int(vehicle.pos[-1][1]) == light.pos[1]:
                # break

            all_sprites.update(t, self.light)

            if graphics:
                screen.blit(background, (0, 0))
                all_sprites.draw(screen)

                show_sensors_motors(screen, vehicle)

                pygame.display.flip()
                pygame.display.set_caption('Braitenberg vehicle simulation - ' + str(format(clock.get_fps(), '.0f')) + 'fps')

                time.sleep(0.03)
            self.number_of_iterations += 1

        if show_sen_mot_graph:
            show_graph(vehicle)

        if graphics:
            print 'PyGame iterations: ' + str(self.number_of_iterations)

        return vehicle

    def init_simulation_random(self, iteration, graphics, veh_rand_pos, veh_rand_angle, light_rand_pos):
        if veh_rand_pos:  # check if vehicle positions are random
            v1_x = random.randint(400, self.window_width - 400)  # was 25
            v1_y = random.randint(100, self.window_height - 100)
        else:
            v1_x = 300
            v1_y = 300

        print veh_rand_angle
        if veh_rand_angle:  # check if vehicle angle is random
            v1_angle = random.randint(0, 360)
        else:
            v1_angle = 180

        print v1_angle
        if light_rand_pos:  # check if light pos is random
            l_x = random.randint(400, self.window_width - 400)
            l_y = random.randint(100, self.window_height - 100)
        else:
            l_x = 1100
            l_y = 600

        # create sprites
        vehicle = BrainVehicle([v1_x, v1_y], v1_angle)
        light = Light([l_x, l_y])

        return self.run_simulation(iteration, graphics, vehicle, light)  # run simulation with given param

    @staticmethod
    def close():
        pygame.display.quit()

    def quick_simulation(self, iteration, graphics, veh_pos=[300, 300], veh_angle=random.randint(0, 360),
                         light_pos=[1100, 600], gamma=0.2, use_seed=None, brain=False):
        """ Runs a simulation then closes then window """
        if brain:
            vehicle = BrainVehicle(veh_pos, veh_angle)
        else:
            vehicle = RandomMotorVehicle(veh_pos, veh_angle, gamma, use_seed)
        self.light = Light(light_pos)
        vehicle = self.run_simulation(iteration, graphics, vehicle)
        self.close()
        return vehicle

    def init_simulation(self, iteration, graphics, veh_pos=[300, 300], veh_angle=random.randint(0, 360),
                         light_pos=[1100, 600], gamma=0.2, use_seed=None, brain=False):
        """ Runs a simulation but doesn't closes the window, used to keep the simulation going with cycles """
        if brain:
            vehicle = BrainVehicle(veh_pos, veh_angle)
        else:
            vehicle = RandomMotorVehicle(veh_pos, veh_angle, gamma, use_seed)
        self.light = Light(light_pos)
        vehicle = self.run_simulation(iteration, graphics, vehicle)
        return vehicle
