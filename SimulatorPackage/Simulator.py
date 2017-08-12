import time
import matplotlib.pyplot as plt
from Sprites import *


def show_sensors_motors(screen, vehicle, my_font):
    """
    This method draws the motor and sensory values onto the vehicle and also its trajectories.
    :param screen: the screen to draw on
    :param vehicle: the vehicle to draw
    :param my_font: the font for the sensory and motor values
    """
    bearing = vehicle.bearing[-1]
    radius = vehicle.radius + 5
    # show left wheel
    direction_x = (vehicle.pos[-1][0] - math.cos(bearing) * radius) + math.sin(bearing) * radius / 2
    direction_y = (vehicle.pos[-1][1] + math.sin(bearing) * radius) + math.cos(bearing) * radius / 2
    left_wheel = my_font.render(format(vehicle.wheel_l, '.1f'), 1, (0, 0, 0))
    screen.blit(left_wheel, [int(direction_x), int(direction_y)])
    # show right wheel
    direction_x = (vehicle.pos[-1][0] + math.cos(bearing) * radius) + math.sin(bearing) * radius / 2
    direction_y = (vehicle.pos[-1][1] - math.sin(bearing) * radius) + math.cos(bearing) * radius / 2
    right_wheel = my_font.render(format(vehicle.wheel_r, '.1f'), 1, (0, 0, 0))
    screen.blit(right_wheel, [int(direction_x), int(direction_y)])
    # show left sensor
    direction_x = (vehicle.pos[-1][0] - math.cos(bearing) * radius) - math.sin(bearing) * radius
    direction_y = (vehicle.pos[-1][1] + math.sin(bearing) * radius) - math.cos(bearing) * radius
    left_sensor = my_font.render(format(vehicle.sensor_left[-1], '.2f'), 1, (100, 100, 0))
    screen.blit(left_sensor, [int(direction_x), int(direction_y)])
    # show right sensor
    direction_x = (vehicle.pos[-1][0] + math.cos(bearing) * radius) - math.sin(bearing) * radius
    direction_y = (vehicle.pos[-1][1] - math.sin(bearing) * radius) - math.cos(bearing) * radius
    right_sensor = my_font.render(format(vehicle.sensor_right[-1], '.2f'), 1, (100, 100, 0))
    screen.blit(right_sensor, [int(direction_x), int(direction_y)])
    # show trajectories of Random vehicle
    if isinstance(vehicle, RandomMotorVehicle):  # random movement in blue
        [pygame.draw.circle(screen, (0, 0, 240), (int(p[0]), int(p[1])), 2) for p in vehicle.pos]
        if vehicle.previous_pos is not None:
            [pygame.draw.circle(screen, (100, 100, 100), (int(p[0]), int(p[1])), 2) for p in vehicle.previous_pos]
    # show trajectories of Controllable vehicle
    if isinstance(vehicle, ControllableVehicle):  # random in blue, predicted in grey
        [pygame.draw.circle(screen, (0, 0, 240), (int(p[0]), int(p[1])), 2) for p in vehicle.random_movement]
        [pygame.draw.circle(screen, (100, 100, 100), (int(p[0]), int(p[1])), 2) for p in vehicle.pos]
    # show trajectories of Brain vehicle
    if isinstance(vehicle, BrainVehicle):  # random in blue, predicted in grey, and actual in red
        if vehicle.training_movement is not None:
            [pygame.draw.circle(screen, (100, 120, 100), (int(p[0]), int(p[1])), 2) for p in vehicle.training_movement]
        [pygame.draw.circle(screen, (0, 0, 240), (int(p[0]), int(p[1])), 2) for p in vehicle.previous_movement]
        [pygame.draw.circle(screen, (100, 100, 100), (int(p[0]), int(p[1])), 2) for p in vehicle.predicted_movement]
        [pygame.draw.circle(screen, (240, 0, 0), (int(p[0]), int(p[1])), 2) for p in vehicle.pos]


def show_graph(vehicle):
    """
    Graph that shows the sensory and motor values of the vehicle in time.
    :param vehicle: the vehicle
    """
    i = range(0, len(vehicle.sensor_left))
    plt.plot(i, vehicle.sensor_left, 'r', label='sensor left')
    plt.plot(i, vehicle.sensor_right, 'y', label='sensor right')
    plt.plot(i, vehicle.motor_left, 'g', label='motor left')
    plt.plot(i, vehicle.motor_right, 'b', label='motor right')
    plt.legend()
    plt.show()


class Simulator:
    """
    Simulator class handles everything to do with the screen simulation of the Braitenberg Vehicles.
    """
    window_width = 1800  # 1240
    window_height = 1200  # 720

    def __init__(self, light):
        """
        The constructor for the Simulator class, creates a pygame, fonts, and light
        :param light: the light assigned to the simulation
        """
        pygame.init()
        self.values_font = pygame.font.SysFont('monospace', 12)
        self.cycle_font = pygame.font.SysFont('', 24)
        self.light = light

    def run_simulation(self, iteration, graphics, vehicle, cycle='', show_sen_mot_graph=False):
        """
        Runs a simulation with a vehicle and a light for a number of iterations.
        :param iteration: number of iterations to run for
        :param graphics: show simulator window or not
        :param vehicle: the vehicle to run
        :param cycle: which cycle the simulation is in, can be blank
        :param show_sen_mot_graph: shows the sensor and motor data over time
        :return: the vehicle that was just run
        """
        clock = pygame.time.Clock()  # clock to count ticks and fps
        all_sprites = pygame.sprite.RenderPlain(vehicle, self.light)
        if graphics:
            screen = pygame.display.set_mode((self.window_width, self.window_height))
            background = pygame.Surface(screen.get_size())
            background = background.convert(background)
            background.fill([240, 240, 240])

        for t in range(1, iteration):  # main for loop (runs from 1 because no movement at t=0)
            clock.tick()

            all_sprites.update(t)  # calls update() in Sprites

            if graphics:
                screen.blit(background, (0, 0))  # ignore warning, will only be referenced if assigned above
                if cycle == 'wake (training)' and t == iteration:  # just before goes to sleep write sleep
                    screen.blit(self.cycle_font.render('sleep', 1, (0, 0, 0)), [5, 5])
                else:
                    screen.blit(self.cycle_font.render(cycle, 1, (0, 0, 0)), [5, 5])
                all_sprites.draw(screen)  # draw sprites

                show_sensors_motors(screen, vehicle, self.values_font)  # draws values and trajectories

                pygame.display.flip()
                pygame.display.set_caption(
                    'Braitenberg vehicle simulation - ' + str(format(clock.get_fps(), '.0f')) + 'fps')

                time.sleep(0.03)

        if show_sen_mot_graph:
            show_graph(vehicle)

        return vehicle

    @staticmethod
    def close():
        pygame.display.quit()

    def quick_simulation(self, iteration, graphics=False, veh_pos=None, veh_angle=random.randint(0, 360),
                         previous_pos=None, gamma=0.3):
        """
        Creates a vehicle then runs the simulation calling run_simulation(). Used by collect_random_data
        :param iteration: number of iterations
        :param graphics: show simulator window or not
        :param veh_pos: initial vehicle position, will default to [300, 300]
        :param veh_angle: initial vehicle angle, will default to random
        :param previous_pos: previous position, for drawing on screen
        :param gamma: var to control random movement intensity
        :return: the vehicle after being run
        """
        if veh_pos is None:
            veh_pos = [300, 300]
        vehicle = RandomMotorVehicle(veh_pos, veh_angle, gamma, self.light)
        vehicle.previous_pos = previous_pos
        vehicle = self.run_simulation(iteration, graphics, vehicle)
        #self.close()
        return vehicle

    def init_simulation(self, iteration, graphics, cycle='', veh_pos=None, veh_angle=random.randint(0, 360),
                        gamma=0.3, brain=None):
        """
        Creates a vehicle and runs in in the simulation. Used by cycles
        :param iteration: number of iterations
        :param graphics: show simulator window or not
        :param cycle: current cycle the simulation is in (string just to display on screen)
        :param veh_pos: initial vehicle position, will default to [300, 300]
        :param veh_angle: initial vehicle angle, will default to random
        :param gamma: var to control random movement intensity
        :param brain: brain of vehicle
        :return: the vehicle after being run
        """
        if veh_pos is None:
            veh_pos = [300, 300]
        if brain is not None:
            vehicle = BrainVehicle(veh_pos, veh_angle, self.light)
            vehicle.set_values(brain)
        else:
            vehicle = RandomMotorVehicle(veh_pos, veh_angle, gamma, self.light)

        vehicle = self.run_simulation(iteration, graphics, vehicle, cycle=cycle)
        return vehicle
