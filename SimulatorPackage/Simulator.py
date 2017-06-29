import sys
import pygame
import math
import random
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from SimulatorPackage.Vehicles import Attacker, Vehicle
from SimulatorPackage.Light import Light


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


def show_graph(vehicle):
    i = range(1, iterations)

    plt.plot(i, vehicle.sensor_left, 'r', i, vehicle.sensor_right, 'y',i, vehicle.motor_left, 'g',
             i, vehicle.motor_right, 'b')
    red_line = mlines.Line2D([], [], color='red', label='left sensor')
    yellow_line = mlines.Line2D([], [], color='yellow', label='right sensor')
    green_line = mlines.Line2D([], [], color='green', label='left motor')
    blue_line = mlines.Line2D([], [], color='blue', label='right motor')
    plt.legend(handles=[red_line, yellow_line, green_line, blue_line])
    plt.show()


def run_simulation(iteration, graphics, clock, all_sprites, light):
    if graphics:
        screen = pygame.display.set_mode((window_width, window_height))
        background = pygame.Surface(screen.get_size())
        background = background.convert(background)
        background.fill([240, 240, 240])

    for t in range(1, iteration):
        clock.tick()
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                sys.exit()

        all_sprites.update(t, light.pos)

        if graphics:
            screen.blit(background, (0, 0))
            all_sprites.draw(screen)

            sprites = [sprite for sprite in all_sprites.sprites() if isinstance(sprite, Attacker) or isinstance(sprite, Vehicle)]
            [show_sensors_motors(screen, sprite) for sprite in sprites]

            pygame.display.flip()
            pygame.display.set_caption('Braitenberg vehicle simulation - ' + str(format(clock.get_fps(), '.0f')) + 'fps')

    if graphics:
        pygame.display.quit()
    print('Finished')

    show_graph(all_sprites.sprites()[0])


def run(iteration, graphics, veh_rand_pos, veh_rand_angle, light_rand_pos):
    clock = pygame.time.Clock()  # clock to count ticks and fps

    if veh_rand_pos:  # check if vehicle positions are random
        v1_x = random.randint(25, window_width - 25)
        v1_y = random.randint(25, window_height - 25)
    else:
        v1_x = 200
        v1_y = 200

    if veh_rand_angle:  # check if vehicle angle is random
        v1_angle = random.randint(0, 360)
    else:
        v1_angle = 0

    if light_rand_pos:  # check if light pos is random
        l_x = random.randint(25, window_width - 25)
        l_y = random.randint(25, window_height - 25)
    else:
        l_x = 400
        l_y = 400

    # create sprites
    v1 = Attacker([v1_x, v1_y], v1_angle)
    light = Light([l_x, l_y])
    all_sprites = pygame.sprite.RenderPlain(v1, light)

    run_simulation(iteration, graphics, clock, all_sprites, light)  # run simulation with given param

# pygame init
pygame.init()
window_width = 1280
window_height = 720

iterations = 500  # number of iterations to run simulation for
show_graphics = True  # True shows graphical window, False doesn't

for x in range(0, 1):
    run(iterations, show_graphics, True, True, True)

