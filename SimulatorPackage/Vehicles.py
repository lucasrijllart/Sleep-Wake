import pygame
import math


class Vehicle(pygame.sprite.Sprite):

    def __init__(self, pos, angle):
        # pygame init
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.image.load('images/simple1.png')  # image of vehicle  TODO: change image
        self.original = self.image  # original image to use when rotating
        self.rect = self.image.get_rect()  # rectangle bounds of image
        self.rect.center = pos # set bounds as vehicle starting position
        self.angle = angle  # starting angle
        self.image = pygame.transform.rotozoom(self.original, self.angle, 0.5)

        # vehicle logic init
        self.dt = 0.5
        self.wheel_l, self.wheel_r = 0, 0  # velocity for left and right wheels
        self.radius = 25  # radius of vehicle size
        self.pos = [pos]  # xy position of vehicle
        self.bearing = [angle / 360 * 2 * math.pi]  # angle of vehicle (converted to rad)
        self.sensor_gain = 100  # amplify sensor signal
        self.motor_gain = 10  # amplify motor signal
        self.bias = 2  # automatically added wheel bias to both

        # vehicle sensory and motor information to extract for neural network
        self.sensor_left = []
        self.sensor_right = []
        self.motor_left = []
        self.motor_right = []

    def update(self, t, light_pos):
        self.update_sensors(t, light_pos)
        self.update_graphics()

    def update_sensors(self, t, light_pos):
        # print('\nt:', t)

        vc = (self.wheel_l + self.wheel_r) / 2  # velocity center
        va = (self.wheel_r - self.wheel_l) / (2 * self.radius)  # velocity average

        # print('pos-1: ', self.pos[-1])
        self.pos.append([self.pos[-1][0] - self.dt * vc * math.sin(self.bearing[t-1]),  # changed top to sin and bottom to cos and it worked
                         self.pos[-1][1] - self.dt * vc * math.cos(self.bearing[t-1])])  # update position
        self.bearing.append(math.fmod(self.bearing[t-1] + self.dt * va, 2 * math.pi))  # update bearing
        # print('pos: ', self.pos[t])
        # print('bea:', self.bearing[-1])

        # calculate left sensor position
        sl0 = (self.pos[t][0] - math.cos(self.bearing[t]) * self.radius) - math.sin(self.bearing[t]) * self.radius
        sl1 = (self.pos[t][1] + math.sin(self.bearing[t]) * self.radius) - math.cos(self.bearing[t]) * self.radius
        # print('sl0:', sl0, 'sl1:', sl1)

        # calculate right sensor position
        sr0 = (self.pos[t][0] + math.cos(self.bearing[t]) * self.radius) - math.sin(self.bearing[t]) * self.radius
        sr1 = (self.pos[t][1] - math.sin(self.bearing[t]) * self.radius) - math.cos(self.bearing[t]) * self.radius
        # print('sr0:', sr0, 'sr1:', sr1, 'topr:', self.rect.topright,)

        # calculate square distance to light
        distance_l = math.sqrt((light_pos[0] - sl0) ** 2 + (light_pos[1] - sl1) ** 2)
        distance_r = math.sqrt((light_pos[0] - sr0) ** 2 + (light_pos[1] - sr1) ** 2)
        # print('dl:', distance_l, 'dr:', distance_r)

        # calculate sensor intensity
        sensor_l, sensor_r = [self.sensor_gain / distance_l, self.sensor_gain / distance_r]
        self.sensor_left.append(sensor_l)
        self.sensor_right.append(sensor_r)
        # print('sl:', sensor_l, 'sr:', sensor_r)

        # add motors to list
        self.motor_left.append(self.wheel_l)
        self.motor_right.append(self.wheel_r)
        # print(self.motors[-1])

    def update_graphics(self):
        previous_center = self.pos[-1]
        degree = self.bearing[-1] * 180 / math.pi
        self.image = pygame.transform.rotozoom(self.original, degree, 0.5)
        self.rect = self.image.get_rect()
        self.rect.center = previous_center


class Attacker(pygame.sprite.Sprite):

    def __init__(self, pos, angle):
        # pygame init
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.image.load('images/simple1.png')  # image of vehicle
        self.original = self.image  # original image to use when rotating
        self.rect = self.image.get_rect()  # rectangle bounds of image
        self.rect.center = pos # set bounds as vehicle starting position
        self.angle = angle  # starting angle
        self.image = pygame.transform.rotozoom(self.original, self.angle, 0.5)

        # vehicle logic init
        self.dt = 0.5
        self.wheel_l, self.wheel_r = 0, 0  # velocity for left and right wheels
        self.radius = 25  # radius of vehicle size
        self.pos = [pos]  # xy position of vehicle
        self.bearing = [angle / 360 * 2 * math.pi]  # angle of vehicle (converted to rad)
        self.sensor_gain = 100  # amplify sensor signal
        self.motor_gain = 10  # amplify motor signal
        self.bias = 2  # automatically added wheel bias to both

        # vehicle sensory and motor information to extract for neural network
        self.sensor_left = []
        self.sensor_right = []
        self.motor_left = []
        self.motor_right = []

    def update(self, t, light_pos):
        self.update_sensors(t, light_pos)
        self.update_graphics()

    def update_sensors(self, t, light_pos):
        # print('\nt:', t)

        vc = (self.wheel_l + self.wheel_r) / 2  # velocity center
        va = (self.wheel_r - self.wheel_l) / (2 * self.radius)  # velocity average

        # print('pos-1: ', self.pos[-1])
        self.pos.append([self.pos[-1][0] - self.dt * vc * math.sin(self.bearing[t-1]),  # changed top to sin and bottom to cos and it worked
                         self.pos[-1][1] - self.dt * vc * math.cos(self.bearing[t-1])])  # update position
        self.bearing.append(math.fmod(self.bearing[t-1] + self.dt * va, 2 * math.pi))  # update bearing
        # print('pos: ', self.pos[t])
        # print('bea:', self.bearing[-1])

        # calculate left sensor position
        sl0 = (self.pos[t][0] - math.cos(self.bearing[t]) * self.radius) - math.sin(self.bearing[t]) * self.radius
        sl1 = (self.pos[t][1] + math.sin(self.bearing[t]) * self.radius) - math.cos(self.bearing[t]) * self.radius
        # print('sl0:', sl0, 'sl1:', sl1)

        # calculate right sensor position
        sr0 = (self.pos[t][0] + math.cos(self.bearing[t]) * self.radius) - math.sin(self.bearing[t]) * self.radius
        sr1 = (self.pos[t][1] - math.sin(self.bearing[t]) * self.radius) - math.cos(self.bearing[t]) * self.radius
        # print('sr0:', sr0, 'sr1:', sr1, 'topr:', self.rect.topright,)

        # calculate square distance to light
        distance_l = math.sqrt((light_pos[0] - sl0) ** 2 + (light_pos[1] - sl1) ** 2)
        distance_r = math.sqrt((light_pos[0] - sr0) ** 2 + (light_pos[1] - sr1) ** 2)
        # print('dl:', distance_l, 'dr:', distance_r)

        # calculate sensor intensity
        sensor_l, sensor_r = [self.sensor_gain / distance_l, self.sensor_gain / distance_r]
        self.sensor_left.append(sensor_l)
        self.sensor_right.append(sensor_r)
        # print('sl:', sensor_l, 'sr:', sensor_r)

        # calculate motor intensity
        self.wheel_l, self.wheel_r = [(sensor_r * self.motor_gain) + self.bias,
                                      (sensor_l * self.motor_gain) + self.bias]
        self.motor_left.append(self.wheel_l)
        self.motor_right.append(self.wheel_r)
        # print(self.motors[-1])

    def update_graphics(self):
        previous_center = self.pos[-1]
        degree = self.bearing[-1] * 180 / math.pi
        self.image = pygame.transform.rotozoom(self.original, degree, 0.5)
        self.rect = self.image.get_rect()
        self.rect.center = previous_center
