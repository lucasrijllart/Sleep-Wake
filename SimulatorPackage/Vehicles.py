import pygame
import math


class Vehicle(pygame.sprite.Sprite):

    def __init__(self, pos, angle):
        # pygame init
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.image.load('images/vehicle.png')  # image of vehicle
        self.original = self.image  # original image to use when rotating
        self.rect = self.image.get_rect()  # rectangle bounds of image
        self.rect.center = pos # set bounds as vehicle starting position
        self.angle = angle  # starting angle
        self.image = pygame.transform.rotozoom(self.original, self.angle, 0.5)

        # vehicle logic init
        self.dt = 0.07
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
        self.image = pygame.image.load('images/attacker.png')  # image of vehicle
        self.original = self.image  # original image to use when rotating
        self.rect = self.image.get_rect()  # rectangle bounds of image
        self.rect.center = pos # set bounds as vehicle starting position
        self.angle = angle  # starting angle
        self.image = pygame.transform.rotozoom(self.original, self.angle, 0.5)

        # vehicle logic init
        self.dt = 1
        self.wheel_l, self.wheel_r = 0, 0  # velocity for left and right wheels
        self.radius = 25  # radius of vehicle size
        self.pos = [pos]  # xy position of vehicle
        self.bearing = [float(angle * math.pi / 180)]  # angle of vehicle (converted to rad)
        self.sensor_gain = 100  # amplify sensor signal
        self.motor_gain_ll, self.motor_gain_rl, self.motor_gain_rr, self.motor_gain_lr = 0, 10, 0, 10  # amplify motor signal
        self.bias_l, self.bias_r = 2, 2  # automatically added wheel bias to wheels
        self.reached_light = False

        # vehicle sensory and motor information to extract for neural network
        self.sensor_left = []
        self.sensor_right = []
        self.motor_left = []
        self.motor_right = []

    def set_values(self, ll, lr, rr, rl, bl, br):
        self.motor_gain_ll = ll
        self.motor_gain_lr = lr
        self.motor_gain_rr = rr
        self.motor_gain_rl = rl
        self.bias_l = bl
        self.bias_r = br

    def update(self, t, light):
        self.update_sensors(t, light.pos)
        self.check_reached_light(light)
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
        distance_l = 1 / (distance_l - 1) ** 2
        distance_r = 1 / (distance_r - 1) ** 2
        # print('dl:', distance_l, 'dr:', distance_r)

        # calculate sensor intensity
        sensor_l, sensor_r = [self.sensor_gain * distance_l, self.sensor_gain * distance_r]
        self.sensor_left.append(sensor_l)
        self.sensor_right.append(sensor_r)
        # print('sl:', sensor_l, 'sr:', sensor_r)

        # calculate motor intensity
        self.wheel_l, self.wheel_r = [(sensor_l * self.motor_gain_ll) + (sensor_r * self.motor_gain_rl) + self.bias_l,
                                      (sensor_r * self.motor_gain_rr) + (sensor_l * self.motor_gain_lr) + self.bias_r]
        self.motor_left.append(self.wheel_l)
        self.motor_right.append(self.wheel_r)
        # print(self.motors[-1])

    def check_reached_light(self, light):
        # get distance
        distance_to_light = math.sqrt((light.pos[0] - self.pos[-1][0])**2 + (light.pos[1] - self.pos[-1][1])**2)
        if distance_to_light < (light.rect.width + light.rect.height)/6:
            self.reached_light = True

    def update_graphics(self):
        previous_center = self.pos[-1]
        degree = self.bearing[-1] * 180 / math.pi
        self.image = pygame.transform.rotozoom(self.original, degree, 0.5)
        self.rect = self.image.get_rect()
        self.rect.center = previous_center
