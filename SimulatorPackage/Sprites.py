import pygame
import math
import random


class ControllableVehicle(pygame.sprite.Sprite):
    def __init__(self, start_pos, start_angle):
        # pygame init
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.image.load('images/vehicle.png')  # image of vehicle
        self.original = self.image  # original image to use when rotating
        self.rect = self.image.get_rect()  # rectangle bounds of image
        self.rect.center = start_pos  # set bounds as vehicle starting position
        self.angle = start_angle  # starting angle
        self.image = pygame.transform.rotozoom(self.original, self.angle, 0.5)
        self.radius = 25  # radius of vehicle size

        # vehicle sensory and motor information to extract for neural network
        self.sensor_left = []
        self.sensor_right = []
        self.motor_left = []
        self.motor_right = []

        # vehicle logic init
        self.dt = 40
        self.wheel_l, self.wheel_r = 0, 0
        self.wheel_data = []
        self.pos = [start_pos]  # xy position of vehicle
        self.bearing = [float(start_angle * math.pi / 180)]  # angle of vehicle (converted to rad)
        self.previous_pos = []
        self.brain = []

        # weights sensor->motor (lr = left sensor to right wheel)
        self.w_ll, self.w_lr, self.w_rr, self.w_rl = 0, 0, 0, 0
        self.bias_l, self.bias_r = 0, 0

    def set_wheels(self, wheel_data):
        self.wheel_data = wheel_data

    def set_values(self, ll, lr, rr, rl, bl, br):
        self.w_ll = ll
        self.w_lr = lr
        self.w_rr = rr
        self.w_rl = rl
        self.bias_l = bl
        self.bias_r = br

    def update(self, t, light):
        self.update_vehicle(t, light.pos)
        self.update_graphics()

    def update_vehicle(self, t, light_pos):
        # print('\nt:', t)

        vc = (self.wheel_l + self.wheel_r) / 2  # velocity center
        va = (self.wheel_r - self.wheel_l) / (2 * self.radius)  # velocity average

        # print('pos-1: ', self.pos[-1])
        self.pos.append([self.pos[-1][0] - self.dt * vc * math.sin(self.bearing[t - 1]),
                         # changed top to sin and bottom to cos and it worked
                         self.pos[-1][1] - self.dt * vc * math.cos(self.bearing[t - 1])])  # update position
        self.bearing.append(math.fmod(self.bearing[t - 1] + self.dt * va, 2 * math.pi))  # update bearing
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
        sensor_l = 10 / (distance_l + 1)
        sensor_r = 10 / (distance_r + 1)
        # print('dl:', distance_l, 'dr:', distance_r)

        # calculate sensor intensity
        # sensor_l, sensor_r = [self.sensor_gain * distance_l, self.sensor_gain * distance_r]
        self.sensor_left.append(sensor_l)
        self.sensor_right.append(sensor_r)
        # print('sl:', sensor_l, 'sr:', sensor_r)

        # calculate motor intensity
        self.wheel_l, self.wheel_r = self.wheel_data.pop(0)
        self.motor_left.append(self.wheel_l)
        self.motor_right.append(self.wheel_r)
        # print(self.motors[-1])

    def update_graphics(self):
        previous_center = self.pos[-1]
        degree = self.bearing[-1] * 180 / math.pi
        self.image = pygame.transform.rotozoom(self.original, degree, 0.5)
        self.rect = self.image.get_rect()
        self.rect.center = previous_center


class RandomMotorVehicle(pygame.sprite.Sprite):
    def __init__(self, start_pos, start_angle, gamma, seed):
        # pygame init
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.image.load('images/vehicle.png')  # image of vehicle
        self.original = self.image  # original image to use when rotating
        self.rect = self.image.get_rect()  # rectangle bounds of image
        self.rect.center = start_pos  # set bounds as vehicle starting position
        self.angle = start_angle  # starting angle
        self.image = pygame.transform.rotozoom(self.original, self.angle, 0.5)

        # vehicle logic init
        self.gamma = gamma
        if seed is not None:
            random.seed(seed)
        self.dt = 40  # 80
        self.wheel_l, self.wheel_r = random.uniform(-0.05, 0.05), random.uniform(-0.05, 0.05)  # velocity for left and right wheels
        self.radius = 25  # radius of vehicle size
        self.pos = [start_pos]  # xy position of vehicle
        self.bearing = [float(start_angle * math.pi / 180)]  # angle of vehicle (converted to rad)
        self.previous_pos = []

        # vehicle sensory and motor information to extract for neural network
        self.sensor_left = []
        self.sensor_right = []
        self.motor_left = []
        self.motor_right = []

    def update(self, t, light):
        self.update_sensors(t, light.pos)
        self.update_graphics()

    def update_sensors(self, t, light_pos):
        # print('\nt:', t)
        vc = (self.wheel_l + self.wheel_r) / 2  # velocity center
        va = (self.wheel_r - self.wheel_l) / (2 * self.radius)  # velocity average

        # print('pos-1: ', self.pos[-1])
        self.pos.append([self.pos[-1][0] - self.dt * vc * math.sin(self.bearing[t - 1]),
                         # changed top to sin and bottom to cos and it worked
                         self.pos[-1][1] - self.dt * vc * math.cos(self.bearing[t - 1])])  # update position
        self.bearing.append(math.fmod(self.bearing[t - 1] + self.dt * va, 2 * math.pi))  # update bearing
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
        sensor_l = 10 / (distance_l + 1)
        sensor_r = 10 / (distance_r + 1)
        # print('dl:', distance_l, 'dr:', distance_r)

        # calculate sensor intensity
        # sensor_l, sensor_r = [self.sensor_gain * distance_l, self.sensor_gain * distance_r]
        self.sensor_left.append(sensor_l)
        self.sensor_right.append(sensor_r)
        # print('sl:', sensor_l, 'sr:', sensor_r)

        # smooth out formula, add bias maybe
        # calculate motor intensity
        self.wheel_l, self.wheel_r = [self.wheel_l + 0.05 * (-self.wheel_l + random.uniform(-1, 1)) + 0.02,
                                      self.wheel_r + 0.05 * (-self.wheel_r + random.uniform(-1, 1)) + 0.02]
        self.motor_left.append(self.wheel_l)
        self.motor_right.append(self.wheel_r)
        # print(self.motors[-1])

    def update_graphics(self):
        previous_center = self.pos[-1]
        degree = self.bearing[-1] * 180 / math.pi
        self.image = pygame.transform.rotozoom(self.original, degree, 0.5)
        self.rect = self.image.get_rect()
        self.rect.center = previous_center


class BrainVehicle(pygame.sprite.Sprite):

    def __init__(self, start_pos, start_angle):
        # pygame init
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.image.load('images/attacker.png')  # image of vehicle
        self.original = self.image  # original image to use when rotating
        self.rect = self.image.get_rect()  # rectangle bounds of image
        self.rect.center = start_pos  # set bounds as vehicle starting position
        self.angle = start_angle  # starting angle
        self.image = pygame.transform.rotozoom(self.original, self.angle, 0.5)

        # vehicle logic init
        self.dt = 40
        self.wheel_l, self.wheel_r = 0, 0  # velocity for left and right wheels
        self.radius = 25  # radius of vehicle size
        self.pos = [start_pos]  # xy position of vehicle
        self.bearing = [float(start_angle * math.pi / 180)]  # angle of vehicle (converted to rad)
        # self.sensor_gain = 1  # amplify sensor signal
        self.motor_gain_ll, self.motor_gain_rl, self.motor_gain_rr, self.motor_gain_lr = 0, 8, 0, 8  # amplify motor signal
        self.bias_l, self.bias_r = 2, 2  # automatically added wheel bias to wheels
        self.reached_light = False
        self.previous_pos = []  # keeps track of the movement before the brain (random movements)

        # vehicle sensory and motor information to extract for neural network
        self.sensor_left = []
        self.sensor_right = []
        self.motor_left = []
        self.motor_right = []

    def set_values(self, ll_lr_rr_rl_bl_br):
        self.motor_gain_ll = ll_lr_rr_rl_bl_br[0]
        self.motor_gain_lr = ll_lr_rr_rl_bl_br[1]
        self.motor_gain_rr = ll_lr_rr_rl_bl_br[2]
        self.motor_gain_rl = ll_lr_rr_rl_bl_br[3]
        self.bias_l = ll_lr_rr_rl_bl_br[4]
        self.bias_r = ll_lr_rr_rl_bl_br[5]

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
        sensor_l = 10 / (distance_l + 1)
        sensor_r = 10 / (distance_r + 1)
        # print('dl:', distance_l, 'dr:', distance_r)

        # calculate sensor intensity
        # sensor_l, sensor_r = [self.sensor_gain * distance_l, self.sensor_gain * distance_r]
        self.sensor_left.append(sensor_l)
        self.sensor_right.append(sensor_r)
        # print('sl:', sensor_l, 'sr:', sensor_r)

        # calculate motor intensity
        self.wheel_l, self.wheel_r = [(sensor_l * self.motor_gain_ll) + (sensor_r * self.motor_gain_rl) + self.bias_l/80,
                                      (sensor_r * self.motor_gain_rr) + (sensor_l * self.motor_gain_lr) + self.bias_r/80]
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


class Light(pygame.sprite.Sprite):

    def __init__(self, pos):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.image.load('images/light.png')
        self.image = pygame.transform.rotozoom(self.image, 0, 0.5)
        self.rect = self.image.get_rect()
        self.pos = pos
        self.rect.center = self.pos
