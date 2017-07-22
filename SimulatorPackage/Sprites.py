import pygame
import math
import random

dt = 2


def get_sensors(pos, time, bearing, radius, light_pos):
    # calculate left sensor position
    sl0 = (pos[time][0] - math.cos(bearing[time]) * radius) - math.sin(bearing[time]) * radius
    sl1 = (pos[time][1] + math.sin(bearing[time]) * radius) - math.cos(bearing[time]) * radius
    # calculate right sensor position
    sr0 = (pos[time][0] + math.cos(bearing[time]) * radius) - math.sin(bearing[time]) * radius
    sr1 = (pos[time][1] - math.sin(bearing[time]) * radius) - math.cos(bearing[time]) * radius
    # calculate square distance to light
    distance_l = math.sqrt((light_pos[0] - sl0) ** 2 + (light_pos[1] - sl1) ** 2)
    distance_r = math.sqrt((light_pos[0] - sr0) ** 2 + (light_pos[1] - sr1) ** 2)
    # calculate light intensity based on position using the formula 10/(d + 1)^0.5 (max light 10)
    sensor_l = 10 / (distance_l + 1) ** 0.5
    sensor_r = 10 / (distance_r + 1) ** 0.5
    return [sensor_l**2, sensor_r**2]


def update_position(wheel_l, wheel_r, radius, pos, bearing, time):
    vc = (wheel_l + wheel_r) / 2  # velocity center
    va = (wheel_r - wheel_l) / (2 * radius)  # velocity average

    # changed top to sin and bottom to cos and it worked
    pos.append([pos[-1][0] - dt * vc * math.sin(bearing[time - 1]),
                pos[-1][1] - dt * vc * math.cos(bearing[time - 1])])  # update position
    bearing.append(math.fmod(bearing[time - 1] + dt * va, 2 * math.pi))  # update bearing
    return [pos, bearing]


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
        self.dt = dt
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

    def set_values(self, ll, lr, rr, rl, bl=0, br=0):
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
        self.pos, self.bearing = update_position(self.wheel_l, self.wheel_r, self.radius, self.pos, self.bearing, t)

        # calculate sensor intensity
        sensor_l, sensor_r = get_sensors(self.pos, t, self.bearing, self.radius, light_pos)
        self.sensor_left.append(sensor_l)
        self.sensor_right.append(sensor_r)

        # get motor intensity
        self.wheel_l, self.wheel_r = self.wheel_data.pop(0)
        self.motor_left.append(self.wheel_l)
        self.motor_right.append(self.wheel_r)

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
        self.dt = dt
        # velocity for left and right wheels
        self.wheel_l, self.wheel_r = random.uniform(-0.05, 0.05), random.uniform(-0.05, 0.05)
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
        self.update_vehicle(t, light.pos)
        self.update_graphics()

    def update_vehicle(self, t, light_pos):
        self.pos, self.bearing = update_position(self.wheel_l, self.wheel_r, self.radius, self.pos, self.bearing, t)

        # calculate sensor intensity
        sensor_l, sensor_r = get_sensors(self.pos, t, self.bearing, self.radius, light_pos)
        self.sensor_left.append(sensor_l)
        self.sensor_right.append(sensor_r)

        # calculate motor intensity
        self.wheel_l, self.wheel_r = [self.wheel_l + self.gamma * (-self.wheel_l + random.normalvariate(2, 4)) + 0.5,
                                      self.wheel_r + self.gamma * (-self.wheel_r + random.normalvariate(2, 4)) + 0.5]
        self.motor_left.append(self.wheel_l)
        self.motor_right.append(self.wheel_r)

    def update_graphics(self):
        previous_center = self.pos[-1]
        degree = self.bearing[-1] * 180 / math.pi
        self.image = pygame.transform.rotozoom(self.original, degree, 0.5)
        self.rect = self.image.get_rect()
        self.rect.center = previous_center


class BrainVehicle(pygame.sprite.Sprite):

    radius = 25  # radius of vehicle size
    bias_constant = 10

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
        self.dt = dt
        self.wheel_l, self.wheel_r = 0, 0  # velocity for left and right wheels

        self.pos = [start_pos]  # xy position of vehicle
        self.bearing = [float(start_angle * math.pi / 180)]  # angle of vehicle (converted to rad)
        self.w_ll, self.w_rl, self.w_rr, self.w_lr = 0, 0, 0, 0
        self.bias_l, self.bias_r = 2, 2  # automatically added wheel bias to wheels
        self.previous_pos = []  # keeps track of the movement before the brain (random movements)

        # vehicle sensory and motor information to extract for neural network
        self.sensor_left = []
        self.sensor_right = []
        self.motor_left = []
        self.motor_right = []

    def set_values(self, ll_lr_rr_rl_bl_br):
        self.w_ll = ll_lr_rr_rl_bl_br[0]
        self.w_lr = ll_lr_rr_rl_bl_br[1]
        self.w_rr = ll_lr_rr_rl_bl_br[2]
        self.w_rl = ll_lr_rr_rl_bl_br[3]
        self.bias_l = ll_lr_rr_rl_bl_br[4]
        self.bias_r = ll_lr_rr_rl_bl_br[5]

    def update(self, t, light):
        self.update_vehicle(t, light.pos)
        self.update_graphics()

    def update_vehicle(self, t, light_pos):
        self.pos, self.bearing = update_position(self.wheel_l, self.wheel_r, self.radius, self.pos, self.bearing, t)

        # calculate sensor intensity
        sensor_l, sensor_r = get_sensors(self.pos, t, self.bearing, self.radius, light_pos)
        self.sensor_left.append(sensor_l)
        self.sensor_right.append(sensor_r)

        # calculate motor intensity
        self.wheel_l, self.wheel_r = [(sensor_l*self.w_ll) + (sensor_r * self.w_rl) + self.bias_l / self.bias_constant,
                                      (sensor_r*self.w_rr) + (sensor_l * self.w_lr) + self.bias_r / self.bias_constant]
        self.motor_left.append(self.wheel_l)
        self.motor_right.append(self.wheel_r)

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
