import pygame
import math
import random
import numpy as np

# dt constant for all sprites
dt = 2


def get_sensors(v, time, light_pos):
    # calculate left sensor position
    sl0 = (v.pos[time][0] - math.cos(v.bearing[time]) * v.radius) - math.sin(v.bearing[time]) * v.radius
    sl1 = (v.pos[time][1] + math.sin(v.bearing[time]) * v.radius) - math.cos(v.bearing[time]) * v.radius
    # calculate right sensor position
    sr0 = (v.pos[time][0] + math.cos(v.bearing[time]) * v.radius) - math.sin(v.bearing[time]) * v.radius
    sr1 = (v.pos[time][1] - math.sin(v.bearing[time]) * v.radius) - math.cos(v.bearing[time]) * v.radius
    # calculate square distance to light
    distance_l = math.sqrt((light_pos[0] - sl0) ** 2 + (light_pos[1] - sl1) ** 2)
    distance_r = math.sqrt((light_pos[0] - sr0) ** 2 + (light_pos[1] - sr1) ** 2)
    # calculate light intensity based on position using the formula 10/(d + 1)^0.5 (max light 10)
    sensor_l = (10 / (distance_l + 1) ** 0.5) ** 2
    sensor_r = (10 / (distance_r + 1) ** 0.5) ** 2
    v.sensor_left.append(sensor_l)
    v.sensor_right.append(sensor_r)


def update_position(vehicle, time):
    vc = (vehicle.wheel_l + vehicle.wheel_r) / 2  # velocity center
    va = (vehicle.wheel_r - vehicle.wheel_l) / (2 * vehicle.radius)  # velocity average

    # changed top to sin and bottom to cos and it worked
    vehicle.pos.append([vehicle.pos[-1][0] - dt * vc * math.sin(vehicle.bearing[time - 1]),
                        vehicle.pos[-1][1] - dt * vc * math.cos(vehicle.bearing[time - 1])])  # update position
    vehicle.bearing.append(math.fmod(vehicle.bearing[time - 1] + dt * va, 2 * math.pi))  # update bearing
    vehicle.angle = vehicle.bearing[-1] * (180 / math.pi)


def update_graphics(vehicle):
    previous_center = vehicle.pos[-1]
    degree = vehicle.bearing[-1] * 180 / math.pi
    vehicle.image = pygame.transform.rotozoom(vehicle.original, degree, 0.5)
    vehicle.rect = vehicle.image.get_rect()
    vehicle.rect.center = previous_center


class ControllableVehicle(pygame.sprite.Sprite):
    # PyGame constants
    image = pygame.image.load('images/vehicle.png')  # image of vehicle
    radius = 25  # radius of vehicle size

    def __init__(self, start_pos, start_angle):
        # PyGame init
        pygame.sprite.Sprite.__init__(self)
        self.original = self.image  # original image to use when rotating
        self.rect = self.image.get_rect()  # rectangle bounds of image
        self.rect.center = start_pos  # set bounds as vehicle starting position
        self.angle = start_angle  # starting angle
        self.image = pygame.transform.rotozoom(self.original, self.angle, 0.5)

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
        self.random_movement = []

        # weights sensor->motor (lr = left sensor to right wheel)
        self.w_ll, self.w_lr, self.w_rr, self.w_rl = 0, 0, 0, 0
        self.bias_l, self.bias_r = 0, 0

    def set_wheels(self, wheel_data):
        self.wheel_data = np.copy(wheel_data).tolist()

    def set_values(self, ll, lr, rr, rl, bl=0, br=0):
        self.w_ll = ll
        self.w_lr = lr
        self.w_rr = rr
        self.w_rl = rl
        self.bias_l = bl
        self.bias_r = br

    def update(self, t, light):
        # update position
        update_position(self, t)

        # calculate sensor intensity
        get_sensors(self, t, light.pos)

        # get motor intensity
        self.wheel_l, self.wheel_r = self.wheel_data.pop(0)
        self.motor_left.append(self.wheel_l)
        self.motor_right.append(self.wheel_r)

        # update graphics
        update_graphics(self)


class RandomMotorVehicle(pygame.sprite.Sprite):
    # PyGame constants
    image = pygame.image.load('images/vehicle.png')  # image of vehicle
    radius = 25  # radius of vehicle size

    def __init__(self, start_pos, start_angle, gamma, seed):
        # PyGame init
        pygame.sprite.Sprite.__init__(self)
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
        self.pos = [start_pos]  # xy position of vehicle
        self.bearing = [float(start_angle * math.pi / 180)]  # angle of vehicle (converted to rad)

        # vehicle sensory and motor information to extract for neural network
        self.sensor_left = []
        self.sensor_right = []
        self.motor_left = []
        self.motor_right = []

    def update(self, t, light):
        # update position
        update_position(self, t)

        # calculate sensor intensity
        get_sensors(self, t, light.pos)

        # calculate motor intensity
        self.wheel_l, self.wheel_r = [self.wheel_l + self.gamma * (-self.wheel_l + random.normalvariate(2, 4)) + 0.5,
                                      self.wheel_r + self.gamma * (-self.wheel_r + random.normalvariate(2, 4)) + 0.5]
        self.motor_left.append(self.wheel_l)
        self.motor_right.append(self.wheel_r)

        # update graphics
        update_graphics(self)


class BrainVehicle(pygame.sprite.Sprite):
    # PyGame constants
    image = pygame.image.load('images/attacker.png')  # image of vehicle
    radius = 25  # radius of vehicle size

    # Brain constant
    bias_constant = 10

    def __init__(self, start_pos, start_angle):
        # PyGame init
        pygame.sprite.Sprite.__init__(self)
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
        self.random_movement = []  # keeps track of the movement before the brain
        self.predicted_movement = []  # keeps track of the predicted movement by the GA

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
        # update vehicle
        update_position(self, t)

        # calculate sensor intensity
        get_sensors(self, t, light.pos)
        sensor_l = self.sensor_left[-1]
        sensor_r = self.sensor_right[-1]

        # calculate motor intensity
        self.wheel_l, self.wheel_r = [
            (sensor_l * self.w_ll) + (sensor_r * self.w_rl) + self.bias_l / self.bias_constant,
            (sensor_r * self.w_rr) + (sensor_l * self.w_lr) + self.bias_r / self.bias_constant]
        self.motor_left.append(self.wheel_l)
        self.motor_right.append(self.wheel_r)

        # update graphics
        update_graphics(self)


class Light(pygame.sprite.Sprite):
    # PyGame constants
    image = pygame.image.load('images/light.png')

    def __init__(self, pos):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.transform.rotozoom(self.image, 0, 0.5)
        self.rect = self.image.get_rect()
        self.pos = pos
        self.rect.center = self.pos
