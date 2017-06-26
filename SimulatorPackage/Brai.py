import pygame
import math


class Vehicle(pygame.sprite.Sprite):

    def __init__(self, surface: pygame.Surface, pos, angle):
        # pygame init
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.image.load("simple1.png")  # image of vehicle
        self.original = self.image  # original image to use when rotating
        self.rect = self.image.get_rect()  # rectangle bounds of image
        self.rect = pos  # set bounds as vehicle starting position
        self.area = surface.get_rect()  # screen boundary for vehicles
        self.angle = angle  # starting angle

        # vehicle logic init
        self.dt = 0.05
        self.wl, self.wr = 0, 0  # velocity for left and right wheels
        self.R = 0.05  # radius (ask chris of what)
        self.pos = [pos]  # xy position of vehicle
        self.bearing = [angle / 360 * 2 * math.pi]  # angle of vehicle (converted to rad)
        self.b = self.angle / 360 * 2 * math.pi  # original angle of vehicle (rad)
        self.sensor_gain = 1  # amplify sensor signal
        self.motor_gain = 1  # amplify motor signal

        # vehicle sensory and motor information
        self.sensors = []
        self.motors = []

    def update(self, t):
        """ FIXED: Something is wrong, the position is a bunch of 0's """
        print('t:', t)

        # velocity
        vc = (self.wl + self.wr) / 2
        va = (self.wr - self.wl) / (2 * self.R)

        # print('pos-1: ', self.pos[-1])
        self.pos.append([self.pos[-1][0] + self.R * math.cos(self.bearing[t - 1]),
                         self.pos[-1][1] + self.R * math.sin(self.bearing[t - 1])])  # update position
        self.bearing.append(math.fmod(self.bearing[t - 1] + self.dt * va, 2 * math.pi))  # update bearing
        # print('pos: ', self.pos[t])

        # calculate left sensor position
        sl0, sl1 = [self.pos[t][0] + self.R * math.cos(self.bearing[t] + self.b),
                    self.pos[t][1] + self.R * math.sin(self.bearing[t] + self.b)]

        # calculate right sensor position
        sr0, sr1 = [self.pos[t][0] + self.R * math.cos(self.bearing[t] - self.b),
                    self.pos[t][1] + self.R * math.sin(self.bearing[t] - self.b)]

        # calculate square distance to element
        distance_l, distance_r = [math.sqrt(sl0 ** 2 + sl1 ** 2), math.sqrt(sr0 ** 2 + sr1 ** 2)]

        # calculate sensor intensity
        sensor_l, sensor_r = [self.sensor_gain / distance_l, self.sensor_gain / distance_r]
        self.sensors.append([sensor_l, sensor_r])

        # calculate motor intensity
        motor_l, motor_r = [sensor_l * self.motor_gain, sensor_r * self.motor_gain]
        self.motors.append([motor_l, motor_r])
