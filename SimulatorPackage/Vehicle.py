import pygame
import math


class Vehicle(pygame.sprite.Sprite):

    def __init__(self, surface: pygame.Surface, pos, angle):
        # pygame init
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.image.load('images/simple1.png')  # image of vehicle
        self.original = self.image  # original image to use when rotating
        self.rect = self.image.get_rect()  # rectangle bounds of image
        self.rect.center = pos # set bounds as vehicle starting position
        self.area = surface.get_rect()  # screen boundary for vehicles
        self.angle = angle  # starting angle
        self.image = pygame.transform.rotozoom(self.original, self.angle, 0.5)

        # vehicle logic init
        self.dt = 0.05
        self.wheel_l, self.wheel_r = 0, 0  # velocity for left and right wheels
        self.R = 25  # radius (ask chris of what)
        self.pos = [pos]  # xy position of vehicle
        self.bearing = [angle / 360 * 2 * math.pi]  # angle of vehicle (converted to rad)
        self.b = self.angle / 360 * 2 * math.pi  # original angle of vehicle (rad)
        self.sensor_gain = 1  # amplify sensor signal
        self.motor_gain = 1  # amplify motor signal

        # vehicle sensory and motor information
        self.sensors = []
        self.motors = []

    def update(self, t, light_pos):
        self.update_sensors(t, light_pos)
        self.update_graphics()

    def update_sensors(self, t, light_pos):
        print('\nt:', t)

        # velocity
        vc = (self.wheel_l + self.wheel_r) / 2
        va = (self.wheel_r - self.wheel_l) / (2 * self.R)

        # print('pos-1: ', self.pos[-1])
        self.pos.append([self.pos[-1][0] + self.dt * vc * math.cos(self.bearing[t-1]),
                         self.pos[-1][1] + self.dt * vc * math.sin(self.bearing[t-1])])  # update position
        self.bearing.append(math.fmod(self.bearing[t-1] + self.dt * va, 2 * math.pi))  # update bearing
        # print('pos: ', self.pos[t])
        print('bea:', self.bearing[-1])

        print('pos:', self.pos[t])
        # calculate left sensor position
        sl0, sl1 = [self.pos[t][0] - self.R * math.cos(self.bearing[t] + self.b) - self.rect.width / 2,
                    self.pos[t][1] + self.R * math.sin(self.bearing[t] + self.b) - self.rect.height / 2]
        # print('sl0:', sl0, 'sl1:', sl1, ' topl:', self.rect.topleft)

        # calculate right sensor position
        sr0, sr1 = [self.pos[t][0] + self.R * math.cos(self.bearing[t] - self.b),
                    self.pos[t][1] + self.R * math.sin(self.bearing[t] - self.b) - self.rect.height / 2]
        # print('sr0:', sr0, 'sr1:', sr1, 'topr:', self.rect.topright,)

        # calculate square distance to light
        distance_l = math.sqrt((light_pos[0] - sl0) ** 2 + (light_pos[1] - sl1) ** 2)
        distance_r = math.sqrt((light_pos[0] - sr0) ** 2 + (light_pos[1] - sr1) ** 2)
        # print('dl:', distance_l, 'dr:', distance_r)

        # calculate sensor intensity
        sensor_l, sensor_r = [self.sensor_gain / distance_l, self.sensor_gain / distance_r]
        self.sensors.append([sensor_l, sensor_r])

        # calculate motor intensity
        self.wheel_l, self.wheel_r = [sensor_l * self.motor_gain + 2, sensor_r * self.motor_gain]
        self.motors.append([self.wheel_l, self.wheel_r])

    def update_graphics(self):
        previous_center = self.rect.center
        degree = self.bearing[-1] * 180 / math.pi
        self.image = pygame.transform.rotozoom(self.original, degree, 0.5)
        self.rect = self.image.get_rect()
        self.rect.center = previous_center
