import pygame, math, random


class Vehicle(pygame.sprite.Sprite):

    def __init__(self, surface: pygame.Surface, pos):
        # pygame init
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.image.load("simple1.png")
        self.original = self.image
        self.rect = self.image.get_rect()
        self.rect.x = 200
        self.rect.y = 200
        self.area = surface.get_rect()
        self.angle = 0
        self.direction = pygame.math.Vector2(self.rect.center, self.rect.top)

        # vehicle logic init
        self.T = 10  # time
        self.dt = 0.05

        # wheel left and wheel right
        self.wl, self.wr = 0, 0
        # radius
        self.R = 0.05

        self.pos = []
        self.pos.append(pos)
        self.bearing = []
        self.bearing.append(90 / 360 * 2 * math.pi)  # TODO: change this to angle of vehicle defined before (passed as param)
        self.b = 90
        self.b = self.b / 360 * 2 * math.pi

        self.sensor_gain = 1
        self.motor_gain = 1

    def update(self, t):
        print('t:', t)
        self.main(t)

    def get_pos(self):
        last_pos = self.pos[-1]
        print(self.pos)
        print('last_pos: ', last_pos)
        return self.pos[-1]

    def main(self, t):
        ''' Something is wrong, the position is a bunch of 0's '''
        # velocity
        vc = (self.wl + self.wr) / 2

        va = (self.wr - self.wl) / (2 * self.R)

        print('pos-1: ', self.pos[-1])

        self.pos.append([self.pos[-1][0] + self.R * math.cos(self.bearing[t-1]),
                         self.pos[-1][1] + self.R * math.sin(self.bearing[t-1])])
        self.bearing.append(math.fmod(self.bearing[t-1] + self.dt * va, 2 * math.pi))

        print('pos: ', self.pos[t])  # pos gets written in last pos entry

        # calculate left sensor position
        sl0 = self.pos[t][0] + self.R * math.cos(self.bearing[t] + self.b)
        sl1 = self.pos[t][1] + self.R * math.sin(self.bearing[t] + self.b)

        # calculate right sensor position
        sr0 = self.pos[t][0] + self.R * math.cos(self.bearing[t] - self.b)
        sr1 = self.pos[t][1] + self.R * math.sin(self.bearing[t] - self.b)

        # calculate square distance to element
        dl = math.sqrt(sl0 ** 2 + sl1 ** 2)
        dr = math.sqrt(sr0 ** 2 + sr1 ** 2)

        # calculate local intensity
        il = self.sensor_gain / dl
        ir = self.sensor_gain / dr

        vl = il * self.motor_gain
        rm = ir * self.motor_gain

