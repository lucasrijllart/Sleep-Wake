import pygame, math, random


class Vehicle(pygame.sprite.Sprite):

    def __init__(self, surface: pygame.Surface):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.image.load("simple1.png")
        self.original = self.image
        self.rect = self.image.get_rect()
        self.rect.x = 200
        self.rect.y = 200
        self.area = surface.get_rect()
        self.angle = 0
        self.direction = pygame.math.Vector2(self.rect.center, self.rect.top)

        pygame.draw.circle(self.image, [0, 255, 0], [27, 0], 3)
        # pygame.draw.circle(self.image, [0, 255, 0], [50, 27], 3)
        # pygame.draw.circle(self.image, [0, 255, 0], [0, 27], 3)
        # pygame.draw.circle(self.image, [255, 0, 0], [25, 25], 26, 1)

        self.T = 10
        self.dt = 0.05

        # wheel left and wheel right
        self.wl, self.wr = 0, 0
        # radius
        self.R = 0.05

    def find_next_move(self):
        self.direction.x = round((-math.sin(math.radians(self.angle))) * 5, 3)
        self.direction.y = round((-math.cos(math.radians(self.angle))) * 5, 3)

    def update(self):

        if True:
            self.move()

    def move(self):


        # velocity
        vc = (self.wl + self.wr) / 2

        va = (self.wr - self.wl) / (2 * self.R)




    def _move(self):
        self.find_next_move()
        newpos = self.rect.move(self.direction)

        # frontal collision
        nose = [self.rect.centerx + self.direction.x * 5.4, self.rect.centery + self.direction.y * 5.4]
        if nose[0] < self.area.left or nose[0] > self.area.right\
                or nose[1] < self.area.top or nose[1] > self.area.bottom:  # left side
            #print('Collision:', self.rect.topleft)

            randomturn = random.randint(20, 40)
            for x in range(0, randomturn):
                self._leftwheel()
        else:
            self.rect = newpos

    def _leftwheel(self):
        self.angle += -6
        old_center = self.rect.center
        self.image = pygame.transform.rotozoom(self.original, self.angle, 1)
        self.direction.rotate(self.angle)
        self.rect = self.image.get_rect()
        self.rect.center = old_center

    def _rightwheel(self):
        self.angle += 6
        old_center = self.rect.center
        self.image = pygame.transform.rotozoom(self.original, self.angle, 1)
        self.direction.rotate(self.angle)
        self.rect = self.image.get_rect()
        self.rect.center = old_center

    def _bothwheels(self):
        self.find_next_move()
        self.rect.x += self.direction.x
        self.rect.y += self.direction.y
