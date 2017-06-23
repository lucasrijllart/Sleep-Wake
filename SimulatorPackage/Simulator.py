import sys
import pygame
from SimulatorPackage.Vehicle import Vehicle


def run():

    pygame.init()
    screen = pygame.display.set_mode((640, 480))
    pygame.display.set_caption("Braitenberg vehicle simulation")
    for e in pygame.event.get():
        pygame.key.set_repeat(50, 20)
    clock = pygame.time.Clock()

    white = 240, 240, 240
    background = pygame.Surface(screen.get_size())
    background = background.convert(background)
    background.fill(white)

    v1 = Vehicle(screen)
    all_sprites = pygame.sprite.RenderPlain(v1)


    while 1:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                sys.exit()
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                all_sprites.add(Vehicle(screen))
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_LEFT:
                v1._leftwheel()
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_RIGHT:
                v1._rightwheel()
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_UP:
                v1._bothwheels()

        all_sprites.update()
        clock.tick()
        screen.blit(background, (0, 0))
        all_sprites.draw(screen)
        pygame.display.flip()
        pygame.display.set_caption('Braitenberg vehicle simulation - ' + str(format(clock.get_fps(), '.0f')) + 'fps')
