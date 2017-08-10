import pygame
import math
import random
import numpy as np

# dt constant for all sprites
dt = 2
# size of original image (does not change!)
image_size = 100.0
# radius of vehicles (can change this for size of vehicles)
all_sprites_radius = 25.0
# ratio of image for transformation (diameter / image size)
image_ratio = (all_sprites_radius * 2) / image_size

world_brain = None


def get_sensors(v, time):
    """
    Gets the sensor position based on the vehicle's position and angle, then calculates the distance to the light and
    adds the values to the vehicle's sensor logs
    :param v: vehicle
    :param time: timestep for finding the current bearing
    """
    light_pos = v.light.pos
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
    """
    Updates the position of a vehicle, getting its wheel velocity and angle. Adds the new position, bearing and angle to
    the vehicle logs.
    :param vehicle: vehicle to update
    :param time: timestep for bearing
    """
    vc = (vehicle.wheel_l + vehicle.wheel_r) / 2  # velocity center
    va = (vehicle.wheel_r - vehicle.wheel_l) / (2 * vehicle.radius)  # velocity average

    # changed top to sin and bottom to cos and it worked
    vehicle.pos.append([vehicle.pos[-1][0] - dt * vc * math.sin(vehicle.bearing[time - 1]),
                        vehicle.pos[-1][1] - dt * vc * math.cos(vehicle.bearing[time - 1])])  # update position
    vehicle.bearing.append(math.fmod(vehicle.bearing[time - 1] + dt * va, 2 * math.pi))  # update bearing
    vehicle.angle = vehicle.bearing[-1] * (180 / math.pi)


def update_graphics(vehicle):
    """
    Rotates the image of the vehicle accordingly.
    :param vehicle: vehicle to rotate
    """
    previous_center = vehicle.pos[-1]
    degree = vehicle.bearing[-1] * 180 / math.pi
    vehicle.image = pygame.transform.rotozoom(vehicle.original, degree, image_ratio)
    vehicle.rect = vehicle.image.get_rect()
    vehicle.rect.center = previous_center


def run_through_brain(prediction, ind):
    """
    Gets the wheel velocity by feeding the predictions through the brain and returning the output.
    :param prediction: array of 2 sensor values (numpy)
    :param ind: array of the brain
    :return: wheel velocity left and right
    """
    if world_brain is None:
        local_world_brain = ind
    else:
        local_world_brain = world_brain
    if len(ind) == 6:
        wheel_l, wheel_r = [(prediction[0] * local_world_brain[0]) + (prediction[1] * local_world_brain[3]) + local_world_brain[4] / BrainVehicle.bias_constant,
                            (prediction[1] * local_world_brain[2]) + (prediction[0] * local_world_brain[1]) + local_world_brain[5] / BrainVehicle.bias_constant]
        wheel_l += (prediction[0] * ind[0]) + (prediction[1] * ind[3]) + ind[4] / BrainVehicle.bias_constant
        wheel_r += (prediction[1] * ind[2]) + (prediction[0] * ind[1]) + ind[5] / BrainVehicle.bias_constant
    else:
        wheel_l, wheel_r = [(prediction[0] * local_world_brain[0]) + (prediction[1] * local_world_brain[3]),
                            (prediction[1] * local_world_brain[2]) + (prediction[0] * local_world_brain[1])]
        wheel_l += (prediction[0] * ind[0]) + (prediction[1] * ind[3])
        wheel_r += (prediction[1] * ind[2]) + (prediction[0] * ind[1])
    if world_brain is not None:
        wheel_l /= 2
        wheel_r /= 2
    return wheel_l, wheel_r


class ControllableVehicle(pygame.sprite.Sprite):
    """
    The controllable vehicle is a vehicle that gets passed a wheel log and acts what the wheels dictate. This is used
    to print on the screen the predicted trajectory of the vehicle.
    """
    # PyGame constants
    image = pygame.image.load('images/vehicle.png')  # image of vehicle
    radius = all_sprites_radius  # radius of vehicle size

    def __init__(self, start_pos, start_angle, light):
        """
        Constructor for Controllable Vehicle.
        :param start_pos: the vehicle's starting position (array of 2 int)
        :param start_angle: the vehicle's starting angle
        :param light: the light present in the environment
        """
        # PyGame init
        pygame.sprite.Sprite.__init__(self)
        self.original = self.image  # original image to use when rotating
        self.rect = self.image.get_rect()  # rectangle bounds of image
        self.rect.center = start_pos  # set bounds as vehicle starting position
        self.angle = start_angle  # starting angle
        self.image = pygame.transform.rotozoom(self.original, self.angle, 0.5)

        # vehicle logic init
        self.dt = dt
        self.wheel_l, self.wheel_r = 0, 0
        self.wheel_data = []
        self.pos = [start_pos]  # xy position of vehicle
        self.bearing = [float(start_angle * math.pi / 180)]  # angle of vehicle (converted to rad)
        self.random_movement = []  # array that holds the previous random movement to draw it on the screen
        self.light = light

        # weights sensor->motor (lr = left sensor to right wheel)
        self.w_ll, self.w_lr, self.w_rr, self.w_rl = 0, 0, 0, 0
        self.bias_l, self.bias_r = 0, 0

        # vehicle sensory and motor information to extract for neural network
        self.sensor_left = []
        self.sensor_right = []
        self.motor_left = [0.0]
        self.motor_right = [0.0]
        get_sensors(self, 0)

    def set_wheels(self, wheel_data):
        self.wheel_data = np.copy(wheel_data).tolist()

    def set_values(self, ll, lr, rr, rl, bl=0, br=0):
        self.w_ll = ll
        self.w_lr = lr
        self.w_rr = rr
        self.w_rl = rl
        self.bias_l = bl
        self.bias_r = br

    def update(self, t):
        # update position
        update_position(self, t)

        # calculate sensor intensity
        get_sensors(self, t)

        # get motor intensity
        wheel_l, wheel_r = self.wheel_data.pop(0)
        self.wheel_l = wheel_l[0]
        self.wheel_r = wheel_r[0]
        self.motor_left.append(self.wheel_l)
        self.motor_right.append(self.wheel_r)

        # update graphics
        update_graphics(self)


class RandomMotorVehicle(pygame.sprite.Sprite):
    """
    The random motor vehicle is used to go around randomly in the environment. From the collection of sensory and motor
    values, we then use those to train a network.
    """
    # PyGame constants
    image = pygame.image.load('images/vehicle.png')  # image of vehicle
    radius = all_sprites_radius

    min_start_iter = 40
    max_start_iter = 80
    min_stop_iter = 5
    max_stop_iter = 10

    def __init__(self, start_pos, start_angle, gamma, light):
        forward = True
        # PyGame init
        pygame.sprite.Sprite.__init__(self)
        self.original = self.image  # original image to use when rotating
        self.rect = self.image.get_rect()  # rectangle bounds of image
        self.rect.center = start_pos  # set bounds as vehicle starting position
        self.angle = start_angle  # starting angle
        self.image = pygame.transform.rotozoom(self.original, self.angle, 0.5)

        # vehicle logic init
        self.light = light
        self.gamma = gamma
        self.dt = dt
        if forward is True:
            self.mean = 2
            self.bias = 0.2
        else:
            self.mean = -2
            self.bias = -0.2
        # velocity for left and right wheels
        self.wheel_l, self.wheel_r = 0, 0
        self.pos = [start_pos]  # xy position of vehicle
        self.bearing = [float(start_angle * math.pi / 180)]  # angle of vehicle (converted to rad)
        self.previous_pos = None

        # vehicle sensory and motor information to extract for neural network
        self.sensor_left = []
        self.sensor_right = []
        self.motor_left = [0.0]
        self.motor_right = [0.0]
        get_sensors(v=self, time=0)

    def update(self, t):
        # update position
        update_position(self, t)

        # calculate sensor intensity
        get_sensors(self, t)

        # calculate motor intensity
        if random.random() < 0.5:
            self.wheel_l = self.wheel_l + self.gamma * (-self.wheel_l + random.normalvariate(self.mean, 4)) + self.bias
            self.wheel_r = self.wheel_r + self.gamma * (-self.wheel_r + random.normalvariate(self.mean, 4)) + self.bias

        # if there is a brain, pass through brain
        if world_brain is not None:
            sensor_l = self.sensor_left[-1]
            sensor_r = self.sensor_right[-1]
            wheels = run_through_brain([sensor_l, sensor_r], world_brain)
            w_l_2 = wheels[0]
            w_r_2 = wheels[1]
            w_l_1 = w_l_2 + self.gamma * (-w_l_2 + random.normalvariate(self.mean, 4)) + self.bias
            w_r_1 = w_r_2 + self.gamma * (-w_r_2 + random.normalvariate(self.mean, 4)) + self.bias
            self.wheel_l = (w_l_1 + w_l_2)/2
            self.wheel_r = (w_r_1 + w_r_2)/2

        self.motor_left.append(self.wheel_l)
        self.motor_right.append(self.wheel_r)

        # update graphics
        update_graphics(self)


class BrainVehicle(pygame.sprite.Sprite):
    """
    The brain vehicle moves by getting its sensory information, passing it through its brain to get the wheel velocity,
    then update the position with those.
    """
    # PyGame constants
    image = pygame.image.load('images/attacker.png')  # image of vehicle
    radius = all_sprites_radius  # radius of vehicle size

    # Brain constant
    bias_constant = 10

    def __init__(self, start_pos, start_angle, light):
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
        self.w_ll, self.w_lr, self.w_rr, self.w_rl = None, None, None, None
        self.bias_l, self.bias_r = None, None  # automatically added wheel bias to wheels
        self.random_movement = []  # keeps track of the movement before the brain
        self.predicted_movement = []  # keeps track of the predicted movement by the GA
        self.light = light

        # vehicle sensory and motor information to extract for neural network
        self.sensor_left = []
        self.sensor_right = []
        self.motor_left = [0.0]
        self.motor_right = [0.0]
        get_sensors(self, 0)

    def set_values(self, ll_lr_rr_rl_bl_br):
        if len(ll_lr_rr_rl_bl_br) >= 4:
            self.w_ll = ll_lr_rr_rl_bl_br[0]
            self.w_lr = ll_lr_rr_rl_bl_br[1]
            self.w_rr = ll_lr_rr_rl_bl_br[2]
            self.w_rl = ll_lr_rr_rl_bl_br[3]
            if len(ll_lr_rr_rl_bl_br) > 4:
                self.bias_l = ll_lr_rr_rl_bl_br[4]
                self.bias_r = ll_lr_rr_rl_bl_br[5]

    def get_brain(self):  # returns the brain with 4 or 6 genes
        if self.bias_l is None and self.bias_r is None:
            return [self.w_ll, self.w_lr, self.w_rr, self.w_rl]
        else:
            return [self.w_ll, self.w_lr, self.w_rr, self.w_rl, self.bias_l, self.bias_r]

    def update(self, t):
        # update vehicle
        update_position(self, t)

        # calculate sensor intensity
        get_sensors(self, t)
        sensor_l = self.sensor_left[-1]
        sensor_r = self.sensor_right[-1]

        # calculate motor intensity
        wheels = run_through_brain([sensor_l, sensor_r], self.get_brain())
        self.wheel_l = wheels[0]
        self.wheel_r = wheels[1]
        self.motor_left.append(self.wheel_l)
        self.motor_right.append(self.wheel_r)

        # update graphics
        update_graphics(self)


class Light(pygame.sprite.Sprite):
    """
    The Light is a non-moving sprite that only has coordinates for calculating Euclidean distances and prints its image
    on the screen.
    """
    # PyGame constants
    image = pygame.image.load('images/light.png')

    def __init__(self, pos):
        """
        Constructor for the Light, only takes in the position.
        :param pos: The position of the light (array of 2 int [x, y])
        """
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.transform.rotozoom(self.image, 0, 0.5)
        self.rect = self.image.get_rect()
        self.pos = pos
        self.rect.center = self.pos
