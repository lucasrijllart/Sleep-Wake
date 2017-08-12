from Simulator import Simulator
from Cycle import Cycles

# Cycles
light_pos = [Simulator.window_width/2, Simulator.window_height/2]

# Train Network
type_of_net = 'pyrenn'  # 'skmlp' or 'pyrenn'
learning_runs = 50
learning_time = 100
layers = [4, 20, 40, 20, 2]  # [input, layer1, layer2, output] don't change in/out
tap_delay = 20
max_epochs = 20
use_mean = False
train_seed = None


# Error graph
testing_time = 300  # has to me predict_after + delays + 1
predict_after = 40
brain = [-1, 10, -1, 10, 10, 10]

# Wake learning (can be less than delay)
initial_random_movement = 40

# Sleep
look_ahead = 100  # this is the same look ahead for the sleep_wake phase
individuals = 30
generations = 30

# Wake testing
wake_test_iter = 100

# Booleans for running
train_network = False
error_graph = True
test_network = False

# Cycle running
run_one_cycle = True
run_cycles = False
run_cycles_net = False

sleep_wake = False
cycles = 2


# Functions
if not train_network:
    cycle = Cycles(light_pos, net_filename='narx/r200t100d40e300')
else:
    cycle = Cycles(light_pos)

veh_pos, veh_angle = None, None
if train_network:
    cycle.train_network(type_of_net, learning_runs, learning_time, layers, tap_delay, max_epochs, use_mean, train_seed,
                        graphics=False)
    veh_pos = cycle.init_pos
    veh_angle = cycle.init_angle

cycle.show_error_graph(veh_pos, veh_angle, testing_time, predict_after, seed=None, graphics=True) if error_graph is True else None

cycle.test_network(graphics=True) if test_network is True else None

if run_one_cycle:
    cycle.wake_learning(50)

    rand_test = cycle.collect_training_data(True, 5, 50)

    cycle.sleep(look_ahead=look_ahead, individuals=individuals, generations=generations, td=rand_test)

    cycle.wake_testing(cycle.random_vehicle.pos[-1], cycle.random_vehicle.angle, wake_test_iter)

if run_cycles:

    cycle.run_2_cylces(look_ahead, individuals, generations, wake_test_iter)

if run_cycles_net:

    cycle.run_2_cycles_with_net(initial_random_movement, look_ahead, individuals, generations, wake_test_iter)