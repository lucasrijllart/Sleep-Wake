from Simulator import Simulator
from Cycle import Cycles

# Cycles
light_pos = [Simulator.window_width/2, Simulator.window_height/2]

# Train Network
type_of_net = 'pyrenn'  # 'skmlp' or 'pyrenn'
learning_runs = 50
learning_time = 100
layers = [4, 20, 20, 2]  # [input, layer1, layer2, output] don't change in/out
tap_delay = 30
max_epochs = 25
use_mean = False
train_seed = None


# Error graph
testing_time = 100  # has to me predict_after + delays + 1
predict_after = 20
brain = [-1, 10, -1, 10, 10, 10]

# Wake learning (can be less than delay)
initial_random_movement = 40

# Sleep
look_ahead = 80  # this is the same look ahead for the sleep_wake phase
individuals = 30
generations = 30

# Wake testing
wake_test_iter = 300

# Booleans for running
train_network = False
error_graph = True
test_network = True

# Cycle running
run_one_cycle = True
run_cycles = False

sleep_wake = False
cycles = 2



# Functions
if not train_network:
    cycle = Cycles(light_pos, net_filename='narx/r100t100d20e50')
else:
    cycle = Cycles(light_pos)

veh_pos, veh_angle = None, None
if train_network:
    cycle.train_network(type_of_net, learning_runs, learning_time, layers, tap_delay, max_epochs, use_mean, train_seed,
                        graphics=False)
    veh_pos = cycle.init_pos
    veh_angle = cycle.init_angle

cycle.show_error_graph(veh_pos, veh_angle, testing_time, predict_after, seed=None, graphics=False) if error_graph is True else None

cycle.test_network(graphics=False) if test_network is True else None

if run_one_cycle:
    cycle.wake_learning(50)

    rand_test = cycle.collect_training_data(True, 5, 50)

    cycle.sleep(look_ahead=look_ahead, individuals=individuals, generations=generations, td=rand_test)

    cycle.wake_testing(cycle.random_vehicle.pos[-1], cycle.random_vehicle.angle, wake_test_iter)

if run_cycles:

    cycle.run_2_cylces(look_ahead, individuals, generations, wake_test_iter)
