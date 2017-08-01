from Simulator import Simulator
from Cycle import Cycles

# Cycles
light_pos = [Simulator.window_width/2, Simulator.window_height/2]  # usually [900,600]

# Train Network
type_of_net = 'pyrenn'  # 'skmlp' or 'pyrenn'
learning_runs = 1
learning_time = 1500
layers = [4, 20, 40, 20, 2]  # [input, layer1, layer2, output] don't change in/out
tap_delay = 40
max_epochs = 20
use_mean = False
train_seed = None
backward_chance = 0.5

# Error graph
testing_time = 200  # has to me predict_after + delays + 1
predict_after = 40
brain = [-1, 10, -1, 10, 5, 5]

# Wake learning (can be less than delay)
initial_random_movement = 40

# Sleep
look_ahead = 40
individuals = 20
generations = 30

# Wake testing
wake_test_iter = 200

# Booleans for running
train_network = False
error_graph = False
test_network = True

run_cycles = True


# Functions
if not train_network:
    cycle = Cycles(light_pos, net_filename='narx/r1t1500d40e50')
else:
    cycle = Cycles(light_pos)


if train_network:
    cycle.train_network(type_of_net, learning_runs, learning_time, layers, tap_delay, max_epochs, use_mean, train_seed,
                        graphics=True)

cycle.show_error_graph(testing_time, predict_after, brain=None, seed=None, graphics=True) if error_graph is True else None

cycle.test_network() if test_network is True else None

if run_cycles:
    cycle.wake_learning(initial_random_movement)

    cycle.sleep(look_ahead, individuals, generations)

    cycle.wake_testing(wake_test_iter)
