from Simulator import Simulator
from Cycle import Cycles

# Cycles
light_pos = [1100, 600]

# Train Network
type_of_net = 'skmlp'# 'pyrenn'  # 'skmlp' or 'pyrenn'
learning_runs = 200
learning_time = 100
layers = [4, 20, 20, 2]  # [input, layer1, layer2, output] don't change in/out
tap_delay = 30 #for the narxMLP errot graph should b set to the same as the training
max_epochs = 200
use_mean = False # get over-writen from the drop column functionality
train_seed = 1 # if we need two nets with same training samples(vehices)


# Error graph
testing_time = 200  # has to me predict_after + delays + 1
predict_after = 40
brain = [-1, 10, -1, 10, 10, 10]

# Wake learning (can be less than delay)
initial_random_movement = 40

# Sleep
look_ahead = 125 # this is the same look ahead for the sleep_wake phase
individuals = 25
generations = 10

# Wake testing
wake_test_iter = 200

# Booleans for running
train_network = False
error_graph = True
test_network = False

# Cycle running
run_cycles = False

sleep_wake = False
cycles = 3



# Functions
if not train_network:
    cycle = Cycles(light_pos, type=type_of_net, net_filename='narx/r200t100d30e200')
else:
    cycle = Cycles(light_pos)


if train_network:
    cycle.train_network(type_of_net, learning_runs, learning_time, layers, tap_delay, max_epochs, use_mean, train_seed,
                        graphics=False)

cycle.show_error_graph(testing_time, predict_after, brain=None, seed=2.5, graphics=True) if error_graph is True else None
cycle.show_error_graph(testing_time, predict_after, brain=None, seed=None, graphics=True) if error_graph is True else None


cycle.test_network() if test_network is True else None

if run_cycles:
    cycle.wake_learning(initial_random_movement)

    cycle.sleep(look_ahead, individuals, generations)

    cycle.wake_testing(wake_test_iter)

if sleep_wake:
    cycle.run_cycles(initial_random_movement, cycles, individuals=individuals, generations=generations)
# TODO: Find the optimal number of randomGA eval. positions
# TODO: Benchmark for the final brain product.
# TODO: Examine the possibility to increase the influence of the new brains as we add more of them
# TODO: and decrease the influence of the old ones.
# TODO: The network training is insuficient and may need to train a network for longer with more data.