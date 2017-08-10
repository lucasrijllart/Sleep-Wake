from Simulator import Simulator
from Cycle import Cycles

# Cycles
light_pos = [1100, 600]

# Train Network
type_of_net = 'pyrenn'  # 'skmlp' or 'pyrenn'
learning_runs = 10
learning_time = 150
layers = [4, 20, 20, 2]  # [input, layer1, layer2, output] don't change in/out
tap_delay = 40
max_epochs = 30
use_mean = False
train_seed = None


# Error graph
testing_time = 200  # has to me predict_after + delays + 1
predict_after = 40
brain = [-1, 10, -1, 10, 10, 10]

# Wake learning (can be less than delay)
initial_random_movement = 40

# Sleep
look_ahead = 100 # this is the same look ahead for the sleep_wake phase
individuals = 10
generations = 20

# Wake testing
wake_test_iter = 200

# Booleans for running
train_network = False
error_graph = False
test_network = False

# Cycle running
run_cycles = False

sleep_wake = True
cycles = 2



# Functions
if not train_network:
    cycle = Cycles(light_pos, net_filename='narx/r2t2500d40e20')
else:
    cycle = Cycles(light_pos)


if train_network:
    cycle.train_network(type_of_net, learning_runs, learning_time, layers, tap_delay, max_epochs, use_mean, train_seed,
                        graphics=True)

cycle.show_error_graph(testing_time, predict_after, brain=None, seed=None, graphics=True) if error_graph is True else None
cycle.show_error_graph(testing_time, predict_after, brain=None, seed=2.5, graphics=True) if error_graph is True else None

cycle.test_network() if test_network is True else None

if run_cycles:
    cycle.wake_learning(initial_random_movement)

    cycle.sleep(look_ahead, individuals, generations)

    cycle.wake_testing(wake_test_iter)

if sleep_wake:
    cycle.sleep_wake(initial_random_movement, cycles, look_ahead, individuals, generations)
# TODO: The random collected data should be slice into training data(The first part)
# TODO: and then fitness_eval data the rest. The last position will be the one considered for the fitness
# TODO: eval. as well.
# TODO: The network training is insuficient and may need to train a network for longer with more data.