from Cycle import Cycles

# Cycles
light_pos = [1100, 600]

# Train Network
type_of_net = 'pyrenn'  # 'skmlp' or 'pyrenn'
learning_runs = 100
learning_time = 100
layers = [4, 20, 40, 20, 2]  # [input, layer1, layer2, output] don't change in/out
tap_delay = 10
max_epochs = 10
use_mean = False
seed = 1
backward_chance = 0

# Error graph
testing_time = 200  # has to me predict_after + delays + 1
predict_after = 50

# Wake learning (can be less than delay)
initial_random_movement = 25

# Sleep
look_ahead = 60
individuals = 30
generations = 20

# Wake testing
wake_test_iter = 200

# Booleans for running
train_network = True
error_graph = True

run_cycles = False


# Functions
if not train_network:
    cycle = Cycles(light_pos, net_filename='narx/r500t100d50e1000')
else:
    cycle = Cycles(light_pos)


if train_network:
    cycle.train_network(type_of_net, learning_runs, learning_time, layers, tap_delay, max_epochs, use_mean, seed,
                        backward_chance, graphics=False)

if error_graph:
    brain = [-1, 10, -1, 10, 5, 5]
    cycle.show_error_graph(testing_time, predict_after, brain=None)


if run_cycles:
    cycle.wake_learning(initial_random_movement)

    cycle.sleep(look_ahead, individuals, generations)

    cycle.wake_testing(wake_test_iter)
