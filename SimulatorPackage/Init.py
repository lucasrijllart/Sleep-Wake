from Cycle import Cycles

# Train Network
learning_runs = 100
learning_time = 100
layers = [4, 20, 20, 2]  # [input, layer1, layer2, output] don't change in/out
tap_delay = 10
max_epochs = 300
type_of_net = None #'skmlp'

# Error graph
testing_time = 100  # has to me predict_after + delays + 1
predict_after = 30  # has to be more than delay

# Wake learning has to be more than delay
initial_random_movement = 50

# Sleep
look_ahead = 50
individuals = 20
generations = 100

# Wake testing
wake_test_iter = 150

use_narx = True
# Booleans for running
train_network = False
error_graph = True

run_cycles = False
benchmark = False

# Functions
if not train_network:
    cycle = Cycles(net_filename='narx/r100t150d20e100', type=type_of_net)
else:
    cycle = Cycles()


if train_network:
    cycle.train_network(type_of_net, learning_runs, learning_time, layers=[4, 20, 20, 2], delay=tap_delay,
                        max_epochs=max_epochs, use_mean=False)

if error_graph:
    brain = [-1, 10, -1, 10, 5, 5]
    cycle.show_error_graph(testing_time=testing_time, predict_after=predict_after, brain=None, use_narx=use_narx)


if run_cycles:
    cycle.wake_learning(initial_random_movement)

    cycle.sleep(look_ahead, individuals, generations, use_narx)

    cycle.wake_testing(wake_test_iter, benchmark)

