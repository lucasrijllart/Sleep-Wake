import Cycle as Cycle
from Cycle import Cycles

# Train Network
learning_runs = 50
learning_time = 100
layers = [4, 20, 20, 2]  # [input, layer1, layer2, output] don't change in/out
tap_delay = 10
max_epochs = 300

# Error graph
testing_time = 30  # has to me predict_after + delays + 1
predict_after = 10  # has to be more than delay

# Wake learning has to be more than delay
initial_random_movement = 30

# Sleep
look_ahead = 30
individuals = 30
generations = 30

# Wake testing
wake_test_iter = 200

use_narx = True
# Booleans for running
train_network = False
error_graph = False

run_cycles = True
benchmark = True

# Functions
if not train_network:
    cycle = Cycles(net_filename='narx/r50t100d10e300')
else:
    cycle = Cycles()


if train_network:
    cycle.train_network(learning_runs, learning_time, layers, tap_delay, max_epochs, use_mean=True)

if error_graph:
    brain = [-1, 10, -1, 10, 10, 10]
    cycle.show_error_graph(testing_time=testing_time, predict_after=predict_after, brain=None, use_narx=use_narx)


if run_cycles:
    cycle.wake_learning(initial_random_movement)

    cycle.sleep(look_ahead, individuals, generations, use_narx)

    cycle.wake_testing(wake_test_iter, benchmark)
