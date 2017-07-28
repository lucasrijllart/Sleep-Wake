import Cycle as Cycle
from Cycle import Cycles
from Genetic import GA

# Train Network
learning_runs = 50
learning_time = 100
tap_delay = 10
max_epochs = 300

# Error graph
testing_time = 30  # has to me predict_after + delays + 1
predict_after = 10  # has to be more than delay

# Wake learning has to be more than delay
initial_random_movement = 50

# Sleep
look_ahead = 50
individuals = 40
generations = 20

# Wake testing
wake_test_iter = 150

use_narx = True
# Booleans for running
train_network = False
error_graph = False

run_cycles = False
benchmark = False

# Functions
if not train_network:
    cycle = Cycles(net_filename='narx/r50t100d10e300')
else:
    cycle = Cycles()


if train_network:
    cycle.train_network(learning_runs, learning_time, tap_delay, max_epochs, use_mean=True)

if error_graph:
    brain = [-1, 10, -1, 10, 10, 10]
    cycle.show_error_graph(testing_time=testing_time, predict_after=predict_after, brain=None, use_narx=use_narx)


if run_cycles:
    cycle.wake_learning(initial_random_movement)

    cycle.sleep(look_ahead, individuals, generations, use_narx)

    cycle.wake_testing(wake_test_iter)

if benchmark:
    Cycle.random_brain_benchmark()