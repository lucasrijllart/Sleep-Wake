from Cycle import Cycle

# Train Network
learning_runs = 500
learning_time = 200
tap_delay = 30
max_epochs = 50

# Wake learning has to be more than delay
initial_random_movement = 50

# Error graph
testing_time = 200
predict_after = 30  # has to be more than delay

# Sleep
look_ahead = 50
individuals = 40
generations = 20

# Wake testing
wake_test_iter = 100

use_narx = True
# Booleans for running
train_network = False
error_graph = True

run_cycles = True

# Functions
cycle = Cycle(net_filename='narx/r100t150d20e100')

if train_network:
    cycle.train_network(learning_runs, learning_time, tap_delay, max_epochs, use_mean=True)

if error_graph:
    brain = [-1, 10, -1, 10, 10, 10]
    cycle.show_error_graph(testing_time=testing_time, predict_after=predict_after, brain=None, use_narx=use_narx)


if run_cycles:
    cycle.wake_learning(initial_random_movement)

    cycle.sleep(look_ahead, individuals, generations, use_narx)

    cycle.wake_testing(wake_test_iter)
