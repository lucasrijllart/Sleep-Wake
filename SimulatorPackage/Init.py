from Cycle import Cycle

# Train Network
learning_runs = 2
learning_time = 50
tap_delay = 3
max_epochs = 100

# Wake learning has to be more than delay
initial_random_movement = 50

# Error graph
testing_time = 200
predict_after = 20  # has to be more than delay

# Sleep
look_ahead = 50
individuals = 40
generations = 20

# Wake testing
wake_test_iter = 100

use_narx = True
# Booleans for running
train_network = True
error_graph = False

run_cycles = False

# Functions
cycle = Cycle(net_filename='narx/r50t100d10e100')

if train_network:
    cycle.train_network(learning_runs, learning_time, layers=[4, 20, 20, 2], delay=tap_delay,
                        max_epochs=max_epochs, use_mean=False)

if error_graph:
    brain = [-1, 10, -1, 10, 5, 5]
    cycle.show_error_graph(testing_time=testing_time, predict_after=predict_after, brain=None, use_narx=use_narx)


if run_cycles:
    cycle.wake_learning(initial_random_movement)

    cycle.sleep(look_ahead, individuals, generations, use_narx)

    cycle.wake_testing(wake_test_iter)
