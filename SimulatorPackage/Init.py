from Cycle import Cycle

# Train Network
learning_runs = 400
learning_time = 150
tap_delay = 20
max_epochs = 100

# Wake learning
initial_random_movement = 25

# Error graph
testing_time = 200
predict_after = 20

# Sleep
look_ahead = 30
individuals = 20
generations = 50

# Wake testing
wake_test_iter = 250


# Booleans for running
train_network = False
error_graph = True

run_cycles = False

# Functions 'narx/r400t200d20e100_good'
cycle = Cycle(net_filename='narx/r100t150d20e100')

if train_network:
    cycle.train_network(learning_runs, learning_time, tap_delay, max_epochs)

if error_graph:
    brain = [-1, 10, -1, 10, 5, 5]
    cycle.show_error_graph(testing_time=testing_time, predict_after=predict_after, brain=None, use_narx=True)


if run_cycles:
    cycle.wake_learning(initial_random_movement)

    cycle.sleep(look_ahead, individuals, generations)

    cycle.wake_testing(wake_test_iter)
