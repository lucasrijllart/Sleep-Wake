from Cycle import Cycle

# Train Network
learning_runs = 2
learning_time = 10
input_delay = 20
output_delay = 20
max_epochs = 100

# Wake learning
initial_random_movement = 25

# Error graph
testing_time = 200
predict_after = 20

# Sleep
look_ahead = 100
individuals = 30
generations = 100

# Wake testing
wake_test_iter = 100


# Booleans for running
train_network = True
error_graph = False

run_cycles = False

# Functions
cycle = Cycle(net_filename='narx/r400t200d20e100_good')

if train_network:
    cycle.train_network(learning_runs, learning_time, input_delay, output_delay, max_epochs)

if error_graph:
    brain = [-1, 10, -1, 10, 10, 10]
    cycle.show_error_graph(testing_time=testing_time, predict_after=predict_after, brain=None)


if run_cycles:
    cycle.wake_learning(initial_random_movement)

    cycle.sleep(look_ahead, individuals, generations)

    cycle.wake_testing(wake_test_iter)
