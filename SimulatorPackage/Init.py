from Cycle import Cycle

# Train Network
learning_runs = 20
learning_time = 100
input_delay = 10
output_delay = 10
max_epochs = 100

# Wake learning
initial_random_movement = 20

# Error graph
testing_time = 200
predict_after = 40

# Sleep
look_ahead = 20
individuals = 30
generations = 30

# Wake testing
wake_test_iter = 100


# Booleans for running
train_network = False
error_graph = True

run_cycles = False


# Functions
cycle = Cycle(net_filename='narx/r20t100d10e100')

if train_network:
    cycle.train_network(learning_runs, learning_time, input_delay, output_delay, max_epochs)

if error_graph:
    brain = [-1, 10, -1, 10, 10, 10]
    cycle.show_error_graph(testing_time=testing_time, predict_after=predict_after, brain=None)


if run_cycles:
    cycle.wake_learning(initial_random_movement)

    cycle.sleep(look_ahead, individuals, generations)

    cycle.wake_testing(wake_test_iter)
