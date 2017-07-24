from Cycle import Cycle

# Train Network
learning_runs = 100
learning_time = 200
input_delay = 20
output_delay = 20
max_epochs = 50

# Wake learning
initial_random_movement = 50

# Error graph
testing_time = 100
predict_after = 40

# Sleep
look_ahead = 20
individuals = 30
generations = 10

# Wake testing
wake_test_iter = 200


# Booleans for running
train_network = False
error_graph =   True

wake_learn =    True
sleep =         True
wake_test =     True


# Functions
cycle = Cycle(net_filename='narx/r400t200d20e100_good')

if train_network:
    cycle.train_network(learning_runs, learning_time, input_delay, output_delay, max_epochs)

if error_graph:
    brain = [-1, 10, -1, 10, 10, 10]
    cycle.show_error_graph(testing_time=testing_time, predict_after=predict_after, brain=None)


if wake_learn:
    cycle.wake_learning(initial_random_movement)

if sleep:
    cycle.sleep(look_ahead, individuals, generations)

if wake_test:
    cycle.wake_testing(wake_test_iter)
