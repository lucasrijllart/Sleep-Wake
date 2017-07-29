from Cycle import Cycles

# Train Network
learning_runs = 100
learning_time = 100
layers = [4, 20, 40, 20, 2]  # [input, layer1, layer2, output] don't change in/out
tap_delay = 20
max_epochs = 201
type_of_net = 'pyrenn'  # 'skmlp' or 'pyrenn'

# Error graph
testing_time = 500  # has to me predict_after + delays + 1
predict_after = 50

# Wake learning has to be more than delay
initial_random_movement = 50

# Sleep
look_ahead = 50
individuals = 30
generations = 20

# Wake testing
wake_test_iter = 200

# Booleans for running
train_network = False
error_graph = False

run_cycles = True

# Functions
if not train_network:
    cycle = Cycles(net_filename='narx/r500t100d50e1000')
else:
    cycle = Cycles()


if train_network:
    cycle.train_network(type_of_net, learning_runs, learning_time, layers=layers, delay=tap_delay,
                        max_epochs=max_epochs, use_mean=False)

if error_graph:
    brain = [-1, 10, -1, 10, 5, 5]
    cycle.show_error_graph(testing_time=testing_time, predict_after=predict_after, brain=None)


if run_cycles:
    cycle.wake_learning(initial_random_movement)

    cycle.sleep(look_ahead, individuals, generations)

    cycle.wake_testing(wake_test_iter)
