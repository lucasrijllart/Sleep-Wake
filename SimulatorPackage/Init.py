from Simulator import Simulator
from Cycle import Cycles

# Cycles
light_pos = [Simulator.window_width/2, Simulator.window_height/2]

# Train Network
type_of_net = 'pyrenn'  # 'skmlp' or 'pyrenn'
learning_runs = 100
learning_time = 100
layers = [4, 20, 40, 20, 2]  # [input, layer1, layer2, output] don't change in/out
tap_delay = 20
max_epochs = 50
use_mean = False
train_seed = 1


# Error graph
testing_time = 200  # has to me predict_after + delays + 1
predict_after = 40
brain = [-1, 10, -1, 10, 10, 10]

# Wake learning (can be less than delay)
initial_random_movement = 40

# Sleep
look_ahead = 80  # this is the same look ahead for the sleep_wake phase
individuals = 20
generations = 20

# Wake testing
wake_test_iter = 40

# Booleans for running
train_network = False
error_graph = False
test_network = False

# Cycle running
run_one_cycle = False
run_cycles = True

sleep_wake = False
cycles = 2



# Functions
if not train_network:
    cycle = Cycles(light_pos, net_filename='narx/r100t100d20e50')
else:
    cycle = Cycles(light_pos)


if train_network:
    cycle.train_network(type_of_net, learning_runs, learning_time, layers, tap_delay, max_epochs, use_mean, train_seed,
                        graphics=False)

cycle.show_error_graph(testing_time, predict_after, brain=None, seed=None, graphics=True) if error_graph is True else None

cycle.test_network() if test_network is True else None

if run_one_cycle:
    cycle.wake_learning(initial_random_movement)

    cycle.sleep(look_ahead, individuals, generations)

    cycle.wake_testing(wake_test_iter)

if run_cycles:
    cycle.wake_learning(initial_random_movement)

    cycle.sleep(look_ahead, individuals, generations)

    cycle.wake_testing(wake_test_iter, benchmark=False)

    cycle.retrain_with_brain()

    cycle.show_error_graph()

    cycle.test_network()

    cycle.assign_testing_as_initial()

    cycle.sleep(100, individuals, generations)

    cycle.wake_testing(200)

if sleep_wake:
    cycle.sleep_wake(initial_random_movement, cycles, look_ahead, individuals, generations)
