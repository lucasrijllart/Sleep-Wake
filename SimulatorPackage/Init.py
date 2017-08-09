from Simulator import Simulator
from Cycle import Cycles

# Cycles
light_pos = [Simulator.window_width/2, Simulator.window_height/2]

# Train Network
type_of_net = 'pyrenn'  # 'skmlp' or 'pyrenn'
learning_runs = 200
learning_time = 100
layers = [4, 20, 20, 2]  # [input, layer1, layer2, output] don't change in/out
tap_delay = 30
max_epochs = 1
use_mean = False
train_seed = None


# Error graph
testing_time = 100  # has to me predict_after + delays + 1
predict_after = 20
brain = [-1, 10, -1, 10, 10, 10]

# Wake learning (can be less than delay)
initial_random_movement = 40

# Sleep
look_ahead = 60  # this is the same look ahead for the sleep_wake phase
individuals = 40
generations = 30

# Wake testing
wake_test_iter = 200

# Booleans for running
train_network = True
error_graph = True
test_network = True

# Cycle running
run_one_cycle = True
run_cycles = False

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
