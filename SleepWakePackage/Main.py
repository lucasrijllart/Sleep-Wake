from Cycle import Cycles
from Tests import Tests

# Light position
light_pos = [1200, 400]

# Train Network
type_of_net = 'pyrenn'  # 'skmlp' or 'pyrenn'
learning_runs = 50
learning_time = 50
layers = [4, 20, 20, 2]  # [input, layer1, layer2, output] don't change in/out
tap_delay = 20
max_epochs = 300
use_mean = False
train_seed = 5

# Error graph
testing_time = 100  # has to me predict_after + delays + 1
predict_after = 40
brain = [-1, 10, -1, 10, 10, 10]

# -- CYCLES --
# Wake learning (can be less than delay)
initial_random_movement = 40

# Sleep
look_ahead = 60  # this is the same look ahead for the sleep_wake phase
individuals = 30
generations = 10

# Wake testing
wake_test_iter = 80

# -- BOOLEANS USED FOR RUNNING --
train_network = False
error_graph = False
test_network = False

# Cycles
run_one_cycle = True
run_cycles = False
run_cycles_net = False
sleep_wake = False
cycles = 2

# TESTS
# 1. Braitenberg evolution
test1 = False
# 2. Environment model
test2_1 = False
test2_2 = False
test2_3 = False
test2_4 = False
# 3. Control System evolution
test3_1 = False
test3_2 = False

# Functions
if not train_network:
    cycle = Cycles(light_pos, net_filename='narx/r200t100d40e300')
else:
    cycle = Cycles(light_pos)

veh_pos, veh_angle = None, None
if train_network:
    cycle.train_network(type_of_net, learning_runs, learning_time, layers, tap_delay, max_epochs, use_mean, train_seed,
                        graphics=False, allow_back=False, continuous=True)
    veh_pos = cycle.pos_before_collect
    veh_angle = cycle.ang_before_collect

cycle.show_error_graph(veh_pos, veh_angle, testing_time, predict_after, graphics=True) if error_graph is True else None

cycle.test_network(test_time=100, graphics=True, seed=1) if test_network is True else None

if run_one_cycle:
    cycle.wake_learning(50)
    cycle.sleep(look_ahead=look_ahead, individuals=individuals, generations=generations)
    cycle.wake_testing(cycle.random_vehicle.pos[-1], cycle.random_vehicle.angle, wake_test_iter)

if run_cycles:
    cycle.run_2_cycles('narx/r200t100d40e300', initial_random_movement, look_ahead, individuals, generations, wake_test_iter)

# Runs 2 cycles with the first network as the one previously loaded
if run_cycles_net:
    cycle.run_2_cycles_with_net(initial_random_movement, look_ahead, individuals, generations, wake_test_iter)

# Used to run tests
test = Tests()
if test1:
    test.test_1()
if test2_1:
    test.test_2_1('narx/test2/r20t100d10e200', 'narx/test2/r20t100d40e200', 40, 100, 10, 40, 8)
if test2_2:
    test.test_2_2('narx/r200t100d40e300', 50)
if test2_3:
    test.test_2_3(num_of_tests=100)
if test2_4:
    test.test_2_4()
if test3_1:
    test.test_3_1('narx/r200t100d40e300')
if test3_2:
    test.test_3_2('narx/r200t100d40e300', tests=2)
