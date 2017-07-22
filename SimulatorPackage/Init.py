from Cycle import Cycle

learning_runs = 400
learning_time = 200
input_delay = 20
output_delay = 20
max_epochs = 100

initial_random_movement = 50

look_ahead = 100


train_network = False
wake_learn = False
error_graph = True
sleep = False
wake_test = False

cycle = Cycle(net_filename='narx/r400t200d20e100')

if train_network:
    cycle.train_network(learning_runs, learning_time, input_delay, output_delay, max_epochs)

if wake_learn:
    cycle.wake_learning(initial_random_movement)

if error_graph:
    brain = [-7, 8,
             6, 4, 4, 4]
    cycle.show_error_graph(testing_time=200, predict_after=20, brain=brain)

if sleep:
    cycle.sleep(look_ahead=look_ahead, individuals=30, generations=10)

if wake_test:
    cycle.wake_testing(iterations=200)
