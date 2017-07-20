from Cycle import Cycle
import time

learning_runs = 5
learning_time = 300
input_delay = 100
output_delay = 100
max_epochs = 100
train_network = 'r50t300d100-100'

intial_random_movement = 50
look_ahead = 100

start_time = time.time()

cycle = Cycle(net_filename='narx/r50t300d100-100')

cycle.wake_learning(intial_random_movement, train_network=train_network, learning_runs=learning_runs, learning_time=learning_time,
                    input_delay=input_delay, output_delay=output_delay, max_epochs=max_epochs)

brain = [1, 7, 1, 7, 3, 3]
cycle.show_error_graph(testing_time=400, predict_after=100, brain=brain)

# cycle.sleep(look_ahead=look_ahead, individuals=30, generations=10)


# cycle.wake_testing(iterations=200)
