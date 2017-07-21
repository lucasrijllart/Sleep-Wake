from Cycle import Cycle

learning_runs = 50
learning_time = 300
input_delay = 100
output_delay = 100
max_epochs = 100
train_network = 'r50t300d10-10'

initial_random_movement = 50
look_ahead = 100


cycle = Cycle(net_filename='narx/r100t300d50-51')

# cycle.wake_learning(initial_random_movement, train_network=None, learning_runs=learning_runs, learning_time=learning_time,
#                     input_delay=input_delay, output_delay=output_delay, max_epochs=max_epochs)

brain = [-2, 6, -2, 6, 1, 1]
cycle.show_error_graph(testing_time=200, predict_after=50)
# cycle.sleep(look_ahead=look_ahead, individuals=10, generations=10)

# cycle.wake_testing(iterations=200)
