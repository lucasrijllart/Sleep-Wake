from Cycle import Cycle
import time

learning_runs = 5
learning_time = 300
input_delay = 10
output_delay = 10
max_epochs = 5
train_network = 'test'

random_movement = 50
look_ahead = 400

start_time = time.time()

cycle = Cycle(net_filename=None)

cycle.wake_learning(random_movement, train_network=train_network, learning_runs=learning_runs, learning_time=learning_time,
                    input_delay=input_delay, output_delay=output_delay, max_epochs=max_epochs)



cycle.show_error_graph(testing_time=400, predict_after=100)

# cycle.sleep(look_ahead=look_ahead, individuals=30, generations=10)


# cycle.wake_testing(iterations=200)
