from Cycle import Cycle

learning_runs = 50
learning_time = 300
input_delay = 10
output_delay = 10
max_epochs = 50
train_network = None  # 'randoms_brains'

intial_random_movement = 50
look_ahead = 100


cycle = Cycle(net_filename='narx/r100t300d10-10e50')

cycle.wake_learning(intial_random_movement, train_network=train_network, learning_runs=learning_runs,
                    learning_time=learning_time, input_delay=input_delay, output_delay=output_delay,
                    max_epochs=max_epochs)

brain = [-2, 8, -2, 8, 5, 5]
cycle.show_error_graph(testing_time=80, predict_after=40, brain=brain)

# cycle.sleep(look_ahead=look_ahead, individuals=30, generations=10)

# cycle.wake_testing(iterations=200)
