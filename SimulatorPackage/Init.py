from Cycle import Cycle

learning_runs = 5
learning_time = 500
testing_time = 100
input_delay = 10
output_delay = 10
max_epochs = 5

random_movement = 50
look_ahead = 400


cycle = Cycle(net_filename='narx/delay100[10, 10]')

cycle.wake_learning(random_movement, train_network=False, learning_runs=learning_runs, learning_time=learning_time,
                    testing_time=testing_time, input_delay=input_delay, output_delay=output_delay,
                    max_epochs=max_epochs, show_error_graph=True)


cycle.show_error_graph()

cycle.sleep(look_ahead=look_ahead, individuals=30, generations=10)


# cycle.wake_testing(iterations=200)
