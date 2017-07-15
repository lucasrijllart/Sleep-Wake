from Cycle import Cycle


# GA(graphics=True).run([300, 300], random.randint(0, 360), [1100, 600])

cycle = Cycle()

learning_runs = 1
learning_time = 500
testing_time = 100
input_delay = 5
output_delay = 5
max_epochs = 10

random_movement = 50
look_ahead = 20

# cycle.wake_learning(vehicle_runs=learning_runs, vehicle_iter=learning_time, test_iter=testing_time, input_delay=input_delay,
#                     output_delay=output_delay, net_max_iter=max_epochs, show_graph=False)

cycle.wake_learning(random_movement, train_network=False, learning_runs=learning_runs, learning_time=learning_time,
                    testing_time=testing_time, input_delay=input_delay, output_delay=output_delay,
                    max_epochs=max_epochs, show_error_graph=True)


cycle.sleep(net_filename='narx/delay100[10, 10]', look_ahead=look_ahead, individuals=30, generations=10, show_error_graph=True)

cycle.wake_testing(iterations=200)
