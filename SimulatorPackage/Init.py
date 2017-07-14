from Cycle import Cycle


# GA(graphics=True).run([300, 300], random.randint(0, 360), [1100, 600])

cycle = Cycle()

learning_runs = 30
learning_time = 1000
testing_time = 100
input_delay = 100
output_delay = 100
max_epochs = 100

random_movement = 100
look_ahead = 50

# cycle.wake_learning(vehicle_runs=learning_runs, vehicle_iter=learning_time, test_iter=testing_time, input_delay=input_delay,
#                     output_delay=output_delay, net_max_iter=max_epochs, show_graph=False)

cycle.wake_learning(random_movement, train_network=False, learning_runs=learning_runs, learning_time=learning_time,
                    testing_time=testing_time, input_delay=input_delay, output_delay=output_delay,
                    max_epochs=max_epochs, show_graph=False)

cycle.sleep(net_filename='r100t1000d25-25', look_ahead=look_ahead, individuals=30, generations=20)

cycle.wake_testing(iterations=look_ahead)
