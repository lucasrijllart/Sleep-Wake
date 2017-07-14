from Cycle import Cycle


# GA(graphics=True).run([300, 300], random.randint(0, 360), [1100, 600])

cycle = Cycle()

learning_runs = 10
learning_time = 1000
testing_time = 1
input_delay = 30
output_delay = 30
max_epochs = 25

# cycle.wake_learning(vehicle_runs=learning_runs, vehicle_iter=learning_time, test_iter=testing_time, input_delay=input_delay,
#                     output_delay=output_delay, net_max_iter=max_epochs, show_graph=False)

cycle.wake_learning(random_movements=100, train_network=False, learning_runs=learning_runs,
                    learning_time=learning_time, input_delay=input_delay, output_delay=output_delay,
                    max_epochs=max_epochs)

cycle.sleep(net_filename='delay100[10, 10]', look_ahead=100, generations=50)

cycle.wake_testing()