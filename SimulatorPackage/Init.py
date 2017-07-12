from Cycle import Cycle


# GA(graphics=True).run([300, 300], random.randint(0, 360), [1100, 600])

cycle = Cycle()

learning_runs = 2
learning_time = 400
testing_time = 500
input_delay = 5
output_delay = 5
max_epochs = 50

# cycle.wake_learning(vehicle_runs=learning_runs, vehicle_iter=learning_time, test_iter=testing_time, input_delay=input_delay,
#                     output_delay=output_delay, net_max_iter=max_epochs, show_graph=False)

cycle.wake_learning(50, train_network=True, learning_runs=learning_runs, learning_time=learning_time, input_delay=input_delay, output_delay=output_delay, max_epochs=max_epochs)

cycle.sleep(net_filename='narxNet', lookAhaid=100, generations=10)
