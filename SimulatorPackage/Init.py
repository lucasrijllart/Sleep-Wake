from Cycle import Cycle


# GA(graphics=True).run([300, 300], random.randint(0, 360), [1100, 600])

cycle = Cycle()

learning_runs = 20
learning_time = 1000
testing_time = 1
input_delay = 20
output_delay = 20
max_epochs = 150

# cycle.wake_learning(vehicle_runs=learning_runs, vehicle_iter=learning_time, test_iter=testing_time, input_delay=input_delay,
#                     output_delay=output_delay, net_max_iter=max_epochs, show_graph=False)


cycle.sleep(netfileName='narxNet', lookAhaid=100, generations=10)
