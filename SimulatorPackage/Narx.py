import pyrenn as pr
import numpy as np
from sklearn.neural_network import MLPRegressor

def load_net(filename='narxNet'):
    return pr.loadNN(filename)


class PyrennNarx:

    def __init__(self, layers=[4, 10, 10, 2], delay=10):
        self.in_nodes = layers[0] * delay
        self.inputs = layers[0]
        layers[0] *= delay
        self.layers = layers
        self.delay = delay
        self.past_data = []
        self.net = pr.CreateNN(self.layers)

    def set_past_data(self, past_data):
        self.past_data = past_data

    def update_past_data(self, data):
        self.past_data = np.concatenate((self.past_data, data), axis=1)

    def set_net(self, net):
        # Get the number of input nodes and devide
        # by time-series inputs to get the delay for the
        # saved network and set the input nodes number
        self.delay = net['nn'][0]/self.inputs
        self.layers[0] = net['nn'][0]
        self.net = net

    def getNet(self):
        return self.net

    def train(self, inputs, targets, max_iter=200, verbose=False):
        print targets
        input_matrices = []
        target_matrices = []
        for vehicle in inputs:
            r, c = vehicle.shape

            # create the TARGET, sensor inputs for both sensors
            # drop the last column as it dow not have past observations
            sl = np.reshape(np.array(vehicle[2]), (1, c))
            sr = np.reshape(np.array(vehicle[3]), (1, c))
            target = np.concatenate((sl, sr), axis=0)
            target = np.delete(target, 0, axis=1)
            # print target

            # create first row in delay matrix with first motor observations
            delay_matrix = np.array([vehicle[0]])
            mean = np.mean(vehicle[0])
            for it in range(1, self.delay):  # add a delay vector(line)
                rolled = np.reshape(np.roll(vehicle[0], it), (1, c))
                # replace the missing observations
                # with zeros or mean of timeseries
                rolled[0, 0:it] = 0.0
                delay_matrix = np.concatenate((delay_matrix, rolled), axis=0)
            # print delay_matrix.shape

            # create input delays for the rest of the time series
            for idx in range(1, r): # iterate the rows
                mean = np.mean(vehicle[idx])
                for it in range(0, self.delay):# add a delay vector(line)
                    rolled = np.reshape(np.roll(vehicle[idx], it), (1, c))
                    # replace the missing observations
                    # with zeros or mean of timeseries
                    rolled[0, 0:it] = 0.0
                    delay_matrix = np.concatenate((delay_matrix, rolled), axis=0)

            delay_matrix = np.delete(delay_matrix, -1, axis=1)

            # collect the input, target matrices
            input_matrices.append(delay_matrix)
            target_matrices.append(target)

        input_matrices = np.array(input_matrices)
        input_matrices = np.concatenate((input_matrices[:]), axis=1)

        target_matrices = np.array(target_matrices)
        target_matrices = np.concatenate((target_matrices[:]), axis=1)
        # print target_matrices
        # print input_matrices

        self.net = pr.train_LM(input_matrices, target_matrices, self.net, k_max=max_iter, verbose=verbose)

    def predict(self, x, pre_inputs=None, pre_outputs=None):
        if pre_inputs != None:
            y = pr.NNOut(x, self.net, pre_inputs, pre_outputs)
        else:
            delayed_input = []
            for idx in range(0, self.inputs):
                delayed_input.append(x[idx, 0])
                delay_data = self.past_data[idx, -(self.delay-1):]
                delayed_input.extend(delay_data[::-1])
                # for idx_2 in range(0, self.delay):
                #     delayed_input.append(self.past_data[idx, -idx_2])
            delayed_input = np.array(delayed_input)
            delayed_input = np.reshape(delayed_input, (self.layers[0], 1))
            y = pr.NNOut(delayed_input, self.net)
        return y

    def save_to_file(self, filename='narxNet'):
        pr.saveNN(self.net, filename=filename)

class NarxMLP:

    def __init__(self):
        self.net = MLPRegressor()