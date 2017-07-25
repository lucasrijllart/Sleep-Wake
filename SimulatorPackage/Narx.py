import pyrenn as pr
import numpy as np
from sklearn.neural_network import MLPRegressor

def load_net(filename='narxNet'):
    return pr.loadNN(filename)


class PyrennNarx:

    def __init__(self, layers=[4, 10, 10, 2], input_delay=10,
                 output_delay=10):
        self.layers = layers
        self.out_delay = range(1, output_delay)
        self.in_delay = range(1, input_delay)

        self.net = pr.CreateNN(self.layers, dIn=self.in_delay, dOut=self.out_delay)

    def set_net(self, net):
        self.net = net

    def getNet(self):
        return self.net

    def train(self, inputs, targets, max_iter=200, verbose=False):
        delay = 3
        r, c = inputs.shape

        # create the target, sensor inputs for both sensors
        # drop the last column as it dow not have past observations
        sl = np.reshape(np.array(inputs[2]), (1, 20))
        sr = np.reshape(np.array(inputs[3]), (1, 20))
        target = np.concatenate((sl, sr), axis=0)
        target = np.delete(target, 0, axis=1)
        print target

        # create first row in delay matrix
        delay_matrix = np.array([inputs[0]])
        mean = np.mean(inputs[0])
        for it in range(1, delay):  # add a delay vector(line)
            rolled = np.reshape(np.roll(inputs[0], it), (1, 20))
            # replace the missing observations
            # with zeros or mean of timeseries
            rolled[0, 0:it] = 0.0
            delay_matrix = np.concatenate((delay_matrix, rolled), axis=0)
        print delay_matrix.shape

        # create input delays for the rest of the time series
        for idx in range(1, r): # iterate the rows
            mean = np.mean(inputs[idx])
            for it in range(0, delay):# add a delay vector(line)
                rolled = np.reshape(np.roll(inputs[idx], it), (1, 20))
                # replace the missing observations
                # with zeros or mean of timeseries
                rolled[0, 0:it] = 0.0
                delay_matrix = np.concatenate((delay_matrix, rolled), axis=0)

        print delay_matrix.shape
        delay_matrix = np.delete(delay_matrix, -1, axis=1)
        print delay_matrix
        self.net = pr.train_LM(inputs, targets, self.net, k_max=max_iter, verbose=verbose)

    def predict(self, x, pre_inputs=None, pre_outputs=None):
        y = pr.NNOut(x, self.net, pre_inputs, pre_outputs)
        return y

    def save_to_file(self, filename='narxNet'):
        pr.saveNN(self.net, filename=filename)

class NarxMLP:

    def __init__(self):
        self.net = MLPRegressor()