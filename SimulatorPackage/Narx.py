import pyrenn as pr
import numpy as np
from sklearn.neural_network import MLPRegressor

def load_net(filename='narxNet'):
    return pr.loadNN(filename)


class PyrennNarx:

    def __init__(self, layers=[4, 10, 10, 2], delay=10):
        layers[0] *= delay
        self.layers = layers
        self.delay = delay

        self.net = pr.CreateNN(self.layers)

    def set_net(self, net):
        self.net = net

    def getNet(self):
        return self.net

    def train(self, inputs, targets, max_iter=200, verbose=False):

        for vehicle in inputs:
            r, c = vehicle.shape

            # create the TARGET, sensor inputs for both sensors
            # drop the last column as it dow not have past observations
            sl = np.reshape(np.array(vehicle[2]), (1, c))
            sr = np.reshape(np.array(vehicle[3]), (1, c))
            target = np.concatenate((sl, sr), axis=0)
            target = np.delete(target, 0, axis=1)
            print target

            # create first row in delay matrix with first motor observations
            delay_matrix = np.array([vehicle[0]])
            mean = np.mean(vehicle[0])
            for it in range(1, self.delay):  # add a delay vector(line)
                rolled = np.reshape(np.roll(vehicle[0], it), (1, c))
                # replace the missing observations
                # with zeros or mean of timeseries
                rolled[0, 0:it] = 0.0
                delay_matrix = np.concatenate((delay_matrix, rolled), axis=0)
            print delay_matrix.shape

            # create input delays for the rest of the time series
            for idx in range(1, r): # iterate the rows
                mean = np.mean(vehicle[idx])
                for it in range(0, self.delay):# add a delay vector(line)
                    rolled = np.reshape(np.roll(vehicle[idx], it), (1, c))
                    # replace the missing observations
                    # with zeros or mean of timeseries
                    rolled[0, 0:it] = 0.0
                    delay_matrix = np.concatenate((delay_matrix, rolled), axis=0)

            print delay_matrix.shape
            delay_matrix = np.delete(delay_matrix, -1, axis=1)
            print delay_matrix

        # TODO: concatenate the delay matrices for every vehicle to represent full
        # network training data
        self.net = pr.train_LM(vehicle, targets, self.net, k_max=max_iter, verbose=verbose)

    def predict(self, x, pre_inputs=None, pre_outputs=None):
        y = pr.NNOut(x, self.net, pre_inputs, pre_outputs)
        return y

    def save_to_file(self, filename='narxNet'):
        pr.saveNN(self.net, filename=filename)

class NarxMLP:

    def __init__(self):
        self.net = MLPRegressor()