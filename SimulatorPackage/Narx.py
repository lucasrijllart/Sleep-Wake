import pyrenn as pr
import numpy as np
from Sprites import run_through_brain
from sklearn.neural_network import MLPRegressor
import pickle


def load_pyrenn(filename='narxNet'):
    return pr.loadNN(filename)


def load_narx_mlp(filename):
    return pickle.load(open(filename, 'rb'))


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

    def train(self, training_data, max_iter=200, verbose=False, use_mean=True):
        input_matrices = []
        target_matrices = []
        slice_point = self.delay - 1
        for vehicle in training_data:
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
                if use_mean:
                    rolled[0, 0:it] = mean
                else:
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
                    if use_mean:
                        rolled[0, 0:it] = mean
                    else:
                        rolled[0, 0:it] = 0.0
                    delay_matrix = np.concatenate((delay_matrix, rolled), axis=0)

            delay_matrix = np.delete(delay_matrix, -1, axis=1)
            # drop the columns that needed padding
            for _ in range(0, slice_point):
                delay_matrix = np.delete(delay_matrix, 0, axis=1)
                target = np.delete(target, 0, axis=1)
            # collect the input, target matrices
            input_matrices.append(delay_matrix)
            target_matrices.append(target)

        input_matrices = np.array(input_matrices)
        input_matrices = np.concatenate((input_matrices[:]), axis=1)

        target_matrices = np.array(target_matrices)
        target_matrices = np.concatenate((target_matrices[:]), axis=1)
        # print target_matrices
        # print input_matrices

        self.net = pr.train_LM(input_matrices, target_matrices, self.net, k_max=max_iter, E_stop=1e-6, verbose=verbose)

    def predict_noe(self, data, individual, look_ahead):
        """ Predicts the next sensory values up until the lookahead for a given individual with the data provided
        using an NOE network """
        sensor_log = np.array([[], []])
        wheel_log = []

        next_input = np.array(data[:, -1])
        next_input = np.array([[x] for x in next_input])

        # Execute the first prediction without adding it to the data, as the first prediction comes from actual data
        # 1. predict next sensory output
        prediction = self.predict(next_input, pre_inputs=data[:, :-1], pre_outputs=data[2:, :-1])
        # 2. log predicted sensory information to list (used for fitness)
        sensor_log = np.concatenate((sensor_log, prediction), axis=1)

        # 3. feed it to the brain to get motor information
        wheel_log.append(run_through_brain(prediction, individual))

        # 4. add this set of data to the input of the prediction
        next_input = np.array([wheel_log[-1][0], wheel_log[-1][1], prediction[0], prediction[1]]).reshape(4, 1)

        for it in range(1, look_ahead):  # loop through the time steps
            # 1. predict next sensory output
            prediction = self.predict(next_input, pre_inputs=data, pre_outputs=data[2:])

            # 2. log predicted sensory information to list (used for fitness)
            sensor_log = np.concatenate((sensor_log, prediction), axis=1)

            # 3. feed it to the brain to get motor information
            wheel_log.append(run_through_brain(prediction, individual))

            # 4. concatenate previous step to the full data
            data = np.concatenate((data, next_input), axis=1)

            # 5. set the predicted data to the next input of the prediction
            next_input = np.array([wheel_log[-1][0], wheel_log[-1][1], prediction[0], prediction[1]]).reshape(4, 1)
            # loop back to 1 until reached timestep
        return sensor_log, wheel_log

    def predict(self, x, pre_inputs=None, pre_outputs=None):
        if pre_inputs is not None:
            y = pr.NNOut(x, self.net, pre_inputs, pre_outputs)
        else:
            delayed_input = []
            for idx in range(0, self.inputs):
                delayed_input.append(x[idx, 0])
                delay_data = self.past_data[idx, -(self.delay-1):]
                delayed_input.extend(delay_data[::-1])
            delayed_input = np.array(delayed_input)
            delayed_input = np.reshape(delayed_input, (self.layers[0], 1))
            y = pr.NNOut(delayed_input, self.net)
        return y

    def save_to_file(self, filename='narxNet'):
        pr.saveNN(self.net, filename=filename)

    def predict_ahead(self, data, ind, look_ahead):
        sensor_log = np.array([[], []]) #return that
        wheel_log = [] # return that

        if data.size == 0:
            data = np.zeros((4, self.delay))
            next_input = data[:, -1]
            next_input = np.array([[x] for x in next_input])
        else:
            next_input = np.array(data[:, -1])
            next_input = np.array([[x] for x in next_input])

        self.set_past_data(data[:, :-1]) #set the past data
        # Execute the first prediction without adding it to the data, as the first prediction comes from actual data
        # 1. predict next sensory output
        prediction = self.predict(next_input)
        # ad the past data to the network
        self.update_past_data(next_input)
        # 2. log predicted sensory information to list (used for fitness)
        sensor_log = np.concatenate((sensor_log, prediction), axis=1)

        # 3. feed it to the brain to get motor information
        wheel_log.append(run_through_brain(prediction, ind))

        # 4. add this set of data to the input of the prediction
        next_input = np.array([wheel_log[-1][0], wheel_log[-1][1], prediction[0], prediction[1]]).reshape(4, 1)

        for it in range(1, look_ahead):  # loop through the time steps
            # 1. predict next sensory output
            prediction = self.predict(next_input)

            # 2. log predicted sensory information to list (used for fitness)
            sensor_log = np.concatenate((sensor_log, prediction), axis=1)

            # 3. feed it to the brain to get motor information
            wheel_log.append(run_through_brain(prediction, ind))

            # 4. append previous step to the full data
            self.update_past_data(next_input)

            # 5. set the predicted data to the next input of the prediction
            next_input = np.array([wheel_log[-1][0], wheel_log[-1][1], prediction[0], prediction[1]]).reshape(4, 1)
            # loop back to 1 until reached time-step
        return sensor_log, wheel_log

    def predict_error_graph(self, data, look_ahead, predict_after, narx):
        if narx:
            # determine the past data
            past_data = np.array(data[:, :predict_after])
            # intialize the arrays for the logs
            sensor_log = np.array([[], []])  # return that
            wheel_log = []  # return that

            # If no previous observations. Add zro padding
            padding = self.delay - past_data.shape[1]
            if padding > 0:
                padding = np.zeros((4, padding), dtype=float)
                past_data = np.concatenate((padding, past_data), axis=1)

            next_input = np.array(past_data[:, -1])
            next_input = np.array([[x] for x in next_input])

            self.set_past_data(past_data[:, :-1])  # set the past data
            # Execute the first prediction without adding it to the data, as the first prediction comes from actual data
            # 1. predict next sensory output
            prediction = self.predict(next_input)
            # ad the past data to the network
            self.update_past_data(next_input)
            # 2. log predicted sensory information to list (used for fitness)
            sensor_log = np.concatenate((sensor_log, prediction), axis=1)

            # 3. Get motor information
            wheel_l, wheel_r = [data[0, predict_after], data[1, predict_after]]
            wheel_log.append([wheel_l, wheel_r])

            # 4. add this set of data to the input of the prediction
            next_input = np.array([[wheel_l], [wheel_r], prediction[0], prediction[1]]).reshape(4, 1)

            for it in range(1, look_ahead):  # loop through the time steps
                # 1. predict next sensory output
                prediction = self.predict(next_input)

                # 2. log predicted sensory information to list (used for fitness)
                sensor_log = np.concatenate((sensor_log, prediction), axis=1)

                # 3. feed it to the brain to get motor information
                wheel_l, wheel_r = [data[0, predict_after+it], data[1, predict_after+it]]
                wheel_log.append([wheel_l, wheel_r])
                # 4. append previous step to the full data
                self.update_past_data(next_input)

                # 5. set the predicted data to the next input of the prediction
                next_input = np.array([[wheel_l], [wheel_r], prediction[0], prediction[1]]).reshape(4, 1)
                # loop back to 1 until reached time-step
            return sensor_log, wheel_log
        else:
            sensor_log = np.array([[], []])
            wheel_log = []

            next_input = np.array(data[:, -1])
            next_input = np.array([[x] for x in next_input])

            # Execute the first prediction without adding it to the data, as the first prediction comes from actual data
            # 1. predict next sensory output
            prediction = self.predict(next_input, pre_inputs=data[:, :-1], pre_outputs=data[2:, :-1])
            # 2. log predicted sensory information to list (used for fitness)
            sensor_log = np.concatenate((sensor_log, prediction), axis=1)

            # 3. Get motor information
            wheel_l, wheel_r = [data[0, predict_after], data[1, predict_after]]
            wheel_log.append([wheel_l, wheel_r])

            # 4. add this set of data to the input of the prediction
            next_input = np.array([wheel_log[-1][0], wheel_log[-1][1], prediction[0], prediction[1]]).reshape(4, 1)

            for it in range(1, look_ahead):  # loop through the time steps
                # 1. predict next sensory output
                prediction = self.predict(next_input, pre_inputs=data, pre_outputs=data[2:])

                # 2. log predicted sensory information to list (used for fitness)
                sensor_log = np.concatenate((sensor_log, prediction), axis=1)

                # 3. Get motor information
                wheel_l, wheel_r = [data[0, predict_after], data[1, predict_after]]
                wheel_log.append([wheel_l, wheel_r])

                # 4. concatenate previous step to the full data
                data = np.concatenate((data, next_input), axis=1)

                # 5. set the predicted data to the next input of the prediction
                next_input = np.array([wheel_log[-1][0], wheel_log[-1][1], prediction[0], prediction[1]]).reshape(4, 1)
                # loop back to 1 until reached timestep
            return sensor_log, wheel_log

class NarxMLP:

    def __init__(self):
        self.net = MLPRegressor(hidden_layer_sizes=(100, 100), activation='tanh'
                ,solver='lbfgs', alpha=0.0001, batch_size='auto'
                ,learning_rate_init=0.001, max_iter=200, random_state=None, tol=0.00001
                , verbose=True, warm_start=False
                ,early_stopping=False, validation_fraction=0.1)

    def fit(self, train_data, delay, use_mean):
        x, t = prepare_data(train_data, delay, use_mean)
        return self.net.fit(x, t)

    def to_file(self, filename):
        pickle.dump(self.net, open(filename, 'wb'))



def prepare_data(data, delay, use_mean):
    input_matrices = []
    target_matrices = []
    slice_point = delay - 1
    for vehicle in data:
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
        for it in range(1, delay):  # add a delay vector(line)
            rolled = np.reshape(np.roll(vehicle[0], it), (1, c))
            # replace the missing observations
            # with zeros or mean of timeseries
            if use_mean:
                rolled[0, 0:it] = mean
            else:
                rolled[0, 0:it] = 0.0
            delay_matrix = np.concatenate((delay_matrix, rolled), axis=0)
        # print delay_matrix.shape

        # create input delays for the rest of the time series
        for idx in range(1, r):  # iterate the rows
            mean = np.mean(vehicle[idx])
            for it in range(0, delay):  # add a delay vector(line)
                rolled = np.reshape(np.roll(vehicle[idx], it), (1, c))
                # replace the missing observations
                # with zeros or mean of timeseries
                if use_mean:
                    rolled[0, 0:it] = mean
                else:
                    rolled[0, 0:it] = 0.0
                delay_matrix = np.concatenate((delay_matrix, rolled), axis=0)

        delay_matrix = np.delete(delay_matrix, -1, axis=1)
        # drop the columns that needed padding
        for _ in range(0, slice_point):
            delay_matrix = np.delete(delay_matrix, 0, axis=1)
            target = np.delete(target, 0, axis=1)
        # collect the input, target matrices
        input_matrices.append(delay_matrix)
        target_matrices.append(target)

    input_matrices = np.array(input_matrices)
    input_matrices = np.concatenate((input_matrices[:]), axis=1)
    input_matrices = np.transpose(input_matrices)

    target_matrices = np.array(target_matrices)
    target_matrices = np.concatenate((target_matrices[:]), axis=1)
    target_matrices = np.transpose(target_matrices)
    # print target_matrices
    # print input_matrices
    return input_matrices, target_matrices
