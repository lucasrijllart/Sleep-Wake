import pyrenn as pr

def load_net(filename='narxNet'):
    return pr.loadNN(filename)

class Narx:

    def __init__(self, layers=[4, 10, 10, 2], input_delay=5,
                 output_delay=5):
        self.layers = layers
        self.out_delay = range(1, output_delay)
        self.in_delay = range(1, input_delay)

        self.net = pr.CreateNN(self.layers, dIn=self.in_delay, dOut=self.out_delay)

    def getNet(self):
        return self.net

    def train(self, inputs, targets, max_iter=200, verbose=False):

        self.net = pr.train_LM(inputs, targets, self.net, k_max=max_iter, verbose=verbose)

    def predict(self, x, pre_inputs=None, pre_outputs=None):
        y = pr.NNOut(x, self.net, pre_inputs, pre_outputs)
        return y

    def save_to_file(self, filename='narxNet'):
        pr.saveNN(self.net, filename=filename)

