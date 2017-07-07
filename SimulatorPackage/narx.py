import pyrenn as pr

class narx:

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

    def predict(self, x):
        y = pr.NNOut(x, self.net)
        return y

    #TODO: May need to add a method to take into account
    #TODO: the old inputs and outputs