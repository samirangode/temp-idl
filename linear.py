# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np
import math

class Linear():
    def __init__(self, in_feature, out_feature, weight_init_fn, bias_init_fn):

        """
        Argument:
            W (np.array): (in feature, out feature)
            dW (np.array): (in feature, out feature)
            momentum_W (np.array): (in feature, out feature)

            b (np.array): (1, out feature)
            db (np.array): (1, out feature)
            momentum_B (np.array): (1, out feature)
        """

        self.W = weight_init_fn(in_feature, out_feature)
        self.b = bias_init_fn(out_feature)

        # TODO: Complete these but do not change the names.
        self.dW = np.zeros((self.W.shape[0],self.W.shape[1]))
        self.db = np.zeros((self.b.shape[0],self.b.shape[1]))
        self.forward_compute = 0

        self.momentum_W = np.zeros((self.W.shape[0],self.W.shape[1]))
        self.momentum_b = np.zeros((self.b.shape[0],self.b.shape[1]))

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """
        Argument:
            x (np.array): (batch size, in feature)
        Return:
            out (np.array): (batch size, out feature)
        """
        self.x = x
        y_out = np.dot(x,self.W) + self.b #lg
        self.forward_compute = y_out #check this
        #raise NotImplemented
        return y_out

        # raise NotImplemented

    def backward(self, delta):

        """
        Argument:
            delta (np.array): (batch size, out feature)
        Return:
            out (np.array): (batch size, in feature)
        """
        size = self.x.shape[0]
        self.dW = np.dot(np.transpose(self.x),delta)/size
        self.db = np.sum(delta,axis = 0)/size
        self.db = self.db.reshape(1,delta.shape[1])
        dZ = np.dot(delta, np.transpose(self.W))
        return dZ
        #raise NotImplemented
