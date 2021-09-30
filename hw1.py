"""
Follow the instructions provided in the writeup to completely
implement the class specifications for a basic MLP, optimizer, .
You will be able to test each section individually by submitting
to autolab after implementing what is required for that section
-- do not worry if some methods required are not implemented yet.

Notes:

The __call__ method is a special reserved method in
python that defines the behaviour of an object when it is
used as a function. For example, take the Linear activation
function whose implementation has been provided.

# >>> activation = Identity()
# >>> activation(3)
# 3
# >>> activation.forward(3)
# 3
"""

# DO NOT import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

# from mytorch.batchnorm import BatchNorm
import numpy as np
import os
import sys

sys.path.append('mytorch')
from loss import *
from activation import *
from batchnorm import *
from linear import *


class MLP(object):

    """
    A simple multilayer perceptron
    """

    def __init__(self, input_size, output_size, hiddens, activations, weight_init_fn,
                 bias_init_fn, criterion, lr, momentum=0.0, num_bn_layers=0):

        # Don't change this -->
        self.train_mode = True
        self.num_bn_layers = num_bn_layers
        self.bn = num_bn_layers > 0
        self.nlayers = len(hiddens) + 1
        self.input_size = input_size
        self.output_size = output_size
        self.activations = activations
        self.criterion = criterion
        self.lr = lr
        self.momentum = momentum
        # <---------------------

        # Don't change the name of the following class attributes,
        # the autograder will check against these attributes. But you will need to change
        # the values in order to initialize them correctly

        # Initialize and add all your linear layers into the list 'self.linear_layers'
        # (HINT: self.foo = [ bar(???) for ?? in ? ])
        # (HINT: Can you use zip here?)
        #linlr = Linear(self.input_size,hidden[i],weight_init_fn,bias_init_fn)
        # layer_sizes = []
        # layer_sizes.append(input_size)
        # for i in hiddens:
        #     layer_sizes.append(i)
        # layer_sizes.append(output_size)
        layer_size = []
        # layer_size.append(input_size)
        # layer_size.append(hiddens)
        # layer_size.append(output_size)
        layer_size = [input_size] + hiddens + [output_size]
        ###[TODO] Try out the other method
        # layer_size.append(zip(input_size,hiddens[0]))
        # for i in range(len(hiddens)-1):
        #     layer_size.append(zip(hiddens[i],hiddens[i+1]))
        # layer_size.append(zip(hiddens[len(hiddens)-1],output_size))
        till_end = -1
        start = 1
        self.linear_layers = [Linear(i, j ,weight_init_fn,bias_init_fn) for i,j in zip(layer_size[:till_end],layer_size[start:])]

        # If batch norm, add batch norm layers into the list 'self.bn_layers'
        if self.bn:
            #self.bn_layers = [BatchNorm(la) for i in ]
            # for i in range(num_bn_layers):
            #     BatchNorm(layer_size[i][1])
            #self.bn_layers = [BatchNorm(in_features) for in_features in layer_size[:num_bn_layers]]
            self.bn_layers = [BatchNorm(layer_size[i+1]) for i in range(self.num_bn_layers)]



    def forward(self, x):
        """
        Argument:
            x (np.array): (batch size, input_size)
        Return:
            out (np.array): (batch size, output_size)
        """
        # Complete the forward pass through your entire MLP.
        input = x
        a = self.num_bn_layers 
        for i in range(self.nlayers):
            linear_layer = self.linear_layers[i]
            input = linear_layer.forward(input)
            # batch_norm = self.bn_layers[i]
            # input = batch_norm(input)
            if self.bn:
                if(a>0):
                    batch_norm = self.bn_layers[i]
                    a = a - 1
                    if(self.train_mode):
                        input = batch_norm.forward(input)
                    else:
                        input = batch_norm.forward(input,eval=True)
            input = self.activations[i].forward(input)
            self.input = input
        return self.input

        # raise NotImplemented

    def zero_grads(self):
        # Use numpyArray.fill(0.0) to zero out your backpropped derivatives in each
        # of your linear and batchnorm layers.
        #self.linear_layers = [layer.backward() for layer in self.linear_layers]
        for layer in self.linear_layers:
            layer.dW.fill(0.0)
            layer.db.fill(0.0)
        if(self.bn):
            for blayer in self.bn_layers:
                blayer.dgamma.fill(0.0)
                blayer.dbeta.fill(0.0)

        
        #raise NotImplemented

    def step(self):
        # Apply a step to the weights and biases of the linear layers.
        # Apply a step to the weights of the batchnorm layers.
        # (You will add momentum later in the assignment to the linear layers only
        # , not the batchnorm layers)

        a = self.num_bn_layers
        #W_change = 0
        for i in range(self.nlayers):
            # Update weights and biases here
            if(self.momentum==0):
                self.linear_layers[i].W = self.linear_layers[i].W - self.lr * self.linear_layers[i].dW
                self.linear_layers[i].b = self.linear_layers[i].b - self.lr * self.linear_layers[i].db
            else:
                #self.linear_layers[i].dW = self.linear_layers[i].dW - self.lr * self
                beta = self.momentum
                lr = self.lr
                self.linear_layers[i].momentum_W =  beta*self.linear_layers[i].momentum_W - lr * self.linear_layers[i].dW
                self.linear_layers[i].momentum_b = beta*self.linear_layers[i].momentum_b - lr * self.linear_layers[i].db
                self.linear_layers[i].W = self.linear_layers[i].W + self.linear_layers[i].momentum_W
                self.linear_layers[i].b = self.linear_layers[i].b + self.linear_layers[i].momentum_b 
            #pass
            if(self.bn):
                if(a>0):
                    self.bn_layers[i].gamma = self.bn_layers[i].gamma - self.lr * self.bn_layers[i].dgamma
                    self.bn_layers[i].beta = self.bn_layers[i].beta - self.lr * self.bn_layers[i].dbeta
                    a = a -1
        # Do the same for batchnorm layers

        #raise NotImplemented

    def backward(self, labels):
        # Backpropagate through the activation functions, batch norm and
        # linear layers.
        # Be aware of which return derivatives and which are pure backward passes
        # i.e. take in a loss w.r.t it's output.
        # criterion.
        #raise NotImplemented
        a= self.num_bn_layers-1
        #logits = self.activations[len(self.activations)-1].state
        logits = self.input        
        loss = self.criterion.forward(logits,labels)
        delta = self.criterion.derivative()
        # delta = self.linear_layers[len(self.linear_layers)-1].backward(delta)
        for i in (range(self.nlayers-1,-1,-1)):
            #linear_layer = self.nlayers[i].backward(delta)
            #delta = self.activations[i].backward(delta)
            delta = self.activations[i].derivative()*delta
            if(self.bn>0):
                if(i<=a):
                    delta = self.bn_layers[i].backward(delta)
            delta = self.linear_layers[i].backward(delta)


    def error(self, labels):
        return (np.argmax(self.output, axis = 1) != np.argmax(labels, axis = 1)).sum()

    def total_loss(self, labels):
        return self.criterion(self.output, labels).sum()

    def __call__(self, x):
        return self.forward(x)

    def train(self):
        self.train_mode = True

    def eval(self):
        self.train_mode = False

#This function does not carry any points. You can try and complete this function to train your network.
def get_training_stats(mlp, dset, nepochs, batch_size):

    train, val, _ = dset
    trainx, trainy = train
    valx, valy = val

    idxs = np.arange(len(trainx))

    training_losses = np.zeros(nepochs)
    training_errors = np.zeros(nepochs)
    validation_losses = np.zeros(nepochs)
    validation_errors = np.zeros(nepochs)

    # Setup ...

    for e in range(nepochs):

        # Per epoch setup ...

        for b in range(0, len(trainx), batch_size):

            pass  # Remove this line when you start implementing this
            # Train ...

        for b in range(0, len(valx), batch_size):

            pass  # Remove this line when you start implementing this
            # Val ...

        # Accumulate data...

    # Cleanup ...

    # Return results ...

    # return (training_losses, training_errors, validation_losses, validation_errors)

    raise NotImplemented
