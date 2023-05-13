# This class is not referred to the Perceptron example but tries to implement a simple neural network environment with 3 layers
import numpy as np


class Neural:
    def __init__(self, in_size, hidden_size, out_size, learning_rate=0.01):
        self.in_nodes = in_size
        self.hidden_nodes = hidden_size
        self.out_nodes = out_size

        self.lr = learning_rate

        # a matrix to represent the weights from input layer to hidden layer
        self.weights_ih = np.zeros((self.hidden_nodes, self.in_nodes))
        self.fill_weights(self.weights_ih)
        # a matrix to represent the weights from hidden layer to output layer
        self.weights_ho = np.zeros((self.out_nodes, self.hidden_nodes))
        self.fill_weights(self.weights_ho)

        # a column vector for the bias of the hidden and for the output
        self.bias_h = np.random.rand(self.hidden_nodes)
        self.bias_o = np.random.rand(self.out_nodes)


    # Feedforward algorithm is equivalent to the Perceptron guess() function -> applies matrix multiplication
    # to weights and inputs to generate the thought answer for a certain input
    def feed_forward(self, input):
        # make sure input is a numpy array
        input = np.array(input)

        # Generating the hidden layer output
        hidden = self.weights_ih @ input
        hidden += self.bias_h

        # apply activation to all elements
        hidden = self.sigmoid(hidden)

        # Next step is same application over the following layer
        output = self.weights_ho @ hidden
        output += self.bias_o

        # apply activations to all elements
        output = self.sigmoid(output)

        # output is returned as a flat array
        return output.flatten()

    # A function to apply the backpropagation algorithm and actually try to nudge the weights in proportion to the error they might generate
    # The method applies SGD algorithm as it passes a single set of inputs and targets and training happens based on those
    def train(self, input, target):
        # a fast and easy method would be to call the self.feedforward() method like this:
        # output = np.array(self.feed_forward(input))
        # set the target to be a numpy array itself

        # truth is it's much better to recall the whole function piece by piece inside here to get outputs piece by piece:

        # make sure input is a numpy array
        input = np.array(input)

        # Generating the hidden layer output
        hidden = self.weights_ih @ input
        hidden += self.bias_h

        # apply activation to all elements
        hidden = self.sigmoid(hidden)

        # Next step is same application over the following layer
        output = self.weights_ho @ hidden
        output += self.bias_o

        # apply activations to all elements
        output = self.sigmoid(output)

        #
        # --> Actual TRAINING from here <--
        #

        target = np.array(target)

        # calculate the output error
        out_error = target - output

        # defining the gradient to perform gradient descend algorithm on hidden to output weights
        gradient = self.dsigmoid(output)
        gradient *= out_error
        gradient *= self.lr

        # calculating the Delta weights from hidden to output
        delta_ho = gradient * hidden.T

        # updating the existing weights
        self.weights_ho += delta_ho

        # updating biases
        self.bias_o += gradient

        # calculate the hidden to output error -> matrix multiplication with the transpose of the weights matrix
        hidden_error = self.weights_ho.T @ out_error

        # defining the gradient to perform gradient descend algorithm on input to hidden weights
        gradient = self.dsigmoid(hidden)
        gradient *= hidden_error
        gradient *= self.lr

        # calculating the Delta weights from input to hidden
        delta_ih = gradient * input.T

        # updating the existing weights
        self.weights_ih += delta_ih

        # updating biases
        self.bias_h += gradient

    @staticmethod
    def fill_weights(matrix):
        # fills the weights with a given matrix with random values between -1 and 1
        for i in range(len(matrix)):
            for j in range(len(matrix[i])):
                matrix[i][j] = np.random.rand() * 2 - 1

    def show_weights(self):
        print(f"Input layer to Hidden layer weight matrix:")
        for i in range(len(self.weights_ih)):
            for j in range(len(self.weights_ih[i])):
                print(self.weights_ih[i][j], end="  |  ")
            print("\n")
        print(f"Hidden layer to Output layer weight matrix:")
        for i in range(len(self.weights_ho)):
            for j in range(len(self.weights_ho[i])):
                print(self.weights_ho[i][j], end="  |  ")
            print("\n")

    @staticmethod
    def sigmoid(x):
        return 1/(1 + np.exp(-x))

    @staticmethod
    def dsigmoid(x):
        return x * (1 - x)


# A test to see if network can actually learn a XOR problem

# xors is a dataset like this: (in1, in2, label)
xors = [(0,0,0), (0,1,1), (1,0,1), (1,1,0)]

# create a neural network with 2 inputs, 2 hidden nodes and 1 output
net = Neural(2, 2, 1, 0.1)

# train the network
for i in range(50000):
    j = np.random.randint(0, len(xors))
    # print the guess and the error
    # print(f"Input:{xors[j][:2]} | Guess: {net.feed_forward(xors[j][:2])} | Error: {xors[j][2] - net.feed_forward(xors[j][:2])}")
    net.train(xors[j][0:2], xors[j][2])


# print 4 tests with [0, 0], [0, 1], [1, 0], [1, 1] inputs
print(f"Input: 0,0 --> Guess: {net.feed_forward([0, 0])}")
print(f"Input: 0,1 --> Guess: {net.feed_forward([0, 1])}")
print(f"Input: 1,0 --> Guess: {net.feed_forward([1, 0])}")
print(f"Input: 1,1 --> Guess: {net.feed_forward([1, 1])}")

net.show_weights()


# a function that uses trained network to apply xor function
def xor(a, b):
    return True if net.feed_forward([a, b]) >= 0.5 else False


print(xor(0,1))
