# A class that represents the perceptron
from random import *
import math


class Perceptron:
    def __init__(self, n=2, lr=0.1, unbiased=False):
        self.weights = []   # a random list of weights
        self.rand_init(n)
        self.lr = lr

        self.bias = 1 if not unbiased else 0
        self.bias_weight = random() if randint(1, 2) % 2 else -random()     # one more randomized weight for the bias

    def rand_init(self, n):
        for i in range(0, n):
            self.weights.append(random() if randint(1, 2) % 2 else -random())

    def guess(self, inputs):
        if len(inputs) != len(self.weights):
            print("Wrong format for given input, doesn't match number of input neurons")
            return 0
        result = 0
        for i in range(0, len(inputs)):
            result += self.weights[i]*inputs[i]
        # also consider the bias
        result += self.bias * self.bias_weight
        return self.activation(result)

    def guessY(self, x):
        return -(self.bias_weight/self.weights[1]) - (self.weights[0]/self.weights[1]) * x

    @staticmethod
    def activation(x):
        # base class implements a sign activation function
        return 1 if x >= 0 else -1

    def train(self, inputs, target):
        guess = self.guess(inputs)
        error = target - guess

        for i in range(0, len(inputs)):
            self.weights[i] += error * inputs[i] * self.lr
        # also train bias weight
        self.bias_weight += error * self.bias * self.lr
