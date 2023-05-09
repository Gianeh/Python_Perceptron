from Perceptron import Perceptron
from random import *

# datapoint format: (in1, in2, result) == (false/true, false/true, label)
andtron = Perceptron()
ortron = Perceptron()
nandtron = Perceptron()

pop = 100
generations_cap = 100
# a trivial dataset of "and" results

ands = []
for i in range(0, pop):
    input1 = randint(0, 1)
    input2 = randint(0, 1)
    label = 1 if input1 and input2 else -1
    ands.append((input1, input2, label))

ors = []
for i in range(0, pop):
    input1 = randint(0, 1)
    input2 = randint(0, 1)
    label = 1 if input1 or input2 else -1
    ors.append((input1, input2, label))

nands = []
for i in range(0, pop):
    input1 = randint(0, 1)
    input2 = randint(0, 1)
    label = 1 if not(input1 and input2) else -1
    nands.append((input1, input2, label))


def training_logic(brain, data, operator, gen_limit=100):
    current = 0
    for i in range(0, gen_limit):
        print(f"Input 1: {bool(data[current][0])}, Input 2: {bool(data[current][1])}, {operator} result is: {True if data[current][2] == 1 else False}")

        # print prediction
        guess = brain.guess(data[current][:2])
        result = True if guess == 1 else False
        print(f"Perceptron guess is: {result} and the Prediction is: {'Correct' if guess == data[current][2] else 'Wrong'}")

        # train on current and result
        brain.train(data[current][:2], data[current][2])

        current += 1
        if current == pop:
            current = 0


# training perceptron and showing results of AND, OR, NAND training
training_logic(ortron, ors, "OR")
print("\n")
training_logic(andtron, ands, "AND")
print("\n")
training_logic(nandtron, nands, "NAND")

# Showing you can't train a XOR
xortron = Perceptron()
xors = []
for i in range(0, pop):
    input1 = randint(0, 1)
    input2 = randint(0, 1)
    label = 1 if input1 ^ input2 else -1
    xors.append((input1, input2, label))

print("\nYou shall not train a XOR operator over a SINGLE PERCEPTRON!")
training_logic(xortron, xors, "XOR", 100)

# Showing that a wise combination of the above trained Perceptrons get the XOR right!
# xor = (x nand y) and (x or y)
input1 = 1
input2 = 1
print(f"\nSolution to {bool(input1)} XOR {bool(input2)} is: "
      f"{True if andtron.guess((ortron.guess((input1, input2)), nandtron.guess((input1, input2)))) == 1 else False}")


# hence you can now define a Frankenstein neural network (just 3 single neurons BTW) that solves the XOR problem! yay!
def xor(x,y):
    return True if andtron.guess((ortron.guess((x, y)), nandtron.guess((x, y)))) == 1 else False


print(xor(0,1))

# "But in python you can just use ^ operator you AS#+@*E!"... I know but this feels so much more SciFi stuff dude!

# PS: The whole above logic is clunky as Perceptron can only return -1 and 1 which bool() casting is always True,
# this could be hard coded differently inside the main class but as it's used in the other example of guessing points label...
# Imma leaving as it is...
