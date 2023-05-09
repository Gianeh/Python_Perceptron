import pygame
from Point import Point
from Perceptron import Perceptron
from Utility import *

# a main file where the actual graphics are shown
W = 1200
H = 1200
FPS = 1000
# initialize pygame
pygame.init()
screen = pygame.display.set_mode((W, H))
clock = pygame.time.Clock()


# Data to be fed to the perceptron
pop = 1500
points = []
for i in range(0, pop):
    points.append(Point(W, H))

# Declaring the actual perceptron
brain = Perceptron(n=2, lr=0.01)

# current trained point
current = 0

# Draw loop
while True:
    # set the FPS
    clock.tick(FPS)
    # check for events
    on_close()
    screen.fill((120, 120, 120))
    # draw stuff
    for p in points:
        p.show(screen)

    # Training the brain and visualizing the guess
    for p in points:
        inputs = [p.x, p.y]

        if brain.guess(inputs) == p.label:
            pygame.draw.circle(screen, (0, 255, 0), (p.px, p.py), p.radius/2)
        else:
            pygame.draw.circle(screen, (255, 0, 0), (p.px, p.py), p.radius/2)

    # Train on mouse click
    train_on_click(points, brain)

    # train one point per frame

    brain.train((points[current].x, points[current].y), points[current].label)

    current += 1
    if current == len(points):
        current = 0

    # draw division line
    draw_line(screen, W, H, f)
    # draw the line that brain thinks is dividing the dataset into two, considering bias weight is outside of weight vector
    draw_line(screen, W, H, brain.guessY, (0, 255, 0))

    # update the screen
    pygame.display.update()
