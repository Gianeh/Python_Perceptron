# just 2d points
from random import *
import pygame
from Utility import map_range, f


class Point:
    def __init__(self, w, h):
        self.x = random() if randint(1,2) % 2 else -random()
        self.y = random() if randint(1,2) % 2 else -random()
        self.label = 1 if self.y > f(self.x) else -1

        self.px = map_range(self.x, -1, 1, 0, w)
        self.py = map_range(self.y, -1, 1, h, 0)    # y value has to be flipped as coordinates in pygame are y-flipped

        self.radius = 5
        self.color = (0, 0, 0) if self.label == 1 else (255, 255, 255)

    def show(self, screen):
        # draw a circle at coordinates
        pygame.draw.circle(screen, self.color, (int(self.px), int(self.py)), self.radius)