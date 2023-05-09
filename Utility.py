import pygame


# utility function to close the window on the quit button
def on_close():
    for event in pygame.event.get():
        # quit if the quit button is pressed
        if event.type == pygame.QUIT:
            pygame.quit()
            quit()


# a function to draw a line corresponding to f function, which is also used in Point class to distinguish labels
def draw_line(screen, w, h, func, color=(255, 0, 0)):
    x1 = map_range(-1, -1, 1, 0, w)
    y1 = map_range(func(-1), -1, 1, h, 0)
    x2 = map_range(1, -1, 1, 0, w)
    y2 = map_range(func(1), -1, 1, h, 0)
    pygame.draw.line(screen, color, (x1, y1), (x2, y2), width=1)
    # print(f"x1={x1}, y1={y1}, x2={x2}, y2={y2}, using function {func.__name__}")


def train_on_click(points, brain):
    if pygame.mouse.get_pressed()[0]:
        for p in points:
            inputs = [p.x, p.y]
            brain.train(inputs, p.label)


# a function that maps a range to another
def map_range(x, current_lower, current_upper, target_lower, target_upper):
    return (x - current_lower) * (target_upper - target_lower) / (current_upper - current_lower) + target_lower


# an arbitrary function to create the dataset
def f(x):
    return 0.2*x + 0.1
