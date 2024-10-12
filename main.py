from typing import Tuple
import pygame
import math
import numpy

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)

GRAVITY = 9.81 / 2
AIR_RESISTANCE = 0.2

MOVE_SENSITIVITY = 8

class Pendulum:
    def __init__(self, pivot_pos: numpy.ndarray, angle: float):
        self.rod_length = 200
        # relative to down, right is positive
        self.angle = angle
        self.pivot_pos = pivot_pos
        rod = numpy.array([math.sin(self.angle), math.cos(self.angle)]) * self.rod_length
        self.bob_pos = self.pivot_pos + rod
        self.bob_vel = numpy.array([0., 0.])

    def step(self, dt: float):
        rod_to_bob = self.bob_pos - self.pivot_pos # points from pivot to bob
        self.angle = math.atan2(rod_to_bob[0], rod_to_bob[1])

        self.bob_vel += numpy.array([0, GRAVITY * dt])

        # Air resistance
        vel_norm = self.bob_vel / numpy.sqrt(self.bob_vel.dot(self.bob_vel))
        air_resistance_acc = vel_norm * -AIR_RESISTANCE
        self.bob_vel += air_resistance_acc * dt

        # Rod force
        # Project velocity onto pivot direction to get the difference between
        # velocity and the tangent to the swing circle
        rod_to_pivot = self.pivot_pos - self.bob_pos
        pivot_dir = rod_to_pivot / self.rod_length

        scale = self.bob_vel.dot(pivot_dir)
        projection = pivot_dir * scale
        self.bob_vel -= projection

        # Zero it out if it's really small
        if numpy.sqrt(self.bob_vel.dot(self.bob_vel)) < 0.001:
            self.bob_vel = numpy.array([0., 0.])

        self.bob_pos += self.bob_vel * dt

        # To make things more precise, we also fix things so the
        # rod length doesn't slowly grow (velocity being the tangent
        # to the circle means on any time step that's not infinitely
        # small, the diameter will continuously grow.)
        rod_to_pivot = self.pivot_pos - self.bob_pos
        length = numpy.sqrt(rod_to_pivot.dot(rod_to_pivot))
        rod_to_pivot_norm = rod_to_pivot / length
        diff = length - self.rod_length
        self.bob_pos += rod_to_pivot_norm * diff

    def move_left(self, dt: float):
        self.pivot_pos += numpy.array([-1., 0.]) * dt * MOVE_SENSITIVITY

    def move_right(self, dt: float):
        self.pivot_pos += numpy.array([1., 0.]) * dt * MOVE_SENSITIVITY

    def move_to_horizontal(self, x_pos: float):
        self.pivot_pos[0] = x_pos

    def enforce_bounds(self, top_left: numpy.ndarray, bottom_right: numpy.ndarray):
        self.pivot_pos[0] = max(self.pivot_pos[0], top_left[0])
        self.pivot_pos[1] = max(self.pivot_pos[1], top_left[1])
        self.pivot_pos[0] = min(self.pivot_pos[0], bottom_right[0])
        self.pivot_pos[1] = min(self.pivot_pos[1], bottom_right[1])

def main():
    # Initialize Pygame
    pygame.init()

    # Set up the display
    width, height = 800, 600
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Pendulum Simulation")

    # Time step
    dt = 0.25

    clock = pygame.time.Clock()

    pendulum = Pendulum(numpy.array([width // 2., height // 4.]), 1 * math.pi / 4)
    prev_mouse_x = 0

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            pendulum.move_left(dt)
        if keys[pygame.K_RIGHT]:
            pendulum.move_right(dt)

        # There's a bug in pygame it seems
        # where sometimes right as your
        # mouse goes off screen you get a much higher
        # value randomly.
        mouse_x, _ = pygame.mouse.get_pos()
        if prev_mouse_x != mouse_x:
            pendulum.move_to_horizontal(mouse_x)
            prev_mouse_x = mouse_x

        padding = 100
        pendulum.enforce_bounds(numpy.array([padding, padding]), numpy.array([width - padding, height - padding]))
        pendulum.step(dt)

        pivot_pos = (pendulum.pivot_pos[0], pendulum.pivot_pos[1])
        bob_pos = (pendulum.bob_pos[0], pendulum.bob_pos[1])

        # Clear the screen
        screen.fill(WHITE)

        # Draw the pendulum
        pygame.draw.line(screen, BLACK, pivot_pos, bob_pos, 2)
        pygame.draw.circle(screen, BLACK, bob_pos, 10)

        # Draw force and pivot

        # Update the display
        pygame.display.flip()

        # Cap the frame rate
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()
