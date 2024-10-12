from typing import Tuple
import pygame
import math
import numpy

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)

GRAVITY = 9.81
AIR_RESISTANCE = 0.25

class Pendulum:
    def __init__(self, screen_dims: Tuple[int, int]):
        self.rod_length = 200
        self.angle = math.pi / 4 # relative to down, right is positive
        width, height = screen_dims
        self.pivot_pos = numpy.array([width // 2., height // 2.])
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
        if numpy.sqrt(self.bob_vel.dot(self.bob_vel)) < 0.01:
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

def main():
    # Initialize Pygame
    pygame.init()

    # Set up the display
    width, height = 800, 600
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Pendulum Simulation")

    # Time step
    dt = 1

    clock = pygame.time.Clock()

    pendulum = Pendulum((width, height))

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

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
