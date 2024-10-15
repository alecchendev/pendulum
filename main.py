from abc import abstractmethod
from typing import Tuple
import pygame
import math
import numpy
import torch
import torch.nn as nn

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)

GRAVITY = 9.81 / 2
AIR_RESISTANCE = 0.1

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

    def move_left(self, sensitivity: float):
        self.move_horizontal(-sensitivity)

    def move_right(self, sensitivity: float):
        self.move_horizontal(sensitivity)

    def move_horizontal(self, amount: float):
        self.pivot_pos += numpy.array([1., 0.]) * amount

    def move_to_horizontal(self, x_pos: float):
        self.pivot_pos[0] = x_pos

    def enforce_bounds(self, min_x: int, max_x: int):
        self.pivot_pos[0] = max(self.pivot_pos[0], min_x)
        self.pivot_pos[0] = min(self.pivot_pos[0], max_x)

class Player:
    @abstractmethod
    def play(self, pendulum: Pendulum, settings: Tuple[float, int, int]):
        raise NotImplementedError()

class HumanPlayer(Player):
    def __init__(self):
        self.prev_mouse_x = 0

    def play(self, pendulum: Pendulum, settings: Tuple[float, int, int]):
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            pendulum.move_left(MOVE_SENSITIVITY)
        if keys[pygame.K_RIGHT]:
            pendulum.move_right(MOVE_SENSITIVITY)

        # There's a bug in pygame it seems
        # where sometimes right as your
        # mouse goes off screen you get a much higher
        # value randomly.
        mouse_x, _ = pygame.mouse.get_pos()
        if self.prev_mouse_x != mouse_x:
            pendulum.move_to_horizontal(mouse_x)
            self.prev_mouse_x = mouse_x

class PhysicsPlayer(Player):
    def __init__(self):
        self.doing_a_thing_countdown = 0
        self.waiting = False
        self.moving_right = False

    def play(self, pendulum: Pendulum, settings: Tuple[float, int, int]):
        if self.doing_a_thing_countdown > 0:
            if self.waiting:
                pendulum.move_left(MOVE_SENSITIVITY)
            elif self.moving_right:
                pendulum.move_right(MOVE_SENSITIVITY)
            self.doing_a_thing_countdown -= 1
            return
        if self.doing_a_thing_countdown == 0:
            self.waiting = False
            self.moving_right = False
        if pendulum.angle < math.pi / 2 and pendulum.angle > -math.pi / 2:
            # if it's swinging, move left and wait
            if numpy.sqrt(pendulum.bob_vel.dot(pendulum.bob_vel)) > 0.001:
                self.doing_a_thing_countdown = 100
                self.waiting = True
            # if it's still, do a new thing
            else:
                self.doing_a_thing_countdown = 100
                self.moving_right = True
            return
        # This works if it's at the right angle already in the air,
        # the other stuff just does a thing, doesn't really work.
        if (pendulum.angle > math.pi / 2):
            pendulum.move_right(MOVE_SENSITIVITY)
        if (pendulum.angle < -math.pi / 2):
            pendulum.move_left(MOVE_SENSITIVITY)

# input: pivot pos, bob pos, bob vel (3x 2d vectors -> 6 floats)
# output: move left, move right, do nothing (3 floats)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_stack = nn.Sequential(
            nn.Linear(9, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 3),
        )

    def forward(self, x):
        logits = self.linear_stack(x)
        return logits

class NeuralNetPlayer(Player):
    def __init__(self):
        self.model = NeuralNetwork()

    def play(self, pendulum: Pendulum, settings: Tuple[float, int, int]):
        pivot_pos_torch = torch.tensor(pendulum.pivot_pos, dtype=torch.float32)
        bob_pos_torch = torch.tensor(pendulum.bob_pos, dtype=torch.float32)
        bob_vel_torch = torch.tensor(pendulum.bob_vel, dtype=torch.float32)
        settings_torch = torch.tensor(list(settings), dtype=torch.float32)
        input_vec = torch.cat((pivot_pos_torch, bob_pos_torch, bob_vel_torch, settings_torch))
        logits = self.model(input_vec)
        action = torch.argmax(logits)
        if action == 0:
            pendulum.move_left(MOVE_SENSITIVITY)
        elif action == 1:
            pendulum.move_right(MOVE_SENSITIVITY)
        # if action == 2, do nothing

class Game:
    def __init__(self, dt: float, padding: int, width: int, pendulum: Pendulum, player: Player):
        self.dt = dt
        self.min_x = padding
        self.max_x = width - padding
        self.pendulum = pendulum
        self.player = player

    def step(self):
        self.player.play(self.pendulum, self.settings())
        self.pendulum.enforce_bounds(self.min_x, self.max_x)
        self.pendulum.step(self.dt)

    def set_player(self, player: Player):
        self.player = player

    def settings(self) -> Tuple[float, int, int]:
        return self.dt, self.min_x, self.max_x

def main():
    # Initialize Pygame
    pygame.init()

    # Set up the display
    width, height = 800, 600
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Pendulum Simulation")

    clock = pygame.time.Clock()

    # pendulum = Pendulum(numpy.array([width // 2., height // 2.]), 6 * math.pi / 8)
    # player = NeuralNetPlayer()
    # game = Game(0.25, 100, width, pendulum, player)
    games = [Game(0.25, 100, width, Pendulum(numpy.array([width // 2., height // 2.]), 2 * math.pi / 10 * i), NeuralNetPlayer()) for i in range(10)]

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # keys = pygame.key.get_pressed()
        # if keys[pygame.K_1]:
        #     game.set_player(HumanPlayer())
        # elif keys[pygame.K_2]:
        #     game.set_player(PhysicsPlayer())
        # elif keys[pygame.K_3]:
        #     game.set_player(NeuralNetPlayer())
        #
        # game.step()
        #
        # pendulum = game.pendulum

        for game in games:
            game.step()

        # Clear the screen
        screen.fill(WHITE)

        # Draw the pendulum
        for i, game in enumerate(games):
            pivot_pos = (game.pendulum.pivot_pos[0], game.pendulum.pivot_pos[1])
            bob_pos = (game.pendulum.bob_pos[0], game.pendulum.bob_pos[1])
            color = (255 // len(games) * i, 0, 255)
            pygame.draw.line(screen, color, pivot_pos, bob_pos, 2)
            pygame.draw.circle(screen, color, bob_pos, 10)

        # Update the display
        pygame.display.flip()

        # Cap the frame rate
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()
