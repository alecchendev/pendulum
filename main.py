from abc import abstractmethod
from enum import Enum
from typing import Optional, Tuple
import pygame
import math
import numpy
import torch
import torch.nn as nn
import copy

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)

GRAVITY = 9.81 / 2
AIR_RESISTANCE = 0.1

MOVE_SENSITIVITY = 8

WIDTH = 800
HEIGHT = 600

ROD_LENGTH = 200

class Mode(Enum):
    SINGLE_PLAYER = 0
    TRAINING = 1

MODE = Mode.SINGLE_PLAYER

class Pendulum:
    def __init__(self, pivot_pos: numpy.ndarray, angle: float, rod_length: int):
        self.rod_length = rod_length
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

    def create_tweaked_copy(self, min_tweak: float, max_tweak: float) -> "NeuralNetwork":
        tweaked_model = copy.deepcopy(self)
        for param in tweaked_model.parameters():
            tweak = torch.rand_like(param) * (max_tweak - min_tweak) + min_tweak
            param.data += tweak
        return tweaked_model

class NeuralNetPlayer(Player):
    def __init__(self, model: Optional[NeuralNetwork] = None):
        self.model = model or NeuralNetwork()

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

    def create_child(self, mutation_rate: float) -> "NeuralNetPlayer":
        return NeuralNetPlayer(self.model.create_tweaked_copy(-mutation_rate, mutation_rate))

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

    def draw_state(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        return (self.pendulum.pivot_pos[0], self.pendulum.pivot_pos[1]), (self.pendulum.bob_pos[0], self.pendulum.bob_pos[1])

class GameBuilder:
    def __init__(self):
        self.dt = 0.25
        self.padding = 100
        self.width = WIDTH
        self.pendulum = Pendulum(numpy.array([WIDTH // 2., HEIGHT // 2.]), 0, 200)
        self.player = None

    def with_player(self, player: Player) -> "GameBuilder":
        self.player = player
        return self
    
    def build(self) -> Game:
        assert self.player is not None
        return Game(self.dt, self.padding, self.width, self.pendulum, self.player)

def draw_pendulum(
    screen: pygame.Surface,
    color: Tuple[int, int, int],
    pivot_pos: Tuple[int, int],
    bob_pos: Tuple[int, int],
):
    pygame.draw.line(screen, color, pivot_pos, bob_pos, 2)
    pygame.draw.circle(screen, color, bob_pos, 10)

def single_player():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Pendulum Simulation")
    clock = pygame.time.Clock()

    game = GameBuilder().with_player(HumanPlayer()).build()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_1]:
            game.set_player(HumanPlayer())
        elif keys[pygame.K_2]:
            game.set_player(PhysicsPlayer())
        elif keys[pygame.K_3]:
            game.set_player(NeuralNetPlayer())

        game.step()

        screen.fill(WHITE)

        # Draw the pendulum
        pivot_pos, bob_pos = game.draw_state()
        draw_pendulum(screen, BLACK, pivot_pos, bob_pos)

        # Update the display
        pygame.display.flip()

        # Cap the frame rate
        clock.tick(60)
    pygame.quit()

def training():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Pendulum Simulation")
    clock = pygame.time.Clock()

    n_games = 64 if Mode.TRAINING else 1
    games = [GameBuilder().with_player(NeuralNetPlayer()).build() for _ in range(n_games)]
    losses = [0 for _ in range(n_games)]
    min_losses = [99999999 for _ in range(n_games)]
    n_steps = 600
    curr_steps = 0
    colors = [(255 // n_games * i, 0, 255) for i in range(n_games)]

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        for i, game in enumerate(games):
            game.step()
            top = HEIGHT // 2. - ROD_LENGTH
            distance = abs(game.pendulum.bob_pos[1] - top)
            losses[i] += distance
            min_losses[i] = min(min_losses[i], distance)

        # Clear the screen
        screen.fill(WHITE)

        # Draw the pendulum
        for color, (pivot_pos, bob_pos) in zip(colors, [game.draw_state() for game in games]):
            draw_pendulum(screen, color, pivot_pos, bob_pos)

        # Update the display
        pygame.display.flip()

        # Cap the frame rate
        clock.tick(60)

        # Do genetic evolution
        if curr_steps == n_steps:
            # Reward them for their min distance as well as total loss
            max_loss = n_steps * ROD_LENGTH * 2
            fitnesses = [max_loss * max_loss / (10 * (min_loss * 10 + loss)) for min_loss, loss in zip(min_losses, losses)]
            print(min(losses), max(fitnesses))
            new_games = []
            # Take the top 25% and make 3 new games from them: themselves and 2 children
            for _fitness, game in sorted(zip(fitnesses, games), key = lambda x: x[0], reverse=True)[:n_games//4]:
                new_games.append(GameBuilder().with_player(game.player).build())
                assert isinstance(game.player, NeuralNetPlayer)
                new_games.extend([GameBuilder().with_player(game.player.create_child(0.05)).build() for _ in range(2)])
            n_children_total = min(int(0.75 * n_games), len(new_games))
            new_games = new_games[:n_children_total]
            n_randos = n_games - n_children_total
            random_games = [GameBuilder().with_player(NeuralNetPlayer()).build() for _ in range(n_randos)]
            new_games.extend(random_games)
            print("n_children_total", n_children_total, "n_randos", n_randos, "len(new_games)", len(new_games))
            games = new_games
            losses = [0 for _ in range(n_games)]
            curr_steps = 0
        else:
            curr_steps += 1
    pygame.quit()

def main():
    if MODE == Mode.SINGLE_PLAYER:
        single_player()
    elif MODE == Mode.TRAINING:
        training()
    else:
        assert False, "Invalid mode"

if __name__ == "__main__":
    main()
