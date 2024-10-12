import pygame
import math

# Initialize Pygame
pygame.init()

# Set up the display
width, height = 800, 600
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Pendulum Simulation")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Pendulum properties
length = 200
angle = math.pi / 4
angular_velocity = 0
gravity = 9.81
pivot = (width // 2, height // 4)

# Time step
dt = 0.1

clock = pygame.time.Clock()

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Calculate angular acceleration
    angular_acceleration = -gravity / length * math.sin(angle)

    # Update angular velocity and angle
    angular_velocity += angular_acceleration * dt
    angle += angular_velocity * dt

    # Calculate bob position
    bob_x = pivot[0] + length * math.sin(angle)
    bob_y = pivot[1] + length * math.cos(angle)

    # Clear the screen
    screen.fill(WHITE)

    # Draw the pendulum
    pygame.draw.line(screen, BLACK, pivot, (int(bob_x), int(bob_y)), 2)
    pygame.draw.circle(screen, BLACK, (int(bob_x), int(bob_y)), 10)

    # Update the display
    pygame.display.flip()

    # Cap the frame rate
    clock.tick(60)

pygame.quit()
