import pygame
import math

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)

def main():
    # Initialize Pygame
    pygame.init()

    # Set up the display
    width, height = 800, 600
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Pendulum Simulation")

    # Pendulum properties
    length = 200
    angle = math.pi / 4 # relative to down, right is positive
    gravity = 9.81
    pivot = (width // 2, height // 4)
    bob_mass = 1
    bob_x = pivot[0] + length * math.sin(angle)
    bob_y = pivot[1] + length * math.cos(angle)
    x_velocity = 0
    y_velocity = 0
    air_resistance = 0.1

    # Time step
    dt = 1

    clock = pygame.time.Clock()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        angle = math.atan2(bob_x - pivot[0], bob_y - pivot[1])

        y_velocity += gravity * dt

        pivot_dir = ((pivot[0] - bob_x) / length, (pivot[1] - bob_y) / length)

        # Project velocity onto pivot direction to get the difference between
        # velocity and the tangent to the swing circle
        scale = x_velocity * pivot_dir[0] + y_velocity * pivot_dir[1]
        projection = (pivot_dir[0] * scale, pivot_dir[1] * scale)
        x_velocity -= projection[0]
        y_velocity -= projection[1]

        vel_magnitude = math.sqrt(x_velocity * x_velocity + y_velocity * y_velocity)
        unit_vel = (x_velocity / vel_magnitude, y_velocity / vel_magnitude)
        air_resistance_vec = (-unit_vel[0] * air_resistance, -unit_vel[1] * air_resistance)

        x_velocity += air_resistance_vec[0] * dt
        y_velocity += air_resistance_vec[1] * dt

        bob_x += x_velocity * dt
        bob_y += y_velocity * dt

        # Clear the screen
        screen.fill(WHITE)

        # Draw the pendulum
        pygame.draw.line(screen, BLACK, pivot, (int(bob_x), int(bob_y)), 2)
        pygame.draw.circle(screen, BLACK, (int(bob_x), int(bob_y)), 10)

        # Draw force and pivot

        # Update the display
        pygame.display.flip()

        # Cap the frame rate
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()
