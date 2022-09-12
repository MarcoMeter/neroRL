import math
import numpy as np
import pygame

from pygame.math import Vector2

class Spotlight():
    def __init__(self, dim, radius, speed, rng, t = 0) -> None:
        self.speed = speed
        self.t = t
        self.done = False

        # Center of the screen
        center = (dim // 2, dim // 2)
        # Length of the diagonal of the screen
        diagonal = math.sqrt(math.pow(dim, 2) + math.pow(dim, 2))
        # Determine final spawn radius to ensure that spotlights are not visible upon spawning
        self.spawn_radius = diagonal / 2 + radius
        self.radius = radius

        # Sample angles for start, end and offset position
        start_angle = rng.integers(0, 360)
        inverted_angle = start_angle + 180
        target_angle = inverted_angle + rng.integers(-45, 45)
        offset_angle = target_angle + rng.integers(-135, 135)

        # Calculate the start position by the sampled angle
        # Code variant A
        # x = spawn_radius * math.cos(math.radians(angle)) + 336 // 2
        # y = spawn_radius * math.sin(math.radians(angle)) + 336 // 2
        # self.start_position = (int(x), int(y))
        # Code variant B
        self.spawn_location = center + Vector2(self.spawn_radius, 0).rotate(start_angle)
        self.current_location = self.spawn_location
        # Calculate target location
        self.target_location = center + Vector2(self.spawn_radius, 0).rotate(target_angle)
        # Calculate offset location
        self.offset_location = center + Vector2(self.spawn_radius, 0).rotate(offset_angle)

    def draw(self, surface):
        lerp_target = self.target_location.lerp(self.offset_location, self.t)
        self.current_location = self.spawn_location.lerp(lerp_target, self.t)
        pygame.draw.circle(surface, (0, 0, 0), (int(self.current_location.x), int(self.current_location.y)), self.radius)
        self.t += self.speed
        if self.t >= 1.0:
            self.t = 1.0
            self.done = True
