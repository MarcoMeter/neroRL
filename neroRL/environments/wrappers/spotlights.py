import math
import numpy as np
import pygame
import os

from neroRL.environments.wrappers.pygame_assets import Spotlight

from gym import spaces
from neroRL.environments.env import Env

class SpotlightsEnv(Env):
    """This wrapper adds wandering spotlights to render on top of a visual observation as noise perturbation."""

    def __init__(self, env):
        """Initializes the wrapper by settting up everything for the spotlights and the PyGame surfaces.
        PyGame blits the spotlight surface onto the visual observation.
        
        Arguments:
            env {Env} -- The to be wrapped environment, which is derived from the Env class
        """
        self._env = env

        # Check if visual observations are available
        assert (self._env.visual_observation_space is not None), "Visual observations of the environment have to be available."
        
        # Spotlights members
        self.initial_spawns = 4
        self.num_spawns = 30
        self.initial_spawn_interval = 30
        self.spawn_interval_threshold = 10
        self.spawn_interval_decay = 0.95
        self.spot_min_radius = 7.5
        self.spot_max_radius = 13.75
        self.spot_min_speed = 0.0025
        self.spot_max_speed = 0.0075

        # PyGame Setup
        os.putenv('SDL_VIDEODRIVER', 'fbcon')
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        self.screen_dim = 84
        self.screen = pygame.display.set_mode((self.screen_dim, self.screen_dim), pygame.NOFRAME)
        self.clock = pygame.time.Clock()
        pygame.event.set_allowed(None)
        self.spotlight_surface = pygame.Surface((self.screen_dim, self.screen_dim))
        self.spotlight_surface.fill((255, 0, 0))
        self.spotlight_surface.set_colorkey((255, 0, 0))
        self.spotlight_surface.set_alpha(125)

        self.np_random = np.random.Generator(np.random.PCG64(0))

    @property
    def unwrapped(self):
        """Return this environment in its vanilla (i.e. unwrapped) state."""
        return self._env.unwrapped

    @property
    def visual_observation_space(self):
        """Returns the shape of the visual component of the observation space as a tuple."""
        return self._env._visual_observation_space

    @property
    def vector_observation_space(self):
        """Returns the shape of the vector component of the observation space as a tuple."""
        return self._env.vector_observation_space

    @property
    def action_space(self):
        """Returns the shape of the action space of the agent."""
        return self._env.action_space

    @property
    def action_names(self):
        """Returns a list of action names. It has to be noted that only the names of action branches are provided and not the actions themselves!"""
        return self._env.action_names

    @property
    def get_episode_trajectory(self):
        """Returns the trajectory of an entire episode as dictionary (vis_obs, vec_obs, rewards, actions). 
        """
        trajectory = self._env.get_episode_trajectory
        trajectory["vis_obs"] = self._obs
        return trajectory

    def reset(self, reset_params = None):
        """Reset the environment. The provided config is a dictionary featuring reset parameters of the environment such as the seed.
        
        Keyword Arguments:
            reset_params {dict} -- Reset parameters to configure the environment (default: {None})
        
        Returns:
            {numpy.ndarray} -- Resized visual observation
            {numpy.ndarray} -- Vector observation
        """
        vis_obs, vec_obs = self._env.reset(reset_params = reset_params)
        self._obs = []

        # Setup spotlights
        self.np_random = np.random.Generator(np.random.PCG64(self._env.unwrapped.seed))
        self.spawn_intervals = self._compute_spawn_intervals()
        self.spotlights = []
        self.spawn_timer = self.spawn_intervals[0] if self.spawn_intervals else 0 # ensure that the first spotlight is spawned right away
        for _ in range(self.initial_spawns):
            self.spotlights.append(Spotlight(self.screen_dim, self.np_random.integers(int(self.spot_min_radius), int(self.spot_max_radius + 1)),
                                                            self.np_random.uniform(self.spot_min_speed, self.spot_max_speed), self.np_random, t=0.3))

        # Use pygame to add the spotlights onto the original observation
        obs_surface = pygame.surfarray.make_surface((vis_obs * 255.0))
        self.screen.blit(obs_surface, (0, 0))
        self.screen.blit(self.spotlight_surface, (0, 0))
        vis_obs = pygame.surfarray.array3d(pygame.display.get_surface()).astype(np.float32) / 255.0 # TODO check vis_obs type and scale
        self._obs.append((vis_obs * 255).astype(np.uint8))
        return vis_obs, vec_obs

    def step(self, action):
        """Executes one step of the agent.
        
        Arguments:
            action {List} -- A list of at least one discrete action to be executed by the agent
        
        Returns:
            {numpy.ndarray} -- Stacked visual observation
            {numpy.ndarray} -- Stacked vector observation
            {float} -- Scalar reward signaled by the environment
            {bool} -- Whether the episode of the environment terminated
            {dict} -- Further episode information retrieved from the environment
        """
        vis_obs, vec_obs, reward, done, info = self._env.step(action)
        
        # Spawn spotlights
        self.spawn_timer += 1
        if self.spawn_intervals:
            if self.spawn_timer >= self.spawn_intervals[0]:
                self.spotlights.append(Spotlight(self.screen_dim, self.np_random.integers(self.spot_min_radius, self.spot_max_radius + 1),
                                                            self.np_random.uniform(self.spot_min_speed, self.spot_max_speed), self.np_random))
                self.spawn_intervals.pop()
                self.spawn_timer = 0
        
        # Draw spotlights
        self.spotlight_surface.fill((255,0,0))
        for spot in self.spotlights:        
            # Remove spotlights that finished traversal
            if spot.done:
                self.spotlights.remove(spot)
            else:
                spot.draw(self.spotlight_surface)

        # Use pygame to add the spotlights onto the original observation
        obs_surface = pygame.surfarray.make_surface((vis_obs * 255.0))
        self.screen.blit(obs_surface, (0, 0))
        self.screen.blit(self.spotlight_surface, (0, 0))
        vis_obs = pygame.surfarray.array3d(pygame.display.get_surface()).astype(np.float32) / 255.0 # TODO check vis_obs type and scale
        self._obs.append((vis_obs * 255).astype(np.uint8))

        # import matplotlib.pyplot as plt
        # plt.imshow(vis_obs)
        # plt.show()

        return vis_obs, vec_obs, reward, done, info

    def close(self):
        """Shuts down the environment."""
        self._env.close()

    def _compute_spawn_intervals(self) -> list:
        intervals = []
        initial =self.initial_spawn_interval
        for i in range(self.num_spawns):
            intervals.append(int(initial + self.spawn_interval_threshold))
            initial = initial * math.pow(self.spawn_interval_decay, 1)
        return intervals