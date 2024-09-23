import numpy as np
import pytest
from gymnasium import spaces
from neroRL.environments.memory_gym_wrapper import MemoryGymWrapper

@pytest.mark.parametrize("wrapper_class,kwargs", [
    (MemoryGymWrapper, {"env_name": "SearingSpotlights-v0"}),
])
def test_observations_are_dicts(wrapper_class, kwargs):
    env = wrapper_class(**kwargs)
    assert isinstance(env.observation_space, spaces.Dict), "Observation space should be a spaces.Dict"
    assert "visual_observation" in env.observation_space.spaces, "Observation space should contain key 'visual_observation'"
    obs, _ = env.reset()
    assert isinstance(obs, dict), "Observation returned by reset should be a dict"
    assert "visual_observation" in obs, "Observation should contain key 'visual_observation'"
    action = 0
    obs, _, _, _ = env.step(env.action_space.sample())
    assert isinstance(obs, dict), "Observation returned by step should be a dict"
    assert "visual_observation" in obs, "Observation should contain key 'visual_observation'"


@pytest.mark.parametrize("wrapper_class,kwargs", [
    (MemoryGymWrapper, {"env_name": "SearingSpotlights-v0"}),
])
def test_vis_obs_bounds_and_shape(wrapper_class, kwargs):
    env = wrapper_class(**kwargs)
    
    assert np.all(env.observation_space["visual_observation"].low == 0.0), \
        "Low value of visual observation should be 0.0"
    assert np.all(env.observation_space["visual_observation"].high == 1.0), \
        "High value of visual observation should be 1.0"
    assert env.observation_space["visual_observation"].shape == (84, 84, 3), \
        "Shape of visual observation should be (84, 84, 3)"
    
    obs, _ = env.reset()
    assert np.all(obs["visual_observation"] >= 0.0), \
        "Visual observation should have values >= 0.0"
    assert np.all(obs["visual_observation"] <= 1.0), \
        "Visual observation should have values <= 1.0"
    assert obs["visual_observation"].shape == (84, 84, 3), \
        "Shape of visual observation should be (84, 84, 3)"
    
    for _ in range(10):
        obs, _, _, _ = env.step(env.action_space.sample())
        assert np.all(obs["visual_observation"] >= 0.0), \
            "Visual observation should have values >= 0.0"
        assert np.all(obs["visual_observation"] <= 1.0), \
            "Visual observation should have values <= 1.0"
        assert obs["visual_observation"].shape == (84, 84, 3), \
            "Shape of visual observation should be (84, 84, 3)"