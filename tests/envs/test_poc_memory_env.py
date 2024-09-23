import numpy as np
import pytest
from gymnasium import spaces
from neroRL.environments.poc_memory_env_wrapper import PocMemoryEnvWrapper

@pytest.mark.parametrize("wrapper_class,kwargs", [
    (PocMemoryEnvWrapper, {}),
])
def test_observations_are_dicts(wrapper_class, kwargs):
    env = wrapper_class(**kwargs)
    assert isinstance(env.observation_space, spaces.Dict), "Observation space should be a spaces.Dict"
    assert "vec_obs" in env.observation_space.spaces, "Observation space should contain key 'vec_obs'"
    assert np.allclose(env.observation_space.spaces["vec_obs"].low, -1.0), "Low of vec_obs should be -1.0"
    assert np.allclose(env.observation_space.spaces["vec_obs"].high, 1.0), "High of vec_obs should be 1.0"
    obs, _ = env.reset()
    assert isinstance(obs, dict), "Observation returned by reset should be a dict"
    assert "vec_obs" in obs, "Observation should contain key 'vec_obs'"
    action = 0
    obs, _, _, _ = env.step(env.action_space.sample())
    assert isinstance(obs, dict), "Observation returned by step should be a dict"
    assert "vec_obs" in obs, "Observation should contain key 'vec_obs'"

@pytest.mark.parametrize("wrapper_class,kwargs", [
    (PocMemoryEnvWrapper, {}),
])
def test_vec_obs(wrapper_class, kwargs):
    reset_params = {"start-seed": 1, "num-seeds": 1}
    env = wrapper_class(**kwargs)
    obs, _ = env.reset(reset_params)
    assert np.allclose(obs["vec_obs"], [-1.,  0.,  1.]), "Initial observation should be [-1.,  0.,  1.]"
    for i in range(5):
        obs, _, _, _ = env.step(1)
    assert np.allclose(obs["vec_obs"], [0., 0.6, 0.]), "Observation should be [0., 0.6, 0.]"