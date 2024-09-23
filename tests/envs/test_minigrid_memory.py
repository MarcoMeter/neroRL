import numpy as np
import pytest
from gymnasium import spaces
from neroRL.environments.minigrid_wrapper import MinigridWrapper
from neroRL.environments.minigrid_vec_wrapper import MinigridVecWrapper

@pytest.mark.parametrize("wrapper_class,kwargs", [
    (MinigridWrapper, {"env_name": "MiniGrid-MemoryS13-v0"}),
    (MinigridVecWrapper, {"env_name": "MiniGrid-MemoryS13-v0"}),
])
def test_observations_are_dicts(wrapper_class, kwargs):
    env = wrapper_class(**kwargs)
    assert isinstance(env.observation_space, spaces.Dict), "Observation space should be a spaces.Dict"
    obs, _ = env.reset()
    assert isinstance(obs, dict), "Observation returned by reset should be a dict"
    action = 0
    obs, _, _, _ = env.step(env.action_space.sample())
    assert isinstance(obs, dict), "Observation returned by step should be a dict"

@pytest.mark.parametrize("wrapper_class,kwargs", [
    (MinigridWrapper, {"env_name": "MiniGrid-MemoryS13-v0"}),
])
def test_vis_obs_bounds_and_shape(wrapper_class, kwargs):
    env = wrapper_class(**kwargs)
    
    assert "vis_obs" in env.observation_space.spaces, "Observation space should contain key 'vis_obs'"
    assert np.all(env.observation_space["vis_obs"].low == 0.0), \
        "Low value of visual observation should be 0.0"
    assert np.all(env.observation_space["vis_obs"].high == 1.0), \
        "High value of visual observation should be 1.0"
    assert env.observation_space["vis_obs"].shape == (84, 84, 3), \
        "Shape of visual observation should be (84, 84, 3)"

    obs, _ = env.reset()
    assert "vis_obs" in obs, "Observation should contain key 'vis_obs'"
    assert np.all(obs["vis_obs"] >= 0.0), \
        "Visual observation should have values >= 0.0"
    assert np.all(obs["vis_obs"] <= 1.0), \
        "Visual observation should have values <= 1.0"
    assert obs["vis_obs"].shape == (84, 84, 3), \
        "Shape of visual observation should be (84, 84, 3)"
    
    for _ in range(10):
        obs, _, _, _ = env.step(env.action_space.sample())
        assert "vis_obs" in obs, "Observation should contain key 'vis_obs'"
        assert np.all(obs["vis_obs"] >= 0.0), \
            "Visual observation should have values >= 0.0"
        assert np.all(obs["vis_obs"] <= 1.0), \
            "Visual observation should have values <= 1.0"
        assert obs["vis_obs"].shape == (84, 84, 3), \
            "Shape of visual observation should be (84, 84, 3)"

@pytest.mark.parametrize("wrapper_class,kwargs", [
    (MinigridWrapper, {"env_name": "MiniGrid-MemoryS13-v0"}),
])
def test_vis_obs_pixels(wrapper_class, kwargs):
    env = wrapper_class(**kwargs)
    obs, _ = env.reset(reset_params={"start-seed": 1, "num-seeds": 1})
    assert np.allclose(obs["vis_obs"][0, 0, :], [0.57254905, 0.57254905, 0.57254905]), "Pixel mismatch at (0, 0) on reset seed 1"
    assert np.allclose(obs["vis_obs"][40, 40, :], [0.29803923, 0.29803923, 0.29803923]), "Pixel mismatch at (40, 40) on reset seed 1"
    assert np.allclose(obs["vis_obs"][75, 40, :], [1.0, 0.29803923, 0.29803923]), "Pixel mismatch at (75, 40) on reset seed 1"

@pytest.mark.parametrize("wrapper_class,kwargs", [
    (MinigridVecWrapper, {"env_name": "MiniGrid-MemoryS13-v0"}),
])
def test_vec_obs_bounds_and_shape(wrapper_class, kwargs):
    env = wrapper_class(**kwargs)
    
    assert "vec_obs" in env.observation_space.spaces, "Observation space should contain key 'vec_obs'"
    assert np.all(env.observation_space["vec_obs"].low == 0.0), \
        "Low value of vector observation should be 0.0"
    assert np.all(env.observation_space["vec_obs"].high == 1.0), \
        "High value of vector observation should be 1.0"
    assert env.observation_space["vec_obs"].shape == (3*3*6,), \
        "Shape of vector observation should be (3*3*6,)"
    
    obs, _ = env.reset()
    assert "vec_obs" in obs, "Observation should contain key 'vec_obs'"
    assert np.all(obs["vec_obs"] >= 0.0), \
        "Vector observation should have values >= 0.0"
    assert np.all(obs["vec_obs"] <= 1.0), \
        "Vector observation should have values <= 1.0"
    assert obs["vec_obs"].shape == (3*3*6,), \
        "Shape of vector observation should be (3*3*6,)"
    
    for _ in range(10):
        obs, _, _, _ = env.step(env.action_space.sample())
        assert "vec_obs" in obs, "Observation should contain key 'vec_obs'"
        assert np.all(obs["vec_obs"] >= 0.0), \
            "Vector observation should have values >= 0.0"
        assert np.all(obs["vec_obs"] <= 1.0), \
            "Vector observation should have values <= 1.0"
        assert obs["vec_obs"].shape == (3*3*6,), \
            "Shape of vector observation should be (3*3*6,)"
        
@pytest.mark.parametrize("wrapper_class,kwargs", [
    (MinigridVecWrapper, {"env_name": "MiniGrid-MemoryS13-v0"}),
])
def test_vec_obs(wrapper_class, kwargs):
    env = wrapper_class(**kwargs)
    obs, _ = env.reset(reset_params={"start-seed": 1, "num-seeds": 1})
    expected = np.array([0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0.,
       0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.,
       1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0.])
    assert np.allclose(obs["vec_obs"], expected), "Vector observation mismatch on reset seed 1"