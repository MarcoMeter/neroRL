import pytest
import numpy as np
from gymnasium import spaces
from neroRL.environments.wrappers.stacked_observation import StackedObservationEnv
from tests.envs.mock_env import MockEnv

@pytest.fixture
def mock_env():
    env = MockEnv()
    return env

@pytest.fixture
def stacked_env(mock_env):
    return StackedObservationEnv(mock_env, num_stacks=4)

def test_wrapped_env_observation_space_unchanged(mock_env, stacked_env):
    assert mock_env.observation_space == stacked_env._env.observation_space, "Wrapped environment's observation space should remain unchanged"
    assert stacked_env.observation_space is not mock_env.observation_space, "Wrapped environment's observation space should be a new object"

def test_observation_space_stacked_correctly(stacked_env):
    obs_space = stacked_env.observation_space

    assert 'vis_obs' in obs_space.spaces, "'vis_obs' should be in the observation space"
    expected_image_shape = (64, 64, 12)  # 3 channels * 4 stacks
    assert obs_space.spaces['vis_obs'].shape == expected_image_shape, \
        f"'vis_obs' should have shape {expected_image_shape}, got {obs_space.spaces['vis_obs'].shape}"
    
    assert 'vec_obs' in obs_space.spaces, "'vec_obs' should be in the observation space"
    expected_vector_shape = (40,)  # 10 features * 4 stacks
    assert obs_space.spaces['vec_obs'].shape == expected_vector_shape, \
        f"'vec_obs' should have shape {expected_vector_shape}, got {obs_space.spaces['vec_obs'].shape}"

def test_initialization(stacked_env):
    obs, _ = stacked_env.reset()
    
    image_stack = obs['vis_obs']
    assert image_stack.shape == (64, 64, 12), f"Stacked 'vis_obs' shape should be (64, 64, 12), got {image_stack.shape}"
    expected_image = np.zeros((64, 64, 12), dtype=np.uint8)
    np.testing.assert_array_equal(image_stack, expected_image, 
                                  err_msg="'vis_obs' stack should be initialized to zeros after reset")
    
    vector_stack = obs['vec_obs']
    assert vector_stack.shape == (40,), f"Stacked 'vec_obs' shape should be (40,), got {vector_stack.shape}"
    expected_vector = np.zeros(40, dtype=np.float32)
    np.testing.assert_array_equal(vector_stack, expected_vector, 
                                  err_msg="'vec_obs' stack should be initialized to zeros after reset")

def test_stacked_observation_order_correctness(stacked_env):
    """Test that the stacked observations maintain the correct order over multiple steps."""
    obs, _ = stacked_env.reset()
    
    # Initial stack should have all observations as t=0
    expected_image = np.zeros((64, 64, 12), dtype=np.uint8)
    expected_vector = np.zeros(40, dtype=np.float32)
    np.testing.assert_array_equal(obs['vis_obs'], expected_image, 
                                  err_msg="'vis_obs' stack should be initialized to zeros after reset")
    np.testing.assert_array_equal(obs['vec_obs'], expected_vector, 
                                  err_msg="'vec_obs' stack should be initialized to zeros after reset")
    
    # Perform 5 steps
    for t in range(5):
        obs, reward, done, info = stacked_env.step(0)
        if done:
            break

    # After 5 steps, stack should contain steps 2,3,4,5
    # Assuming MockEnv's 'vis_obs' is filled with t and 'vec_obs' is filled with t
    expected_image = np.concatenate([
        np.full((64, 64, 3), t, dtype=np.uint8) for t in range(2, 6)
    ], axis=2)
    expected_vector = np.concatenate([
        np.full(10, t, dtype=np.float32) for t in range(2, 6)
    ], axis=0)
    
    assert obs['vis_obs'].shape == (64, 64, 12), f"Stacked 'vis_obs' shape should be (64, 64, 12), got {obs['vis_obs'].shape}"
    np.testing.assert_array_equal(obs['vis_obs'], expected_image, 
                                  err_msg="'vis_obs' stack does not have the correct stacked order")
    
    assert obs['vec_obs'].shape == (40,), f"Stacked 'vec_obs' shape should be (40,), got {obs['vec_obs'].shape}"
    np.testing.assert_array_equal(obs['vec_obs'], expected_vector, 
                                  err_msg="'vec_obs' stack does not have the correct stacked order")
