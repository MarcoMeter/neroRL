import pytest
import numpy as np
from gymnasium import spaces
from neroRL.environments.wrappers.grayscale_visual_observation import GrayscaleVisualObsEnv
from tests.envs.mock_env import MockEnv

@pytest.fixture
def mock_env():
    return MockEnv()

@pytest.fixture
def grayscale_env(mock_env):
    return GrayscaleVisualObsEnv(mock_env)

def test_observation_space_grayscale_correctly_modified(grayscale_env):
    obs_space = grayscale_env.observation_space
    
    # Check 'vis_obs' space
    assert 'vis_obs' in obs_space.spaces, "'vis_obs' should be in the observation space"
    expected_image_shape = (64, 64, 1)  # Grayscale has 1 channel
    assert obs_space.spaces['vis_obs'].shape == expected_image_shape, \
        f"'vis_obs' should have shape {expected_image_shape}, got {obs_space.spaces['vis_obs'].shape}"
    
    # Check 'vec_obs' space remains unchanged
    assert 'vec_obs' in obs_space.spaces, "'vec_obs' should be in the observation space"
    expected_vector_shape = (10,)
    assert obs_space.spaces['vec_obs'].shape == expected_vector_shape, \
        f"'vec_obs' should have shape {expected_vector_shape}, got {obs_space.spaces['vec_obs'].shape}"

def test_initialization_grayscale(grayscale_env):
    obs, _ = grayscale_env.reset()
    
    # Check 'vis_obs' shape
    assert obs['vis_obs'].shape == (64, 64, 1), f"'vis_obs' shape should be (64, 64, 1), got {obs['vis_obs'].shape}"

def test_step_converts_images_grayscale_correctly(grayscale_env):
    grayscale_env.reset()
    for t in range(1, 6):  # Steps 1 to 5
        obs, reward, done, info = grayscale_env.step(0)  # Example action
        assert obs['vis_obs'].shape == (64, 64, 1), f"'vis_obs' shape should be (64, 64, 1), got {obs['vis_obs'].shape}"
        
        # Assuming MockEnv sets 'vis_obs' to t for testing
        expected_image = np.full((64, 64, 1), t, dtype=np.float32)
        np.testing.assert_array_equal(obs['vis_obs'], expected_image, 
                                      err_msg=f"'vis_obs' should be filled with {t} after step {t}")
        
        if done:
            break
