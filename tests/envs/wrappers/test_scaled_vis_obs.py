import cv2
import pytest
import numpy as np
from gymnasium import spaces
from neroRL.environments.wrappers.scaled_visual_observation import ScaledVisualObsEnv
from tests.envs.mock_env import MockEnv

@pytest.fixture
def mock_env():
    return MockEnv()

@pytest.fixture
def scaled_env(mock_env):
    return ScaledVisualObsEnv(mock_env, width=20, height=20)

def test_observation_space_scaled_correctly(scaled_env):
    obs_space = scaled_env.observation_space

    # Check 'vis_obs' space
    assert 'vis_obs' in obs_space.spaces, "'vis_obs' should be in the observation space"
    expected_image_shape = (20, 20, 3)
    assert obs_space.spaces['vis_obs'].shape == expected_image_shape, \
        f"'vis_obs' should have shape {expected_image_shape}, got {obs_space.spaces['vis_obs'].shape}"
    
    # Check 'vec_obs' space remains unchanged
    assert 'vec_obs' in obs_space.spaces, "'vec_obs' should be in the observation space"
    expected_vector_shape = (10,)
    assert obs_space.spaces['vec_obs'].shape == expected_vector_shape, \
        f"'vec_obs' should have shape {expected_vector_shape}, got {obs_space.spaces['vec_obs'].shape}"

def test_initialization_scaled(scaled_env):
    obs, _ = scaled_env.reset()
    
    # Check 'vis_obs' shape and dtype
    assert obs['vis_obs'].shape == (20, 20, 3), f"'vis_obs' shape should be (20, 20, 3), got {obs['vis_obs'].shape}"
    
    # Check 'vis_obs' values are scaled correctly
    expected_image = np.zeros((20, 20, 3), dtype=np.float32)  # Assuming MockEnv returns zeros after reset
    np.testing.assert_array_equal(obs['vis_obs'], expected_image, 
                                  err_msg="'vis_obs' should be initialized to zeros after reset and scaled to [0.0, 1.0]")

def test_resize_vis_obs_correctness(scaled_env):
    # Create a sample input image
    input_image = np.zeros((64, 64, 3), dtype=np.uint8)
    for i in range(64):
        for j in range(64):
            input_image[i, j, 0] = i
            input_image[i, j, 1] = j
            input_image[i, j, 2] = (i + j) // 2
    
    # Define the expected resized image using cv2.resize
    expected_resized = cv2.resize(
        input_image.astype(np.float32), 
        (scaled_env._width, scaled_env._height), 
        interpolation=cv2.INTER_AREA
    )
    
    # Call the private method _resize_vis_obs
    actual_resized = scaled_env._resize_vis_obs(input_image)
    
    # Compare the output with the expected resized image
    np.testing.assert_array_almost_equal(
        actual_resized, 
        expected_resized, 
        decimal=5,
        err_msg="_resize_vis_obs did not correctly resize and scale the image"
    )

