import pytest
import numpy as np
from gymnasium import spaces
from neroRL.environments.wrappers.pytorch_shape import PyTorchEnv
from tests.envs.mock_env import MockEnv

@pytest.fixture
def mock_env():
    return MockEnv()

@pytest.fixture
def wrapped_env(mock_env):
    return PyTorchEnv(mock_env)

def test_wrapped_env_observation_space_unchanged(mock_env, wrapped_env):
    assert mock_env.observation_space == wrapped_env._env.observation_space, "Wrapped environment's observation space should remain unchanged"
    assert wrapped_env.observation_space is not mock_env.observation_space, "Wrapped environment's observation space should be a new object"

def test_pytorch_env_modifies_image_observation_space(wrapped_env):
    original_spaces = wrapped_env._env.observation_space.spaces
    modified_spaces = wrapped_env.observation_space.spaces

    for key, original_space in original_spaces.items():
        assert key in modified_spaces, f"Key '{key}' should be present in the modified observation space"
        if isinstance(original_space, spaces.Box) and len(original_space.shape) == 3:
            # Expect the shape to be (C, H, W)
            expected_shape = (original_space.shape[2], original_space.shape[0], original_space.shape[1])
            modified_space = modified_spaces[key]
            assert isinstance(modified_space, spaces.Box), f"'{key}' should be a spaces.Box in the modified observation space"
            assert modified_space.shape == expected_shape, f"'{key}' should have shape {expected_shape}, got {modified_space.shape}"
            assert modified_space.dtype == original_space.dtype, f"'{key}' should have dtype {original_space.dtype}, got {modified_space.dtype}"
        else:
            # Non-visual observations should remain unchanged
            modified_space = modified_spaces[key]
            assert modified_space == original_space, f"'{key}' should remain unchanged in the modified observation space"

def test_visual_observations_are_channels_first(wrapped_env):
    obs, _ = wrapped_env.reset()
    for key, value in obs.items():
        if isinstance(value, np.ndarray) and value.ndim == 3:
            # Original shape: (H, W, C) -> Expected: (C, H, W)
            original_shape = (64, 64, 3)  # As defined in MockEnv
            expected_shape = (original_shape[2], original_shape[0], original_shape[1])
            assert value.shape == expected_shape, f"Visual observation '{key}' should have shape {expected_shape}, got {value.shape}"
    obs, _, _, _ = wrapped_env.step(0)
    for key, value in obs.items():
        if isinstance(value, np.ndarray) and value.ndim == 3:
            # Original shape: (H, W, C) -> Expected: (C, H, W)
            original_shape = (64, 64, 3)
            expected_shape = (original_shape[2], original_shape[0], original_shape[1])
            assert value.shape == expected_shape, f"Visual observation '{key}' should have shape {expected_shape}, got {value.shape}"

def test_observation_shapes_after_reset(wrapped_env):
    obs, _ = wrapped_env.reset()
    for key, value in obs.items():
        if isinstance(value, np.ndarray):
            if value.ndim == 3:
                expected_shape = (3, 64, 64)  # Channels-first, as defined in MockEnv
                assert value.shape == expected_shape, f"Visual observation '{key}' should have shape {expected_shape}, got {value.shape}"

def test_observation_shapes_after_step(wrapped_env):
    wrapped_env.reset()
    action = 0
    obs, _, _, _ = wrapped_env.step(action)
    for key, value in obs.items():
        if isinstance(value, np.ndarray):
            if value.ndim == 3:
                expected_shape = (3, 64, 64)  # Channels-first, as defined in MockEnv
                assert value.shape == expected_shape, f"Visual observation '{key}' should have shape {expected_shape}, got {value.shape}"
