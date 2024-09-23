import numpy as np
import pytest
from gymnasium import spaces
from neroRL.environments.wrappers.positional_encoding import PositionalEncodingEnv
from tests.envs.mock_env import MockEnv

@pytest.fixture
def mock_env():
    return MockEnv()

@pytest.fixture
def wrapped_env(mock_env):
    return PositionalEncodingEnv(mock_env)

def test_wrapped_env_observation_space_unchanged(mock_env, wrapped_env):
    assert mock_env.observation_space == wrapped_env._env.observation_space, "Wrapped environment's observation space should remain unchanged"
    assert wrapped_env.observation_space is not mock_env.observation_space, "Wrapped environment's observation space should be a new object"

def test_wrapper_adds_positional_encoding_space(wrapped_env):
    assert "positional_encoding" in wrapped_env.observation_space.spaces, "'positional_encoding' should be in the observation space"

    pos_enc_space = wrapped_env.observation_space.spaces["positional_encoding"]
    assert isinstance(pos_enc_space, spaces.Box), "'positional_encoding' should be a spaces.Box"
    assert pos_enc_space.shape == (16,), f"'positional_encoding' should have shape (16,), got {pos_enc_space.shape}"
    assert pos_enc_space.dtype == np.float32, f"'positional_encoding' should have dtype np.float32, got {pos_enc_space.dtype}"

def test_observations_are_dicts(wrapped_env):
    obs, _ = wrapped_env.reset()
    assert isinstance(obs, dict), "Observation returned by reset should be a dict"
    assert "positional_encoding" in obs, "'positional_encoding' should be in the observation dictionary"

    action = 0
    obs, _, _, _ = wrapped_env.step(action)
    assert isinstance(obs, dict), "Observation returned by step should be a dict"
    assert "positional_encoding" in obs, "'positional_encoding' should be in the observation dictionary"

def test_positional_encoding_on_reset(wrapped_env):
    obs, _ = wrapped_env.reset()
    expected_pos_enc = wrapped_env.pos_encoding[0]
    np.testing.assert_array_almost_equal(obs["positional_encoding"], expected_pos_enc, err_msg="Positional encoding after reset is incorrect")

def test_positional_encoding_on_step(wrapped_env):
    wrapped_env.reset()
    for t in range(1, 10):
        obs, _, _, _ = wrapped_env.step(1)
        expected_pos_enc = wrapped_env.pos_encoding[t]
        np.testing.assert_array_almost_equal(obs["positional_encoding"], expected_pos_enc, err_msg=f"Positional encoding at step {t} is incorrect")

def test_positional_encoding_shape_and_dtype(wrapped_env):
    obs, _ = wrapped_env.reset()
    pos_enc = obs["positional_encoding"]
    assert pos_enc.shape == (16,), f"Positional encoding should have shape (16,), got {pos_enc.shape}"
    assert pos_enc.dtype == np.float32, f"Positional encoding should have dtype np.float32, got {pos_enc.dtype}"

def test_positional_encoding_values(wrapped_env):
    wrapped_env.reset()
    for t in range(1, 512):
        obs, _, _, _ = wrapped_env.step(0)
        expected = wrapped_env.pos_encoding[t]
        np.testing.assert_array_almost_equal(obs["positional_encoding"], expected, decimal=5, err_msg=f"Positional encoding at step {t} does not match expected values")
        if t >= wrapped_env.max_episode_steps:
            break
