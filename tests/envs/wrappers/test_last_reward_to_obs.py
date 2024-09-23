import pytest
import numpy as np
from gymnasium import spaces
from neroRL.environments.wrappers.last_action_to_obs import LastActionToObs
from neroRL.environments.wrappers.last_reward_to_obs import LastRewardToObs
from tests.envs.mock_env import MockEnv

@pytest.fixture
def mock_env():
    return MockEnv()

@pytest.fixture
def wrapped_env(mock_env):
    return LastRewardToObs(mock_env)

@pytest.fixture
def  wrapped_last_action_env(mock_env):
    wrapped_env = LastActionToObs(mock_env)
    return LastRewardToObs(wrapped_env)

def test_wrapped_env_observation_space_unchanged(mock_env, wrapped_env):
    assert mock_env.observation_space == wrapped_env._env.observation_space, "Wrapped environment's observation space should remain unchanged"

def test_last_reward_space_added(wrapped_env):
    assert "last_reward" in wrapped_env.observation_space.spaces, "'last_reward' should be in the observation space"

    last_reward_space = wrapped_env.observation_space.spaces["last_reward"]
    assert isinstance(last_reward_space, spaces.Box), "'last_reward' should be a spaces.Box"
    assert last_reward_space.shape == (1,), f"'last_reward' should have shape (1,), got {last_reward_space.shape}"
    assert last_reward_space.dtype == np.float32, f"'last_reward' should have dtype np.float32, got {last_reward_space.dtype}"
    assert last_reward_space.low == -np.inf, "'last_reward' space lower bound should be -inf"
    assert last_reward_space.high == np.inf, "'last_reward' space upper bound should be inf"

def test_last_reward_initialization(wrapped_env):
    obs, _ = wrapped_env.reset()
    last_reward = obs.get("last_reward", None)
    assert last_reward is not None, "'last_reward' should be present in the observation after reset"
    expected = np.array([0.0], dtype=np.float32)
    np.testing.assert_array_equal(last_reward, expected, err_msg="'last_reward' should be initialized to zero after reset")

def test_last_reward_after_step(wrapped_env):
    wrapped_env.reset()
    action = 1
    for i in range(3):
        obs, reward, done, info = wrapped_env.step(action)
    last_reward = obs.get("last_reward", None)
    assert last_reward is not None, "'last_reward' should be present in the observation after step"
    expected = np.array([1.0], dtype=np.float32)
    np.testing.assert_array_equal(last_reward, expected, err_msg="'last_reward' should correctly reflect the received reward")

def test_observation_space_with_last_action(wrapped_last_action_env):
    assert "last_action_last_reward" in wrapped_last_action_env.observation_space.spaces, "'last_action_last_reward' should be in the observation space"

    last_action_last_reward_space = wrapped_last_action_env.observation_space.spaces["last_action_last_reward"]
    expected_shape = (wrapped_last_action_env._env._num_actions + 1,)
    assert isinstance(last_action_last_reward_space, spaces.Box), "'last_action_last_reward' should be a spaces.Box"
    assert last_action_last_reward_space.shape == expected_shape, f"'last_action_last_reward' should have shape {expected_shape}, got {last_action_last_reward_space.shape}"
    assert last_action_last_reward_space.dtype == np.float32, f"'last_action_last_reward' should have dtype np.float32, got {last_action_last_reward_space.dtype}"

def test_observations_are_dicts_with_last_action_last_reward(wrapped_last_action_env):
    obs, _ = wrapped_last_action_env.reset()
    assert isinstance(obs, dict), "Observation returned by reset should be a dict"
    assert "last_action_last_reward" in obs, "'last_action_last_reward' should be present in the observation after reset"
    assert "last_action" not in obs, "'last_action' should not be present in the observation after reset"
    assert "last_reward" not in obs, "'last_reward' should not be present in the observation after reset"

    action = 0
    obs, _, _, _ = wrapped_last_action_env.step(action)
    assert isinstance(obs, dict), "Observation returned by step should be a dict"
    assert "last_action_last_reward" in obs, "'last_action_last_reward' should be present in the observation after step"
    assert "last_action" not in obs, "'last_action' should not be present in the observation after step"
    assert "last_reward" not in obs, "'last_reward' should not be present in the observation after step"

def test_last_action_last_reward_initialization(wrapped_last_action_env):
    obs, _ = wrapped_last_action_env.reset()
    last_action_last_reward = obs.get("last_action_last_reward", None)
    assert last_action_last_reward is not None, "'last_action_last_reward' should be present in the observation after reset"
    expected = np.zeros(wrapped_last_action_env._env._num_actions + 1, dtype=np.float32)
    np.testing.assert_array_equal(last_action_last_reward, expected, err_msg="'last_action_last_reward' should be initialized to zeros after reset")

def test_last_action_last_reward_after_step(wrapped_last_action_env):
    wrapped_last_action_env.reset()
    action = 2
    _, last_reward, _, _ = wrapped_last_action_env.step(action)
    obs, _, _, _ = wrapped_last_action_env.step(action)
    last_action_last_reward = obs.get("last_action_last_reward", None)
    assert last_action_last_reward is not None, "'last_action_last_reward' should be present in the observation after step"

    # Create expected one-hot encoding for the action
    expected_last_action = np.zeros(wrapped_last_action_env._env._num_actions, dtype=np.float32)
    expected_last_action[action] = 1.0

    # Concatenate the last_reward
    expected = np.concatenate([expected_last_action, [last_reward]]).astype(np.float32)
    np.testing.assert_array_equal(last_action_last_reward, expected, err_msg="'last_action_last_reward' should correctly reflect the last action and reward")

def test_last_action_last_reward_not_present(wrapped_last_action_env):
    obs, _ = wrapped_last_action_env.reset()
    assert "last_action" not in obs, "'last_action' should not be present in the observation"
    assert "last_reward" not in obs, "'last_reward' should not be present in the observation"

    action = 1
    obs, _, _, _ = wrapped_last_action_env.step(action)
    assert "last_action" not in obs, "'last_action' should not be present in the observation after step"
    assert "last_reward" not in obs, "'last_reward' should not be present in the observation after step"
