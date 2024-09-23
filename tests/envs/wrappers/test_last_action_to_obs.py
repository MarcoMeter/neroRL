import pytest
import numpy as np
from gymnasium import spaces
from neroRL.environments.wrappers.last_action_to_obs import LastActionToObs
from tests.envs.mock_env import MockEnv

@pytest.fixture
def mock_env_discrete():
    return MockEnv()

@pytest.fixture
def wrapped_env_discrete(mock_env_discrete):
    return LastActionToObs(mock_env_discrete)

@pytest.fixture
def mock_env_multidiscrete():
    class MockEnvMultiDiscrete(MockEnv):
        def __init__(self):
            super().__init__()
            self._action_space = spaces.MultiDiscrete([2, 3])
    return MockEnvMultiDiscrete()

@pytest.fixture
def wrapped_env_multidiscrete(mock_env_multidiscrete):
    return LastActionToObs(mock_env_multidiscrete)

def test_wrapped_env_observation_space_unchanged(mock_env_discrete, wrapped_env_discrete):
    assert mock_env_discrete.observation_space == wrapped_env_discrete._env.observation_space, "Wrapped environment's observation space should remain unchanged"

def test_last_action_space_added_discrete(wrapped_env_discrete):
    assert "last_action" in wrapped_env_discrete.observation_space.spaces, "'last_action' should be in the observation space"

    last_action_space = wrapped_env_discrete.observation_space.spaces["last_action"]
    num_actions = wrapped_env_discrete._action_space.n
    expected_shape = (num_actions,)
    assert isinstance(last_action_space, spaces.Box), "'last_action' should be a spaces.Box"
    assert last_action_space.shape == expected_shape, f"'last_action' should have shape {expected_shape}, got {last_action_space.shape}"
    assert last_action_space.dtype == np.float32, f"'last_action' should have dtype np.float32, got {last_action_space.dtype}"
    assert np.all(last_action_space.low == 0.0), "'last_action' space lower bound should be 0.0"
    assert np.all(last_action_space.high == 1.0), "'last_action' space upper bound should be 1.0"

def test_last_action_space_added_multidiscrete(wrapped_env_multidiscrete):
    assert "last_action" in wrapped_env_multidiscrete.observation_space.spaces, "'last_action' should be in the observation space"

    last_action_space = wrapped_env_multidiscrete.observation_space.spaces["last_action"]
    num_actions = sum(wrapped_env_multidiscrete._action_space.nvec)
    expected_shape = (num_actions,)
    assert isinstance(last_action_space, spaces.Box), "'last_action' should be a spaces.Box"
    assert last_action_space.shape == expected_shape, f"'last_action' should have shape {expected_shape}, got {last_action_space.shape}"
    assert last_action_space.dtype == np.float32, f"'last_action' should have dtype np.float32, got {last_action_space.dtype}"
    assert np.all(last_action_space.low == 0.0), "'last_action' space lower bound should be 0.0"
    assert np.all(last_action_space.high == 1.0), "'last_action' space upper bound should be 1.0"

def test_last_action_initialization(wrapped_env_discrete):
    obs, _ = wrapped_env_discrete.reset()
    last_action = obs.get("last_action", None)
    assert last_action is not None, "'last_action' should be present in the observation after reset"
    expected = np.zeros(wrapped_env_discrete._num_actions, dtype=np.float32)
    np.testing.assert_array_equal(last_action, expected, err_msg="'last_action' should be initialized to zeros after reset")

def test_last_action_after_step_discrete(wrapped_env_discrete):
    wrapped_env_discrete.reset()
    action = 2  # Example action within Discrete(4)
    obs, reward, done, info = wrapped_env_discrete.step(action)
    last_action = obs.get("last_action", None)
    assert last_action is not None, "'last_action' should be present in the observation after step"
    expected = np.zeros(wrapped_env_discrete._num_actions, dtype=np.float32)
    expected[action] = 1.0
    np.testing.assert_array_equal(last_action, expected, err_msg="'last_action' should correctly reflect the taken action")

def test_last_action_after_step_multidiscrete(wrapped_env_multidiscrete):
    wrapped_env_multidiscrete.reset()
    action = [1, 2]  # Example action within MultiDiscrete([2,3])
    obs, _, _, _ = wrapped_env_multidiscrete.step(action)
    last_action = obs.get("last_action", None)
    assert last_action is not None, "'last_action' should be present in the observation after step"
    expected = np.zeros(wrapped_env_multidiscrete._num_actions, dtype=np.float32)
    expected[1] = 1
    expected[-1] = 1
    np.testing.assert_array_equal(last_action, expected, err_msg="'last_action' should correctly reflect the taken action")
    action = [0, 1]  # Example action within MultiDiscrete([2,3])
    obs, _, _, _ = wrapped_env_multidiscrete.step(action)
    last_action = obs.get("last_action", None)
    assert last_action is not None, "'last_action' should be present in the observation after step"
    expected = np.zeros(wrapped_env_multidiscrete._num_actions, dtype=np.float32)
    expected[0] = 1
    expected[3] = 1
    np.testing.assert_array_equal(last_action, expected, err_msg="'last_action' should correctly reflect the taken action")

def test_last_action_multiple_steps_discrete(wrapped_env_discrete):
    wrapped_env_discrete.reset()
    actions = [1, 3, 0, 2]  # Example sequence of actions
    for action in actions:
        obs, reward, done, info = wrapped_env_discrete.step(action)
        last_action = obs.get("last_action", None)
        assert last_action is not None, "'last_action' should be present in the observation after step"
        expected = np.zeros(wrapped_env_discrete._num_actions, dtype=np.float32)
        expected[action] = 1.0
        np.testing.assert_array_equal(last_action, expected, err_msg=f"'last_action' should correctly reflect action {action}")

@pytest.mark.parametrize("action_space_type, action, expected_one_hot", [
    ("Discrete", 1, [0, 1, 0, 0]),
    ("Discrete", 3, [0, 0, 0, 1]),
    ("MultiDiscrete", [0, 1], [1, 0, 0, 1, 0]),
    ("MultiDiscrete", [1, 2], [0, 1, 0, 0, 1]),
])
def test_action_to_one_hot(wrapped_env_discrete, wrapped_env_multidiscrete, action_space_type, action, expected_one_hot):
    if action_space_type == "Discrete":
        wrapped_env_discrete.reset()
        one_hot = wrapped_env_discrete._action_to_one_hot(action)
    elif action_space_type == "MultiDiscrete":
        wrapped_env_multidiscrete.reset()
        one_hot = wrapped_env_multidiscrete._action_to_one_hot(action)
    else:
        pytest.fail(f"Unsupported action_space_type: {action_space_type}")

    expected = np.array(expected_one_hot, dtype=np.float32)
    np.testing.assert_array_equal(one_hot, expected, err_msg=f"One-hot encoding for action {action} is incorrect")

def test_last_action_not_modified_other_keys(wrapped_env_discrete):
    """Ensure that other keys in the observation dictionary are not modified."""
    wrapped_env_discrete.reset()
    action = 1
    obs, _, _, _ = wrapped_env_discrete.step(action)
    # Assuming MockEnv has 'vec_obs' and 'vis_obs'
    vec_obs = obs.get("vec_obs", None)
    vis_obs = obs.get("vis_obs", None)
    assert vec_obs is not None, "'vec_obs' should be present in the observation"
    assert vis_obs is not None, "'vis_obs' should be present in the observation"
    # Check that 'vec_obs' and 'vis_obs' are as expected
    expected_vec_obs = np.ones(10, dtype=np.float32) * wrapped_env_discrete._env.t
    expected_vis_obs = np.ones((64, 64, 3), dtype=np.uint8) * wrapped_env_discrete._env.t
    np.testing.assert_array_equal(vec_obs, expected_vec_obs, err_msg="'vec_obs' should not be modified by the wrapper")
    np.testing.assert_array_equal(vis_obs, expected_vis_obs, err_msg="'vis_obs' should not be modified by the wrapper")
