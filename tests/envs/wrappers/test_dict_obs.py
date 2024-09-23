import pytest
from gymnasium import spaces
from neroRL.environments.wrappers.stacked_observation import StackedObservationEnv
from neroRL.environments.wrappers.grayscale_visual_observation import GrayscaleVisualObsEnv
from neroRL.environments.wrappers.scaled_visual_observation import ScaledVisualObsEnv
from neroRL.environments.wrappers.pytorch_shape import PyTorchEnv
from neroRL.environments.wrappers.last_action_to_obs import LastActionToObs
from neroRL.environments.wrappers.last_reward_to_obs import LastRewardToObs
from neroRL.environments.wrappers.positional_encoding import PositionalEncodingEnv
from neroRL.environments.wrappers.frame_skip import FrameSkipEnv
from neroRL.environments.wrappers.reward_normalization import RewardNormalizer
from neroRL.environments.memory_gym_wrapper import MemoryGymWrapper
from tests.envs.mock_env import MockEnv

@pytest.fixture
def mock_env():
    return MockEnv()

@pytest.mark.parametrize("wrapper_class,kwargs", [
    (StackedObservationEnv, {"num_stacks": 4}),
    (GrayscaleVisualObsEnv, {}),
    (ScaledVisualObsEnv, {"width": 20, "height": 20}),
    (PyTorchEnv, {}),
    (LastActionToObs, {}),
    (LastRewardToObs, {}),
    (PositionalEncodingEnv, {}),
    (FrameSkipEnv, {"skip": 4}),
    (RewardNormalizer, {"max_reward": 5}),
])
def test_observations_are_dicts(wrapper_class, kwargs, mock_env):
    env_wrapped = wrapper_class(mock_env, **kwargs)
    assert isinstance(env_wrapped.observation_space, spaces.Dict), "Observation space should be a spaces.Dict"
    obs, _ = env_wrapped.reset()
    assert isinstance(obs, dict), "Observation returned by reset should be a dict"
    action = 0
    obs, _, _, _ = env_wrapped.step(action)
    assert isinstance(obs, dict), "Observation returned by step should be a dict"

@pytest.mark.parametrize("wrapper_class,kwargs", [
    (StackedObservationEnv, {"num_stacks": 4}),
    (GrayscaleVisualObsEnv, {}),
    (ScaledVisualObsEnv, {"width": 20, "height": 20}),
    (PyTorchEnv, {}),
    (LastActionToObs, {}),
    (LastRewardToObs, {}),
    (PositionalEncodingEnv, {}),
    (FrameSkipEnv, {"skip": 4}),
    (RewardNormalizer, {"max_reward": 5}),
])
def test_close(wrapper_class, kwargs, mock_env):
    env_wrapped = wrapper_class(mock_env, **kwargs)
    try:
        env_wrapped.close()
    except Exception as e:
        pytest.fail(f"Closing the {wrapper_class.__name__} raised an exception: {e}")
