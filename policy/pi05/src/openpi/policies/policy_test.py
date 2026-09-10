from openpi_client import action_chunk_broker
import numpy as np
import pytest

from openpi import transforms
from openpi.policies import aloha_policy
from openpi.policies.policy import Policy
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config


def test_transform_input_does_not_mutate_expert_action_targets():
    policy = object.__new__(Policy)
    policy._input_transform = transforms.compose([
        aloha_policy.AlohaInputs(adapt_to_pi=False),
        transforms.DeltaActions(transforms.make_bool_mask(6, -1, 6, -1)),
    ])
    sample = aloha_policy.make_aloha_example()
    sample["actions"] = np.full((2, 14), 2.0, dtype=np.float32)
    original = sample["actions"].copy()

    first = policy.transform_input(sample)["actions"]
    second = policy.transform_input(sample)["actions"]

    np.testing.assert_array_equal(sample["actions"], original)
    np.testing.assert_array_equal(first, second)


@pytest.mark.manual
def test_infer():
    config = _config.get_config("pi0_aloha_sim")
    policy = _policy_config.create_trained_policy(config, "gs://openpi-assets/checkpoints/pi0_aloha_sim")

    example = aloha_policy.make_aloha_example()
    result = policy.infer(example)

    assert result["actions"].shape == (config.model.action_horizon, 14)


@pytest.mark.manual
def test_broker():
    config = _config.get_config("pi0_aloha_sim")
    policy = _policy_config.create_trained_policy(config, "gs://openpi-assets/checkpoints/pi0_aloha_sim")

    broker = action_chunk_broker.ActionChunkBroker(
        policy,
        # Only execute the first half of the chunk.
        action_horizon=config.model.action_horizon // 2,
    )

    example = aloha_policy.make_aloha_example()
    for _ in range(config.model.action_horizon):
        outputs = broker.infer(example)
        assert outputs["actions"].shape == (14,)
