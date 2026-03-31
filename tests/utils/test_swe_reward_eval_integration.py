import asyncio
from types import SimpleNamespace

import pytest

from examples.swe_bench.reward import post_process_rewards, reward_func
from slime.utils.types import Sample


def _make_args(**overrides):
    base = dict(
        reward_key="reward",
        swe_eval_reward_enable=True,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _make_reward_payload():
    return {
        "meta_info": {
            "input_token_logprobs": [
                [0.0],   # prompt/bos
                [-0.1],  # response token 1
                [-0.2],  # response token 2
            ]
        },
        "reward": 0.0,
    }


def _make_sample(*, index: int, group_index: int, session_id: str, with_reward_payload: bool = True) -> Sample:
    sample = Sample(index=index, group_index=group_index)
    sample.session_id = session_id
    sample.response_length = 2
    sample.tokens = [1, 2, 3, 4]
    if with_reward_payload:
        sample.reward = _make_reward_payload()
    sample.metadata["instance_id"] = "pydata__xarray-1"
    sample.metadata["patch"] = "diff --git a/a b/a\n"
    return sample


def test_reward_func_applies_eval_mapping_single(monkeypatch):
    s1 = _make_sample(index=0, group_index=7, session_id="s1", with_reward_payload=False)

    async def fake_teacher(*_args, **_kwargs):
        return _make_reward_payload()

    def fake_group_eval(args, group_samples):
        assert len(group_samples) == 1
        return {"s1": {"resolved": True, "status": "resolved", "error": "", "run_id": "r1"}}

    monkeypatch.setattr("examples.swe_bench.reward._query_teacher_logprobs", fake_teacher)
    monkeypatch.setattr("examples.swe_bench.reward.evaluate_group_for_reward", fake_group_eval)

    out = asyncio.run(reward_func(_make_args(), s1))
    s1.reward = out
    raw_rewards, norm_rewards = post_process_rewards(_make_args(), [s1])

    assert out["reward"] == 1.0
    assert raw_rewards == [1.0]
    assert norm_rewards == [1.0]
    assert s1.metadata["swe_eval_status"] == "resolved"
    assert s1.teacher_log_probs.tolist() == pytest.approx([-0.1, -0.2])


def test_reward_func_applies_eval_mapping_batch(monkeypatch):
    s1 = _make_sample(index=0, group_index=3, session_id="s1", with_reward_payload=False)
    s2 = _make_sample(index=1, group_index=3, session_id="s2", with_reward_payload=False)
    calls = {"count": 0}

    async def fake_teacher(*_args, **_kwargs):
        return _make_reward_payload()

    def fake_group_eval(args, group_samples):
        calls["count"] += 1
        assert len(group_samples) == 2
        return {
            "s1": {"resolved": True, "status": "resolved", "error": "", "run_id": "r"},
            "s2": {"resolved": False, "status": "unresolved", "error": "", "run_id": "r"},
        }

    monkeypatch.setattr("examples.swe_bench.reward._query_teacher_logprobs", fake_teacher)
    monkeypatch.setattr("examples.swe_bench.reward.evaluate_group_for_reward", fake_group_eval)

    out = asyncio.run(reward_func(_make_args(), [s1, s2]))
    s1.reward, s2.reward = out
    raw_rewards, norm_rewards = post_process_rewards(_make_args(), [s1, s2])

    assert calls["count"] == 1
    assert raw_rewards == [1.0, -1.0]
    assert norm_rewards == [1.0, -1.0]


def test_reward_func_keeps_zero_reward_when_eval_disabled(monkeypatch):
    s1 = _make_sample(index=0, group_index=7, session_id="s1", with_reward_payload=False)

    async def fake_teacher(*_args, **_kwargs):
        return _make_reward_payload()

    monkeypatch.setattr("examples.swe_bench.reward._query_teacher_logprobs", fake_teacher)

    out = asyncio.run(reward_func(_make_args(swe_eval_reward_enable=False), s1))
    s1.reward = out
    raw_rewards, norm_rewards = post_process_rewards(_make_args(swe_eval_reward_enable=False), [s1])

    assert raw_rewards == [0.0]
    assert norm_rewards == [0.0]
    assert s1.reward["reward"] == 0.0
    assert s1.metadata["swe_eval_status"] == "eval_disabled"
