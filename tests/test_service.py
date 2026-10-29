"""Tests for the HTTP service handler (exercised without a socket)."""

from __future__ import annotations

from rlhf_reward_model.domain import Domain, demonstrations, preferences
from rlhf_reward_model.evaluation import evaluate
from rlhf_reward_model.pipeline import PipelineResult
from rlhf_reward_model.ppo import PPOConfig, train_ppo
from rlhf_reward_model.reward_model import RewardModelConfig, train_reward_model
from rlhf_reward_model.service import RLHFService, handle
from rlhf_reward_model.sft import SFTConfig, behavior_clone


def _service() -> RLHFService:
    domain = Domain.build(seed=0)
    sft = demonstrations(domain, 400, temperature=1.0, seed=1)
    reference = behavior_clone(domain, sft, SFTConfig(seed=2))
    train = preferences(domain, 3000, seed=3)
    held_out = preferences(domain, 400, seed=4)
    model = train_reward_model(domain, train, RewardModelConfig(iterations=250, seed=5))
    policy = train_ppo(domain, reference, model, PPOConfig(iterations=30, seed=6))
    report = evaluate(domain, policy, reference, model, held_out)
    result = PipelineResult(
        domain=domain, reference=reference, reward_model=model, policy=policy, report=report
    )
    return RLHFService(result=result)


def test_health_and_ready() -> None:
    service = _service()
    status, payload = handle("GET", "/health", {}, service)
    assert status == 200
    assert payload["prompts"] == service.result.domain.n_prompts
    status, payload = handle("GET", "/ready", {}, service)
    assert status == 200
    assert payload["status"] == "ready"


def test_report_endpoint() -> None:
    service = _service()
    status, payload = handle("GET", "/report", {}, service)
    assert status == 200
    assert payload["reference_reward"] <= payload["optimal_reward"]
    assert 0.0 <= payload["reward_model_accuracy"] <= 1.0


def test_score_endpoint_returns_rewards() -> None:
    service = _service()
    status, payload = handle("POST", "/score", {"prompt": 0}, service)
    assert status == 200
    assert len(payload["responses"]) == service.result.domain.n_responses
    for item in payload["responses"]:
        assert "true_reward" in item
        assert "learned_reward" in item


def test_score_rejects_bad_prompt() -> None:
    service = _service()
    for bad in (-1, 999, "x", True):
        status, payload = handle("POST", "/score", {"prompt": bad}, service)
        assert status == 400
        assert "error" in payload


def test_unknown_route_and_metrics() -> None:
    service = _service()
    status, _ = handle("GET", "/nope", {}, service)
    assert status == 404
    handle("GET", "/report", {}, service)
    rendered = service.metrics.render_prometheus()
    assert "service_requests_total" in rendered
    assert service.metrics.total_errors >= 1
