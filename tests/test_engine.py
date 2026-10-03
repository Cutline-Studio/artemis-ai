import pytest

from artemis.engine import Arm, BudgetExceeded, DecisionEngine


@pytest.fixture
def eng(tmp_path):
    return DecisionEngine(tmp_path / "experiments.jsonl")


def test_starts_small_and_untried_first(eng):
    a = eng.next_arm()
    assert a.size == "100m"


def test_larger_sizes_unlock_only_after_smaller_succeed(eng):
    for a in [x for x in eng.arms if x.size == "100m"]:
        eng.record(a, 3.5, 100, "2026-10", True, 1)
    assert eng.next_arm().size in {"100m", "1b"}
    assert all(x.size in {"100m", "1b"} for x in [eng.next_arm()])


def test_ucb_prefers_better_arms(eng):
    arms = [x for x in eng.arms if x.size == "100m"]
    for a in arms:
        eng.record(a, 4.0, 10, "2026-10", True, 1)
    best = arms[3]
    for _ in range(5):
        eng.record(best, 2.0, 10, "2026-10", True, 1)
    picks = [eng.next_arm().key for _ in range(3)]
    assert best.key in picks or eng.next_arm().size == "1b"


def test_budget_cap_blocks_overspend(eng):
    eng.record(Arm("100m", 3e-4, 0.3, 0.01), 3.0, 1_999_000, "2026-10", True, 1)
    with pytest.raises(BudgetExceeded):
        eng.check_budget("2026-10", vms=1, hours=24)
    assert eng.check_budget("2026-11", vms=1, hours=24) == pytest.approx(24 * 98.32)


def test_ramp_advances_only_on_improvement(eng):
    a = Arm("100m", 3e-4, 0.3, 0.01)
    assert eng.allowed_vms() == 1
    eng.record(a, 3.5, 10, "2026-10", True, 1)
    assert eng.allowed_vms() == 1
    eng.record(a, 3.9, 10, "2026-10", True, 1)  # worse: no advance
    assert eng.allowed_vms() == 1
    eng.record(a, 3.2, 10, "2026-10", True, 1)  # improved: unlock 4 VMs
    assert eng.allowed_vms() == 4
    eng.record(a, None, 10, "2026-10", False, 4)  # unstable run does not advance
    assert eng.allowed_vms() == 4


def test_promotion_requires_margin(eng):
    assert eng.should_promote(3.0, None)
    assert not eng.should_promote(3.115, 3.12)
    assert eng.should_promote(3.05, 3.12)
