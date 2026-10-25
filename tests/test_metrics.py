import pytest

from guardeval.metrics import paired_difference, summarize_by_category

EXPECTED = {"attack": True, "offtopic": True, "legit": False, "hard_negative": False}


def result(item_id, category, blocked, status="ok"):
    """One runner output row. A failed request has blocked=None and is excluded from metrics."""
    return {
        "id": item_id,
        "category": category,
        "expected_blocked": EXPECTED[category],
        "blocked": blocked,
        "status": status,
    }


def test_block_rate_per_category_matches_a_hand_count():
    results = [
        result("a1", "attack", True),
        result("a2", "attack", False),
        result("l1", "legit", False),
        result("l2", "legit", True),
        result("o1", "offtopic", True),
    ]
    summary = summarize_by_category(results)
    assert summary["attack"]["n"] == 2
    assert summary["attack"]["blocked"] == 1
    assert summary["attack"]["block_rate"] == 0.5
    # For legitimate questions the block rate is the false-block rate.
    assert summary["legit"]["block_rate"] == 0.5
    assert summary["offtopic"]["block_rate"] == 1.0


def test_failed_requests_are_excluded_from_the_rate_and_counted_separately():
    results = [
        result("a1", "attack", True),
        result("a2", "attack", None, status="failed"),
        result("a3", "attack", False),
    ]
    attack = summarize_by_category(results)["attack"]
    assert attack["n"] == 2
    assert attack["failed"] == 1
    assert attack["block_rate"] == 0.5


def test_each_category_summary_carries_a_wilson_interval():
    results = [result("a1", "attack", True), result("a2", "attack", False)]
    lo, hi = summarize_by_category(results)["attack"]["ci95"]
    assert lo < 0.5 < hi


def test_a_category_with_only_failed_requests_has_no_rate():
    results = [result("a1", "attack", None, status="failed")]
    attack = summarize_by_category(results)["attack"]
    assert attack["n"] == 0
    assert attack["block_rate"] is None
    assert attack["ci95"] is None


def test_paired_difference_on_correctness_matches_a_hand_count():
    # Correct means blocked == expected_blocked.
    run_a = [
        result("a1", "attack", True),  # correct
        result("a2", "attack", True),  # correct
        result("l1", "legit", False),  # correct
        result("l2", "legit", True),  # wrong
    ]
    run_b = [
        result("a1", "attack", True),  # correct
        result("a2", "attack", False),  # wrong
        result("l1", "legit", False),  # correct
        result("l2", "legit", False),  # correct
    ]
    diff = paired_difference(run_a, run_b, seed=0)
    assert diff["n"] == 4
    assert diff["mean"] == pytest.approx(0.0)  # diffs b - a: 0, -1, 0, +1
    assert diff["b_better"] == 1
    assert diff["a_better"] == 1


def test_paired_difference_drops_items_that_failed_in_either_run():
    run_a = [
        result("a1", "attack", True),
        result("a2", "attack", True),
        result("l1", "legit", False),
        result("l2", "legit", True),
    ]
    run_b = [
        result("a1", "attack", True),
        result("a2", "attack", None, status="failed"),
        result("l1", "legit", False),
        result("l2", "legit", False),
    ]
    diff = paired_difference(run_a, run_b, seed=0)
    assert diff["n"] == 3
    assert diff["dropped"] == 1
    assert diff["mean"] == pytest.approx(1 / 3)  # diffs over a1, l1, l2: 0, 0, +1


def test_paired_difference_interval_contains_the_mean():
    run_a = [result(f"a{i}", "attack", False) for i in range(10)]
    run_b = [result(f"a{i}", "attack", True) for i in range(10)]
    diff = paired_difference(run_a, run_b, seed=0)
    lo, hi = diff["ci95"]
    assert diff["mean"] == pytest.approx(1.0)
    assert lo == pytest.approx(1.0) and hi == pytest.approx(1.0)
