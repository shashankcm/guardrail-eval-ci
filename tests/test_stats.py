import pytest

from guardeval.stats import bootstrap_mean_interval, wilson_interval


def test_wilson_interval_matches_a_hand_computed_value_for_9_of_10():
    # Hand calculation with z = 1.96: centre 0.789, half-width 0.193.
    lo, hi = wilson_interval(9, 10)
    assert round(lo, 3) == 0.596
    assert round(hi, 3) == 0.982


def test_wilson_interval_for_zero_successes_starts_at_zero_and_stays_informative():
    lo, hi = wilson_interval(0, 10)
    assert lo == 0.0
    assert round(hi, 3) == 0.278  # z^2 / (n + z^2)


def test_wilson_interval_for_all_successes_ends_at_one():
    lo, hi = wilson_interval(10, 10)
    assert round(lo, 3) == 0.722
    assert hi == 1.0


def test_wilson_interval_contains_the_observed_rate():
    lo, hi = wilson_interval(3, 7)
    assert lo < 3 / 7 < hi


def test_wilson_interval_rejects_an_empty_sample():
    with pytest.raises(ValueError):
        wilson_interval(0, 0)


def test_wilson_interval_rejects_more_successes_than_trials():
    with pytest.raises(ValueError):
        wilson_interval(5, 4)


def test_bootstrap_of_all_zero_differences_has_zero_width():
    assert bootstrap_mean_interval([0.0] * 20, seed=1) == (0.0, 0.0, 0.0)


def test_bootstrap_is_reproducible_with_the_same_seed():
    diffs = [1, 0, 0, 0, 1, 0, 0, 0, 0, 0]
    assert bootstrap_mean_interval(diffs, seed=7) == bootstrap_mean_interval(diffs, seed=7)


def test_bootstrap_interval_contains_the_mean_and_stays_in_range():
    diffs = [1, 0, 0, 0, 1, 0, 0, 0, 0, 0]  # mean 0.2
    mean, lo, hi = bootstrap_mean_interval(diffs, seed=0)
    assert mean == pytest.approx(0.2)
    assert lo <= mean <= hi
    assert -1.0 <= lo and hi <= 1.0


def test_bootstrap_interval_excludes_zero_when_most_differences_are_positive():
    diffs = [1] * 25 + [0] * 5  # mean 0.83
    _, lo, _ = bootstrap_mean_interval(diffs, seed=0)
    assert lo > 0


def test_bootstrap_of_no_differences_is_an_error():
    with pytest.raises(ValueError):
        bootstrap_mean_interval([], seed=0)
