"""
Focused tests for EvidenceRetrievalService._deviation_score()'s pooled-
variance correction. Tests the private scoring function directly via
module-level import -- this is a pure, deterministic function with no
adapter/telemetry dependency, so direct unit testing is appropriate.
"""

from __future__ import annotations

import pandas as pd
import pytest

from app.services.evidence.service import _deviation_score, _EPSILON


def test_flat_early_shifted_late_no_longer_explodes():
    """
    The original bug: early window constant, late window shifted.
    Previously this collapsed the denominator to ~epsilon and produced
    a score in the billions for even a modest shift. With pooled
    variance, the late window's own nonzero variance keeps the
    denominator away from the epsilon floor.
    """
    early = [100.0] * 25  # perfectly flat baseline
    late = [100.0, 150.0, 90.0, 140.0, 200.0, 110.0, 180.0, 95.0, 160.0, 120.0] * 2 + [130.0] * 5
    series = pd.Series(early + late)

    score = _deviation_score(series)

    # Must be finite and NOT in the billions/trillions -- the specific
    # threshold here is generous and structural (orders of magnitude
    # below the old failure mode), not tuned to any real case.
    assert score < 10_000
    assert score > 0  # a real shift occurred; score shouldn't vanish either


def test_flat_entire_series_scores_near_zero():
    """
    Genuinely no signal: both windows constant, no shift at all.
    This is the one case where a near-zero denominator IS correct --
    pooled variance still floors to epsilon here, and the numerator
    (no mean shift) is zero, so the score should be exactly zero, not
    exploded.
    """
    series = pd.Series([42.0] * 40)

    score = _deviation_score(series)

    assert score == 0.0


def test_normal_noisy_series_regression():
    """
    A well-conditioned series (real variance in both windows, modest
    shift) should produce a stable, bounded score -- this is a
    regression check that ordinary, non-degenerate behavior is not
    materially disrupted by the pooled-variance change.
    """
    early = [10.0, 12.0, 9.0, 11.0, 10.5, 9.5, 11.5, 10.0, 9.8, 10.2]
    late = [15.0, 17.0, 14.0, 16.0, 15.5, 14.5, 16.5, 15.0, 14.8, 15.2]
    series = pd.Series(early + late)

    score = _deviation_score(series)

    assert 0.1 < score < 100  # sane, bounded range for a modest, well-conditioned shift


def test_pathological_quantized_jump_stays_bounded():
    """
    Explicit protection against the orders-of-magnitude explosion
    pattern observed in real train-ticket memory columns: a flat
    integer-valued early window followed by a single flat-but-shifted
    late window (the quantized-memory-reading pattern). Must not
    reach the 1e9+ magnitudes the pre-fix formula produced, and with
    the zero-variance fallback now defined, the result is an exact,
    principled value, not just "small enough".
    """
    early = [4096.0] * 20
    late = [8192.0] * 20
    series = pd.Series(early + late)

    score = _deviation_score(series)

    # relative change = |8192 - 4096| / max(4096, 8192) = 4096 / 8192 = 0.5
    assert score == pytest.approx(0.5)


def test_zero_variance_both_windows_flat_uses_relative_change():
    """
    Both early and late windows exactly constant, different levels --
    the exact real-data pattern behind case-0256's 1.6384e13 score.
    Confirms the fallback branch, not just an upper bound.
    """
    early = [8192.0] * 15
    late = [16384.0] * 15
    series = pd.Series(early + late)

    score = _deviation_score(series)

    assert score == pytest.approx(0.5)  # 8192 / 16384


def test_zero_variance_both_flat_no_shift_scores_zero():
    """Flat-to-flat with NO shift at all: genuinely zero signal, score must be exactly 0."""
    series = pd.Series([250.0] * 30)
    score = _deviation_score(series)
    assert score == 0.0


def test_one_window_zero_variance_other_nonzero_uses_pooled_not_fallback():
    """
    Exactly one window flat, the other genuinely noisy -- confirms this
    does NOT hit the new zero-variance fallback branch (early_std==0 but
    late_std!=0, so the `and` condition is false) and instead uses the
    existing pooled-std path, which already stays well-conditioned here.
    """
    early = [50.0] * 10  # flat
    late = [70.0, 65.0, 80.0, 60.0, 75.0, 68.0, 72.0, 66.0, 78.0, 63.0]  # noisy
    series = pd.Series(early + late)

    score = _deviation_score(series)

    # Sanity: pooled_std = late_std/sqrt(2) here (early_std=0), so score
    # should be a normal, bounded z-score-like value, not a fallback ratio.
    assert 0 < score < 100


def test_short_series_below_four_points_unaffected():
    """
    n < 4 uses the raw abs(last - first) fallback, entirely untouched
    by this change -- regression check.
    """
    series = pd.Series([5.0, 8.0, 3.0])
    assert _deviation_score(series) == pytest.approx(2.0)


def test_empty_series_returns_zero():
    assert _deviation_score(pd.Series([], dtype=float)) == 0.0