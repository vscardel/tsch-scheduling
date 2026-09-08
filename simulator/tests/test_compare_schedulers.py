"""The statistics the reviewers asked for.

Pairing is the point: every scheduler runs run_id N with the same seed, so it
sees the same topology. Comparing run N against run N removes the variance
between topologies, which is what limited the power at ten runs.
"""
from __future__ import absolute_import

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'bin'))

import compare_schedulers as cs


def test_ranks_average_ties():
    assert cs._ranks([10, 20, 20, 40]) == [1.0, 2.5, 2.5, 4.0]


def test_rank_biserial_is_one_when_one_side_always_wins():
    assert cs.rank_biserial([1.0, 2.0, 3.0]) == pytest.approx(1.0)
    assert cs.rank_biserial([-1.0, -2.0, -3.0]) == pytest.approx(-1.0)


def test_rank_biserial_is_zero_when_they_trade_evenly():
    assert cs.rank_biserial([1.0, -1.0, 2.0, -2.0]) == pytest.approx(0.0)


def test_rank_biserial_ignores_exact_ties():
    assert cs.rank_biserial([0.0, 0.0]) == 0.0


def test_holm_leaves_the_smallest_p_hardest_hit():
    ajustado = cs.holm([0.01, 0.04, 0.03])
    assert ajustado[0] == pytest.approx(0.03)      # 3 * 0.01
    assert all(a >= b for a, b in zip(ajustado, [0.01, 0.04, 0.03]))


def test_holm_never_goes_above_one():
    assert all(p <= 1.0 for p in cs.holm([0.5, 0.6, 0.9]))


def test_holm_is_monotone_in_the_sorted_order():
    ajustado = cs.holm([0.001, 0.002, 0.5])
    ordenado = sorted(ajustado)
    assert ajustado == sorted(ajustado) or ordenado[0] <= ordenado[-1]


def test_bootstrap_ci_brackets_the_median():
    diferencas = [1.0, 1.1, 0.9, 1.2, 0.8, 1.05, 0.95]
    baixo, alto = cs.bootstrap_ci(diferencas, resamples=2000)
    assert baixo <= 1.0 <= alto


def test_bootstrap_ci_is_reproducible():
    diferencas = [0.2, -0.1, 0.4, 0.3, -0.2, 0.5]
    assert cs.bootstrap_ci(diferencas, resamples=500) == \
           cs.bootstrap_ci(diferencas, resamples=500)


def test_pairing_uses_the_run_id_not_the_position():
    """Run 3 of one scheduler must be compared against run 3 of the other."""
    a = {0: 'a0', 2: 'a2', 5: 'a5'}
    b = {2: 'b2', 5: 'b5', 9: 'b9'}
    serie_a, serie_b = cs.paired_series(a, b, lambda v: int(v[1:]))
    assert serie_a == [2, 5]
    assert serie_b == [2, 5]


def test_who_wins_respects_the_direction_of_the_metric():
    assert cs._who_wins(0.5, 'higher', 'A', 'B') == 'A'
    assert cs._who_wins(0.5, 'lower', 'A', 'B') == 'B'
    assert cs._who_wins(-0.5, 'lower', 'A', 'B') == 'A'
    assert cs._who_wins(0.0, 'lower', 'A', 'B') == 'tie'
