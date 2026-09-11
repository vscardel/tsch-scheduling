"""The paired test is computed exactly, because ten runs is a small sample.

The scipy in the image is 1.2.3, whose wilcoxon has no exact mode: it uses the
normal approximation and warns that the sample is too small for it. With ten
paired runs the difference decides the paper's claims. Ten runs all falling
the same way have an exact p of 0.00195, which survives the Holm correction
over the ten pairs of five schedulers, and an approximate p of 0.00506, which
does not.
"""
from __future__ import absolute_import
from __future__ import division

import pytest

from compare_schedulers import exact_signed_rank_p, holm


def test_ten_runs_all_one_way_is_the_smallest_p_ten_runs_can_give():
    assert exact_signed_rank_p(list(range(1, 11))) == pytest.approx(2 / 1024)
    assert exact_signed_rank_p([-d for d in range(1, 11)]) == pytest.approx(2 / 1024)


def test_that_p_survives_a_holm_correction_over_ten_pairs():
    p = exact_signed_rank_p(list(range(1, 11)))
    assert holm([p] * 10)[0] < 0.05


def test_a_textbook_case():
    """n = 8, positive ranks summing to 3, two sided p = 0.0234."""
    assert exact_signed_rank_p(
        [-1, 2, -3, -4, -5, -6, -7, -8]
    ) == pytest.approx(0.0234375)


def test_a_perfectly_balanced_sample_says_nothing():
    assert exact_signed_rank_p([1, -1, 2, -2]) == 1.0


def test_no_difference_at_all_is_not_an_error():
    assert exact_signed_rank_p([0, 0, 0]) == 1.0


def test_zero_differences_are_dropped_rather_than_counted():
    assert exact_signed_rank_p([1, 2, 3, 0]) == exact_signed_rank_p([1, 2, 3])


def test_ties_in_the_magnitudes_keep_their_mid_ranks():
    """Five equal differences: every rank is 3, so the sum can only take six
    values and the exact p is 2/32."""
    assert exact_signed_rank_p([1, 1, 1, 1, 1]) == pytest.approx(2 / 32)


def test_the_exact_test_is_used_for_the_sample_sizes_we_run():
    from compare_schedulers import EXACT_UP_TO
    assert EXACT_UP_TO >= 15        # the comparison runs 10 or 15 seeds
