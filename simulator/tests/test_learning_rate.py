"""The learning rate schedule, and what it is for.

Watkins and Dayan need the rate to shrink. A constant one fails that condition
and the failure is not academic: the estimate never settles, it tracks, moving
by alpha times the noise of the last sample forever. The published DynQ
configuration uses 0.79, so its table is very nearly rewritten on every visit,
which is what its policy churn of 0.20 rows per decision looks like from the
inside.

What is protected here is that the schedule exists, that it counts per
state-action pair as the condition is stated, and above all that leaving it
off reproduces every run made before it existed.
"""
from __future__ import absolute_import

import pytest

from SimEngine.Mote.scheduling_functions.learning_rate import step_size


def test_off_by_default_is_the_constant_rate():
    """Every run recorded so far used a constant rate and still does."""
    visitas = {}
    for _ in range(100):
        assert step_size(0.5, 0, visitas, ('s', 'a')) == 0.5


def test_the_first_update_to_a_cell_takes_the_full_rate():
    """However small tau is. The count is of updates already applied."""
    assert step_size(0.8, 1, {}, ('s', 'a')) == 0.8


def test_the_nth_update_takes_alpha_over_one_plus_n_over_tau():
    visitas = {}
    chave = ('s', 'a')
    obtidos = [step_size(1.0, 2, visitas, chave) for _ in range(4)]
    assert obtidos == pytest.approx([1.0, 1 / 1.5, 1 / 2.0, 1 / 2.5])


def test_the_count_is_per_cell_and_not_global():
    """The condition is on visits to a state-action pair. Counting globally
    would shrink the rate of a cell being seen for the first time because
    other cells had been busy."""
    visitas = {}
    step_size(1.0, 1, visitas, ('s', 0))
    step_size(1.0, 1, visitas, ('s', 0))
    assert step_size(1.0, 1, visitas, ('s', 1)) == 1.0


def test_a_shrinking_rate_still_adds_up_to_infinity():
    """The other half of the condition. A rate that dies too fast freezes the
    estimate before it arrives, so alpha/n is required to remain unsummable."""
    visitas = {}
    chave = ('s', 'a')
    soma = sum(step_size(1.0, 10, visitas, chave) for _ in range(20000))
    assert soma > 50            # grows without bound, slowly

    visitas = {}
    quadrados = sum(
        step_size(1.0, 10, visitas, chave) ** 2 for _ in range(20000)
    )
    assert quadrados < 100      # and its square settles


def test_the_rate_never_goes_negative_or_grows():
    visitas = {}
    chave = ('s', 'a')
    anterior = None
    for _ in range(50):
        atual = step_size(0.3, 5, visitas, chave)
        assert 0 < atual <= 0.3
        if anterior is not None:
            assert atual < anterior
        anterior = atual
