"""The score of a run has to depend on that run.

compute_score called compute_average_lifetime(kpis) from inside its per-run
loop, so every run received the mean lifetime over all of them. A quarter of the
score was then the same constant for every run and could not tell them apart.
That quarter is also what the Bayesian optimisation is handed.
"""
from __future__ import absolute_import

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'bin'))

import runExperiments as rx


def run(lifetimes, latency=1.0, pdr=0.9, join=50000.0):
    """One run's KPIs, with one mote per lifetime given."""
    kpis = {
        'global-stats': {
            'e2e-upstream-latency':  [{'mean': latency}],
            'e2e-upstream-delivery': [{'value': pdr}],
            'joining-time':          [{'mean': join}],
        }
    }
    for i, years in enumerate(lifetimes):
        kpis[str(i)] = {'lifetime_AA_years': years}
    return kpis


def test_a_run_lifetime_is_the_mean_over_its_motes():
    assert rx.compute_run_lifetime(run([1.0, 2.0, 3.0])) == pytest.approx(2.0)


def test_a_mote_without_an_estimate_counts_as_zero():
    # the simulator writes a string when it cannot estimate a lifetime
    assert rx.compute_run_lifetime(
        run([4.0, u'N/A'])
    ) == pytest.approx(2.0)
    assert rx.compute_run_lifetime(run([])) == 0.0


def test_the_average_is_taken_over_the_runs():
    kpis = {'0': run([1.0]), '1': run([3.0])}

    assert rx.compute_average_lifetime(kpis) == pytest.approx(2.0)


def test_a_longer_lifetime_gives_a_better_score():
    short = rx.compute_score({'0': run([0.1])})[0]
    long_lived = rx.compute_score({'0': run([5.0])})[0]

    # a lower score is better
    assert long_lived < short


def test_two_runs_that_differ_only_in_lifetime_get_different_scores():
    # before the fix both runs were handed the mean of the two and came out equal
    first, second = rx.compute_score({'0': run([0.1]), '1': run([5.0])})

    assert first != second


def test_the_lifetime_of_one_run_does_not_move_another():
    # scores come back in dictionary order, so compare them as a set
    apart = (
        rx.compute_score({'0': run([0.1])}) +
        rx.compute_score({'0': run([5.0])})
    )
    together = rx.compute_score({'0': run([0.1]), '1': run([5.0])})

    assert sorted(together) == pytest.approx(sorted(apart))
