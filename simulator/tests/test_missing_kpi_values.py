"""A statistic the simulator could not compute is absent, not a number.

compute_kpis writes the string 'N/A' wherever there is no sample: a run in
which nothing arrived has no latency and no delivery ratio. On the Linear
topology, where the far motes may never join, whole runs come out that way.
Python 2 compares a string as greater than any float, so reading one as a
number does not raise; it silently scores a run on a value that does not
exist, and the run wins.
"""
from __future__ import absolute_import

import pytest

from score_model import as_number, run_metrics, run_score
from compare_schedulers import _latency, _pdr, _join, _lifetime_min


def run_with(latency='N/A', pdr='N/A', join='N/A', lifetime_min='N/A'):
    return {
        'global-stats': {
            'e2e-upstream-latency' : [{'mean' : latency}],
            'e2e-upstream-delivery': [{'value': pdr}],
            'joining-time'         : [{'mean' : join}],
            'network_lifetime'     : [{'min'  : lifetime_min}],
        },
        '1': {'lifetime_AA_years': 1.5},
    }


@pytest.mark.parametrize('valor', ['N/A', None, u'N/A', True])
def test_a_value_that_is_not_a_number_reads_as_absent(valor):
    assert as_number(valor) is None


def test_a_number_reads_as_itself_and_can_be_scaled():
    assert as_number(5) == 5.0
    assert as_number(500.0, 100.0) == 5.0


def test_every_reader_reports_an_absent_value_as_absent():
    run = run_with()
    for reader in (_latency, _pdr, _join, _lifetime_min):
        assert reader(run) is None


@pytest.mark.parametrize('ausente', ['latency', 'pdr', 'join'])
def test_a_run_missing_a_global_statistic_cannot_be_scored(ausente):
    completo = {'latency': 1.0, 'pdr': 0.9, 'join': 500.0, 'lifetime_min': 1.0}
    completo[ausente] = 'N/A'
    assert run_score(run_with(**completo)) is None


def test_the_lifetime_of_a_run_with_no_estimate_counts_as_zero():
    """Not absent: a mote whose lifetime could not be estimated is a mote that
    drains its battery, and the score is meant to feel that."""
    run = run_with(latency=1.0, pdr=0.9, join=500.0, lifetime_min=1.0)
    run['1'] = {'lifetime_AA_years': 'N/A'}
    assert run_metrics(run)['lifetime'] == 0.0
    assert run_score(run) is not None


def test_a_complete_run_is_scored_as_before():
    run = run_with(latency=1.0, pdr=0.9, join=500.0, lifetime_min=1.0)
    valor = run_score(run)
    assert valor is not None
    assert 0.0 <= valor <= 1.0


def test_the_join_time_is_still_divided_by_a_hundred():
    run = run_with(latency=1.0, pdr=0.9, join=500.0, lifetime_min=1.0)
    assert run_metrics(run)['join_time'] == 5.0
