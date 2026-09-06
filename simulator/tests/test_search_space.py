"""The search should not spend evaluations on a parameter that does nothing.

DynQ is epsilon-greedy: epsilon is the chance of exploring on each decision and
there is no threshold to cross, so EPSLON_THRESHOLD decides nothing there.
Q-static still compares epsilon against the threshold, so for it the parameter
is real. Each evaluation of the objective is a whole simulation run, so a dead
dimension is paid for in CPU hours.
"""
from __future__ import absolute_import

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'bin'))

import runExperiments as rx


def nomes(sf):
    return [nome for nome, _ in rx.search_space(sf)]


def test_dynq_does_not_search_the_threshold():
    assert 'EPSLON_THRESHOLD' not in nomes('Qlearning')


def test_q_static_still_searches_the_threshold():
    assert 'EPSLON_THRESHOLD' in nomes('QlearningSBRC24')


def test_dynq_searches_one_dimension_fewer():
    assert len(nomes('Qlearning')) == len(nomes('QlearningSBRC24')) - 1


def test_the_parameters_they_share_come_in_the_same_order():
    comuns = [n for n in nomes('QlearningSBRC24') if n != 'EPSLON_THRESHOLD']

    assert nomes('Qlearning') == comuns


def test_every_range_is_a_pair_of_bounds():
    for sf in ('Qlearning', 'QlearningSBRC24'):
        for nome, faixa in rx.search_space(sf):
            assert len(faixa) == 2
            assert faixa[0] < faixa[1]


def test_an_unknown_function_gets_the_full_space():
    # nao vale calar a dimensao para quem nao sabemos como decide
    assert nomes('SomethingElse') == nomes('QlearningSBRC24')


def test_rlsf_has_the_same_number_of_dimensions_as_dynq():
    """Neither method should be the only one that was tuned."""
    assert len(nomes('RLSF')) == len(nomes('Qlearning'))


def test_rlsf_searches_its_own_settings():
    """The names go straight into the config, so they must match RLSF.py."""
    assert nomes('RLSF') == [
        'RLSF_ALFA', 'RLSF_BETA', 'RLSF_EPSILON_DECAY', 'RLSF_EPSILON_END'
    ]


def test_no_search_touches_the_reward_weights():
    """Searching the reward changes the question, not the answer."""
    for sf in ('Qlearning', 'QlearningSBRC24', 'RLSF'):
        for nome in nomes(sf):
            assert not nome.startswith('W_')
            assert 'THETA' not in nome


def test_budget_defaults_leave_most_evaluations_to_the_model():
    n_calls, n_random = rx.optimisation_budget(None, None)
    assert n_calls > n_random
    assert n_calls - n_random >= 2 * n_random


def test_budget_refuses_a_pure_random_search():
    """n_calls equal to n_random_starts is what the published run did."""
    import pytest
    with pytest.raises(ValueError):
        rx.optimisation_budget(10, 10)
    with pytest.raises(ValueError):
        rx.optimisation_budget(5, 10)


def test_budget_honours_what_is_asked_for():
    assert rx.optimisation_budget(30, 8) == (30, 8)
