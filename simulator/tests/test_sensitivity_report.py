"""The report says both things about every arm, and reuses the statistics.

A configuration can score better and never settle, or settle on a policy that
scores worse. The published DynQ does the first and the published Q-static
does the second, so a table that reports only one of the two answers half the
question.
"""
from __future__ import absolute_import

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'bin'))

import sensitivity_report as sr


def test_the_statistics_are_the_ones_already_in_use():
    """Nothing new is invented for this table: a second implementation of a
    paired test would be a second thing to keep right."""
    import compare_schedulers as cs
    assert sr.paired_series is cs.paired_series
    assert sr.bootstrap_ci is cs.bootstrap_ci
    assert sr.rank_biserial is cs.rank_biserial
    assert sr.METRICS is cs.METRICS


def test_the_criterion_is_the_one_already_defined():
    import learning_report as lr
    assert sr.verdict is lr.verdict


def test_who_won_reads_the_direction_of_the_metric():
    assert sr._melhor(0.5, 'higher') == 'braco'
    assert sr._melhor(0.5, 'lower') == 'base'
    assert sr._melhor(-0.5, 'lower') == 'braco'
    assert sr._melhor(0, 'lower') == 'igual'


def test_the_three_letters_are_the_three_parts_in_order():
    v = {
        'reward_flat': {'flat': True},
        'table_still': {'settled': False},
        'policy_settled': {'settled': None},
        'converged': False,
    }
    assert sr.marca(v).startswith(' sn-')
    assert 'CONVERGIU' not in sr.marca(v)


def test_all_three_passing_reads_as_converged():
    v = {
        'reward_flat': {'flat': True},
        'table_still': {'settled': True},
        'policy_settled': {'settled': True},
        'converged': True,
    }
    assert 'CONVERGIU' in sr.marca(v)
    assert sr.marca(v).startswith(' sss')


def test_an_arm_with_no_trace_says_so_rather_than_failing():
    assert 'sem traco' in sr.marca(None)


def test_the_score_is_a_column_and_not_the_verdict():
    """Victor's call: it aggregates four quantities with weights nobody
    defended, and the agent is not optimising it."""
    assert 'score' in sr.DESTAQUE
    assert sr.DESTAQUE[0] == 'score'
    fonte = open(os.path.join(
        os.path.dirname(__file__), '..', 'bin', 'sensitivity_report.py'
    )).read()
    assert 'summary and not the finding' in fonte


def test_the_folder_of_an_arm_follows_the_driver(tmpdir):
    caminho = sr.folder_of('resultados', 'dynq_alfa_0p2', 50)
    assert caminho == os.path.join(
        'resultados', 'dynq_alfa_0p2_n50', 'exec_numMotes_50'
    )


def test_an_arm_without_results_is_reported_and_not_crashed_on(tmpdir):
    assert sr.read_arm(str(tmpdir), 'nao_existe', 50) is None
    assert sr.convergence_of(str(tmpdir), 'nao_existe', 50, 10) is None


def test_the_manifest_groups_the_arms_by_factor(tmpdir, capsys):
    """With no results at all the report still runs and says what is missing,
    rather than half-writing a table."""
    manifesto = {
        'dynq_base': {'learner': 'dynq', 'factor': 'baseline',
                      'setting': None, 'value': None, 'baseline': 'dynq_base'},
        'dynq_alfa_0p2': {'learner': 'dynq', 'factor': 'alfa',
                          'setting': 'ALFA', 'value': 0.2,
                          'baseline': 'dynq_base'},
    }
    saida = sr.report(str(tmpdir), 50, manifesto, 10, ['score'])
    assert saida == {}
    assert 'linha de base' in capsys.readouterr().out
