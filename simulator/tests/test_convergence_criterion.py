"""The convergence criterion, stated so a reviewer can check it.

Two reviewers asked for a convergence criterion and the manuscript has none.
The asymptotic guarantee is not available here: Watkins and Dayan need the
learning rate to shrink, which no constant rate does, and every state-action
pair to be visited without end, which the static learner stops doing the
moment its epsilon crosses the threshold. So the claim is not made, and this
practical criterion is used instead.

Three parts, never collapsed into one number, because failing one and passing
two says something different in each case.
"""
from __future__ import absolute_import

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'bin'))

import learning_report as lr


def mote(decisoes, visitas=None, tabela=None):
    return {
        'DECISIONS': dict((str(i + 1), d) for i, d in enumerate(decisoes)),
        'STATE_VISITS': visitas or {},
        'STATE_ACTION': {},
        'Q_TABLE': tabela or {},
        'GREEDY_POLICY': {},
    }


def decisao(asn, reward=0.0, td_error=0.0, delta_q=0.0, policy_code=0):
    return {'asn': asn, 'reward': reward, 'td_error': td_error,
            'delta_q': delta_q, 'policy_changes': 0, 'policy_code': policy_code}


def runs_de(construir, quantas=5):
    return dict(('run_%d' % i, [construir(i)]) for i in range(quantas))


# --------------------------------------------------------- (a) the flat curve

def test_a_curve_that_stopped_rising_is_flat():
    runs = runs_de(lambda i: mote([
        decisao(asn, reward=2.0) for asn in range(0, 100, 5)
    ]))
    a = lr.flatness(runs, span=100, bins=9)
    assert a['flat'] is True
    assert a['level'] == 2.0


def test_a_curve_still_rising_is_not_flat():
    """The manuscript reads an upward trend as evidence of learning. An
    upward trend is exactly what has not converged yet."""
    runs = runs_de(lambda i: mote([
        decisao(asn, reward=asn / 10.0) for asn in range(0, 100, 5)
    ]))
    a = lr.flatness(runs, span=100, bins=9)
    assert a['flat'] is False
    assert a['change'] > 0


def test_the_level_is_reported_beside_the_flatness():
    """Q-static's reward levels off below zero on a long run. Flat is not the
    same as converged on something worth having."""
    runs = runs_de(lambda i: mote([
        decisao(asn, reward=-0.5) for asn in range(0, 100, 5)
    ]))
    a = lr.flatness(runs, span=100, bins=9)
    assert a['flat'] is True
    assert a['level'] == -0.5


# ------------------------------------------------------- (b) the still table

def test_updates_smaller_than_a_hundredth_of_the_reward_scale_count_as_still():
    runs = runs_de(lambda i: mote([
        decisao(asn, reward=asn % 3, delta_q=0.0001)
        for asn in range(0, 100, 2)
    ]))
    b = lr.step_sizes(runs, span=100, bins=9)
    assert b['settled'] is True
    assert b['scale'] > 0


def test_a_table_still_jumping_is_not_still():
    """DynQ updates by 0.156 against a reward scale of about 3."""
    runs = runs_de(lambda i: mote([
        decisao(asn, reward=asn % 3, delta_q=0.156)
        for asn in range(0, 100, 2)
    ]))
    assert lr.step_sizes(runs, span=100, bins=9)['settled'] is False


def test_the_scale_is_the_spread_and_not_the_mean():
    """The mean reward passes through zero on a long Q-static run, and a
    tolerance relative to the mean would blow up exactly there."""
    runs = runs_de(lambda i: mote([
        decisao(asn, reward=(1.0 if asn % 4 else -1.0), delta_q=0.001)
        for asn in range(0, 100, 2)
    ]))
    b = lr.step_sizes(runs, span=100, bins=9)
    assert b['scale'] == 2.0
    assert b['settled'] is True


# ---------------------------------------------------- (c) the settled policy

# keyed by string, as the simulator writes it and as JSON reads it back
TABELA = dict((str(linha), [0.0, 0.0, 0.0]) for linha in range(3))
VISITAS = dict((str(linha), 5) for linha in range(3))


def test_a_policy_that_stops_changing_reports_when_it_stopped():
    def constroi(_):
        # muda ate metade da rodada, depois fica
        decisoes = [decisao(asn, policy_code=asn) for asn in range(0, 50, 10)]
        decisoes += [decisao(asn, policy_code=7) for asn in range(50, 100, 10)]
        return mote(decisoes, VISITAS, TABELA)
    v = lr.verdict(runs_de(constroi), bins=10)
    c = v['policy_settled']
    assert c['settled'] is True
    assert 0.4 <= c['from_fraction'] <= 0.6


def test_a_policy_still_changing_at_the_end_never_settled():
    def constroi(_):
        return mote(
            [decisao(asn, policy_code=asn // 10) for asn in range(0, 100, 10)],
            VISITAS, TABELA
        )
    c = lr.verdict(runs_de(constroi), bins=10)['policy_settled']
    assert c['settled'] is False


def test_only_the_rows_the_agent_stood_in_are_judged():
    """A row never reached holds whatever its tie-break gave it, and counting
    it as settled would describe the initialisation."""
    codigo = lr.decode_policy(7, 3, 3)
    assert codigo == [1, 2, 0]
    assert lr.decode_policy(0, 3, 3) == [0, 0, 0]


# ------------------------------------------------------------- the verdict

def test_all_three_must_hold():
    def bom(_):
        return mote(
            [decisao(asn, reward=2.0 + (asn % 3), delta_q=0.0001,
                     policy_code=5)
             for asn in range(0, 100, 5)],
            VISITAS, TABELA
        )
    v = lr.verdict(runs_de(bom), bins=10)
    assert v['converged'] is True

    def ruim(_):
        # tudo igual, so que a tabela ainda pula
        return mote(
            [decisao(asn, reward=2.0 + (asn % 3), delta_q=1.0, policy_code=5)
             for asn in range(0, 100, 5)],
            VISITAS, TABELA
        )
    v = lr.verdict(runs_de(ruim), bins=10)
    assert v['converged'] is False
    assert v['reward_flat']['flat'] is True
    assert v['policy_settled']['settled'] is True
    assert v['table_still']['settled'] is False


def test_a_trace_without_the_new_fields_does_not_crash():
    """The runs of 2026-09-08 predate delta_q and policy_code."""
    antigo = {'asn': 10, 'reward': 1.0, 'td_error': 0.5, 'policy_changes': 0}
    runs = runs_de(lambda i: mote([antigo, dict(antigo, asn=90)]))
    v = lr.verdict(runs, bins=6)
    assert v['table_still']['settled'] is None
    assert v['policy_settled']['settled'] is None
    assert v['converged'] is False


def test_a_reward_that_never_varies_is_undecidable_not_failed():
    """Nothing to be a hundredth of, and nothing to learn either."""
    runs = runs_de(lambda i: mote(
        [decisao(asn, reward=1.0, delta_q=0.0) for asn in range(0, 100, 5)]
    ))
    assert lr.step_sizes(runs, span=100, bins=9)['settled'] is None
