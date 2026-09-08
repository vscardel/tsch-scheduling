"""The reader that turns the trace into the figure and the criterion.

The existing plotting script fills a window a mote took no decision in with
zero. On a cumulative curve that is not a gap, it is a fall to nothing, and
averaging several motes with different gaps drags the whole curve down. That
defect is the reason Figure 6 cannot simply be redrawn from the old code, so
the fill rule is pinned here.
"""
from __future__ import absolute_import

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'bin'))

import learning_report as lr


def mote(decisoes, visitas=None, tabela=None):
    """A per-mote statistics file, as the simulator writes it."""
    return {
        'DECISIONS': dict(
            (str(i + 1), d) for i, d in enumerate(decisoes)
        ),
        'STATE_VISITS': visitas or {},
        'STATE_ACTION': {},
        'Q_TABLE': tabela or {},
        'GREEDY_POLICY': {},
    }


def decisao(asn, reward=0.0, td_error=0.0, policy_changes=0):
    return {'asn': asn, 'reward': reward, 'td_error': td_error,
            'policy_changes': policy_changes}


# --------------------------------------------------------------- the fill

def test_a_gap_carries_the_last_value_instead_of_dropping_to_zero():
    assert lr.forward_fill([1.0, None, None, 4.0]) == [1.0, 1.0, 1.0, 4.0]


def test_a_mote_that_had_not_started_contributes_nothing_yet():
    """Leading gaps stay empty. A mote that has not joined the network has no
    reward to carry forward, and calling it zero would be an opinion."""
    assert lr.forward_fill([None, None, 3.0]) == [None, None, 3.0]


def test_a_cumulative_curve_does_not_sag_over_a_quiet_window():
    m = mote([decisao(0, reward=1.0), decisao(100, reward=1.0)])
    serie = lr.mote_series(m, span=100, bins=5)
    assert serie['cumulative'] == [1.0, 1.0, 1.0, 1.0, 2.0]


def test_a_run_averages_over_the_motes_that_have_data():
    m1 = mote([decisao(0, reward=2.0)])
    m2 = mote([decisao(100, reward=4.0)])
    serie = lr.run_series([m1, m2], span=100, bins=2)
    # first window: only m1 has decided, so the average is its own value
    assert serie['reward'] == [2.0, 3.0]


# --------------------------------------------------- the convergence criterion

def test_a_policy_that_settles_reports_when_it_settled():
    churn = [2.0, 1.0, 0.0, 0.0, 0.0]
    assert lr.convergence(churn, bins=5) == 2


def test_a_policy_still_moving_at_the_end_never_settled():
    churn = [2.0, 1.0, 0.0, 0.0, 1.0]
    assert lr.convergence(churn, bins=5) is None


def test_an_empty_window_is_not_a_change_of_mind():
    churn = [1.0, None, 0.0, 0.0]
    assert lr.convergence(churn, bins=4) == 1


# ------------------------------------------------------------- the final table

def test_only_rows_the_agent_stood_in_are_judged():
    """An untouched row holds the zeros it was created with, and counting it
    as 'no preference' would describe the initialisation, not the learning."""
    runs = {'run_0': [mote(
        [decisao(0)],
        visitas={'0': 5},
        tabela={'0': [0.0, 3.0, 1.0], '1': [0.0, 0.0, 0.0]}
    )]}
    resumo = lr.table_summary(runs)
    assert resumo['rows'] == 1
    assert resumo['mean_separation'] == 3.0
    assert resumo['share_with_preference'] == 1.0


def test_a_row_with_two_equal_best_actions_has_no_preference():
    """Which is what a reward that does not depend on the action leaves
    behind, and a reviewer asked about exactly that."""
    runs = {'run_0': [mote(
        [decisao(0)], visitas={'3': 1}, tabela={'3': [5.0, 5.0, 0.0]}
    )]}
    assert lr.table_summary(runs)['share_with_preference'] == 0.0


# ------------------------------------------------------------------ the band

def test_the_band_is_taken_over_runs_and_not_over_motes():
    """Motes inside one run share a topology and a seed. Runs do not."""
    por_run = [[1.0], [3.0], [5.0]]
    media, baixo, alto = lr.band(por_run, resamples=200)
    assert media == [3.0]
    assert baixo[0] <= 3.0 <= alto[0]


def test_a_window_no_run_reached_stays_empty():
    media, baixo, alto = lr.band([[None], [None]], resamples=50)
    assert media == [None] and baixo == [None] and alto == [None]


# ------------------------------------------------------------- the whole read

def test_a_folder_of_runs_reads_end_to_end(tmpdir):
    for run in range(3):
        for mote_id in range(2):
            destino = tmpdir.join('exec_numMotes_4', 'run_%d' % run,
                                  str(mote_id))
            destino.ensure(dir=True)
            destino.join('qlearning_stats.json').write(json_of(mote([
                decisao(asn, reward=1.0, td_error=0.5, policy_changes=0)
                for asn in (0, 50, 100)
            ], visitas={'0': 3}, tabela={'0': [1.0, 0.0, 0.0]})))

    runs = lr.load_runs(str(tmpdir))
    assert sorted(runs) == ['run_0', 'run_1', 'run_2']
    assert lr.span_of(runs) == 100

    resumo = lr.summarise('teste', runs, bins=4)
    assert resumo['runs'] == 3
    assert resumo['motes'] == 6
    assert resumo['decisions'] == 18
    assert abs(resumo['mean_reward'] - 1.0) < 1e-9
    assert resumo['converged_at_bin'] == 0
    assert resumo['table']['rows'] == 6


def json_of(objeto):
    import json
    return json.dumps(objeto)


# ------------------------------------- the per-run numbers the statistics use

def test_one_number_per_run_for_the_paired_statistics(tmpdir):
    for run in range(2):
        destino = tmpdir.join('exec_numMotes_4', 'run_%d' % run, '1')
        destino.ensure(dir=True)
        destino.join('qlearning_stats.json').write(json_of(mote(
            [decisao(0, reward=float(run)), decisao(10, reward=float(run))],
            visitas={'0': 2}, tabela={'0': [2.0, 0.0, 0.0]}
        )))

    metricas = lr.per_run_metrics(str(tmpdir))
    assert sorted(metricas) == [0, 1]          # keyed the way .kpi runs are
    assert metricas[1]['mean_reward'] == 1.0
    assert metricas[1]['final_policy_gap'] == 2.0


def test_a_folder_without_learners_yields_nothing_to_pair(tmpdir):
    """MSF and EMSF write no trace, and the pair is dropped rather than
    given a zero that would read as a real value."""
    assert lr.per_run_metrics(str(tmpdir)) == {}
