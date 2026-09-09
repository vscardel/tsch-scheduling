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


def test_a_stretch_no_run_reached_is_not_a_measurement_of_zero():
    """Motes decide nothing before they join, so early windows are empty.
    A mean of zero there would put a settled temporal difference and a reward
    of zero where there is no measurement at all."""
    assert lr.media_de([None, None]) is None
    assert lr.media_de([None, 2.0, 4.0]) == 3.0


# --------------------------------------------- tracking a disturbance

def trace_json(pares, every=10, slotframe=101):
    import json
    t = {'asn': [], 'sent': [], 'received': [], 'every': every}
    for i, (enviados, recebidos) in enumerate(pares):
        t['asn'].append(i * slotframe * every)
        t['sent'].append(enviados)
        t['received'].append(recebidos)
    return json.dumps(t)


def escrever_traces(tmpdir, pares, runs=3):
    for run in range(runs):
        destino = tmpdir.join('exec_numMotes_4', 'run_%d' % run)
        destino.ensure(dir=True)
        destino.join('network_trace.json').write(trace_json(pares))


def test_the_traces_are_read_per_run(tmpdir):
    escrever_traces(tmpdir, [(100 * i, 80 * i) for i in range(20)])
    traces = lr.load_network_traces(str(tmpdir))
    assert sorted(traces) == ['run_0', 'run_1', 'run_2']


def test_a_disturbance_that_did_not_bite_reads_as_no_fall(tmpdir):
    """Steady delivery all the way through: the disturbance never reached
    this configuration, which is a result and not a recovery."""
    escrever_traces(tmpdir, [(100 * i, 80 * i) for i in range(20)])
    linhas = lr.tracking(lr.load_network_traces(str(tmpdir)), [0.5])
    assert linhas[0]['drop'] == 0.0
    assert linhas[0]['runs'] == 3


def test_a_fall_and_a_return_are_both_reported(tmpdir):
    antes = [(100 * i, 80 * i) for i in range(1, 12)]      # 80%
    caido = [(1200, 920)]                                   # 40%
    volta = [(1200 + 100 * i, 920 + 80 * i) for i in range(1, 6)]
    escrever_traces(tmpdir, [(0, 0)] + antes + caido + volta)

    traces = lr.load_network_traces(str(tmpdir))
    fim = max(list(traces.values())[0]['asn'])
    fracao = (12 * 101 * 10) / float(fim)
    linhas = lr.tracking(traces, [fracao])
    assert linhas[0]['drop'] > 0.4
    assert linhas[0]['recovered'] == 3
    assert linhas[0]['never_recovered'] == 0


def test_an_arm_that_never_comes_back_is_counted(tmpdir):
    antes = [(100 * i, 80 * i) for i in range(1, 12)]
    depois = [(1100 + 100 * i, 880 + 20 * i) for i in range(1, 8)]
    escrever_traces(tmpdir, [(0, 0)] + antes + depois)

    traces = lr.load_network_traces(str(tmpdir))
    fim = max(list(traces.values())[0]['asn'])
    fracao = (12 * 101 * 10) / float(fim)
    linhas = lr.tracking(traces, [fracao])
    assert linhas[0]['never_recovered'] == 3
    assert linhas[0]['recovery_asn'] is None


def test_no_traces_means_no_tracking_rather_than_a_crash(tmpdir):
    """Runs made before the trace existed, and every unperturbed arm."""
    assert lr.tracking({}, [0.4, 0.7]) == [
        {'fraction': 0.4, 'runs': 0, 'drop': None, 'recovery_asn': None,
         'recovered': 0, 'never_recovered': 0},
        {'fraction': 0.7, 'runs': 0, 'drop': None, 'recovery_asn': None,
         'recovered': 0, 'never_recovered': 0},
    ]
