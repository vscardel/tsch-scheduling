"""The per-decision trace that lets a reviewer see whether learning happened.

Twelve reviewer comments turn on one question: does the agent learn, and how
do you know. The run used to keep a cumulative reward and nothing else, so the
answer was a curve with no variance band, no control, and no convergence
criterion, for one of the two learners.

What is protected here is the evidence itself: that every decision leaves a
record, that the record holds the temporal difference the update already
computed, that the table the run ended with is recoverable, and above all that
recording changed nothing about the run being recorded.
"""
import json
import random

from SimEngine.Mote.scheduling_functions.state_visits import (
    empty_state_stats, greedy_action, greedy_policy, record_decision
)


def test_a_fresh_agent_has_decided_nothing():
    stats = empty_state_stats()
    assert stats['DECISIONS'] == {}
    assert stats['Q_TABLE'] == {}
    assert stats['GREEDY_POLICY'] == {}


def test_one_record_per_decision():
    stats = empty_state_stats()
    tabela = {0: [0, 0, 0], 1: [0, 0, 0]}
    for passo in range(1, 4):
        record_decision(stats, passo, asn=passo * 100, reward=1.5,
                        td_error=-0.25, delta_q=-0.1, q_table=tabela)
    assert sorted(stats['DECISIONS'].keys()) == ['1', '2', '3']
    assert stats['DECISIONS']['2'] == {
        'asn': 200, 'reward': 1.5, 'td_error': -0.25, 'delta_q': -0.1,
        'policy_changes': 0, 'policy_code': 0
    }


def test_the_temporal_difference_is_kept_rather_than_discarded():
    """It was computed on every update in both learners and never stored."""
    stats = empty_state_stats()
    record_decision(stats, 1, asn=10, reward=0.0, td_error=3.75,
                    delta_q=0.5, q_table={0: [0, 0, 0]})
    assert stats['DECISIONS']['1']['td_error'] == 3.75


def test_a_changed_preference_is_counted_and_an_unchanged_one_is_not():
    stats = empty_state_stats()
    tabela = {0: [0.0, 0.0, 0.0], 1: [0.0, 0.0, 0.0]}

    record_decision(stats, 1, 10, 0.0, 0.0, 0.0, tabela)
    assert stats['DECISIONS']['1']['policy_changes'] == 0

    tabela[0][2] = 1.0                      # row 0 now prefers action 2
    record_decision(stats, 2, 20, 0.0, 0.0, 0.0, tabela)
    assert stats['DECISIONS']['2']['policy_changes'] == 1

    tabela[0][2] = 2.0                      # bigger, but still action 2
    record_decision(stats, 3, 30, 0.0, 0.0, 0.0, tabela)
    assert stats['DECISIONS']['3']['policy_changes'] == 0


def test_the_first_decision_has_nothing_to_have_changed_from():
    """Eight rows appearing at once is not eight changes of mind."""
    stats = empty_state_stats()
    record_decision(stats, 1, 10, 0.0, 0.0, 0.0, dict(
        (linha, [0, 0, 1]) for linha in range(8)
    ))
    assert stats['DECISIONS']['1']['policy_changes'] == 0


def test_the_table_written_is_the_table_in_memory():
    """The learned policy did not survive the run at all before this."""
    tabela = {0: [1.0, 2.0, 3.0], 5: [-1.0, 0.0, 0.0]}
    stats = empty_state_stats()
    record_decision(stats, 1, 10, 0.0, 0.0, 0.0, tabela)
    assert stats['Q_TABLE'] == {'0': [1.0, 2.0, 3.0], '5': [-1.0, 0.0, 0.0]}
    assert stats['GREEDY_POLICY'] == {'0': 2, '5': 1}


def test_the_table_is_the_last_one_and_not_every_one():
    """It is overwritten each decision, so it costs a table and not a history."""
    stats = empty_state_stats()
    record_decision(stats, 1, 10, 0.0, 0.0, 0.0, {0: [1.0, 0.0, 0.0]})
    record_decision(stats, 2, 20, 0.0, 0.0, 0.0, {0: [0.0, 0.0, 9.0]})
    assert stats['Q_TABLE'] == {'0': [0.0, 0.0, 9.0]}


def test_a_snapshot_does_not_alias_the_live_table():
    tabela = {0: [0.0, 0.0, 0.0]}
    stats = empty_state_stats()
    record_decision(stats, 1, 10, 0.0, 0.0, 0.0, tabela)
    tabela[0][1] = 99.0
    assert stats['Q_TABLE'] == {'0': [0.0, 0.0, 0.0]}


def test_a_tie_prefers_the_lowest_action_as_argmax_does():
    assert greedy_action([0, 0, 0]) == 0
    assert greedy_action([1.0, 1.0, 0.5]) == 0
    assert greedy_policy({3: [0.0, 5.0, 5.0]}) == {'3': 1}


def test_recording_draws_no_random_numbers():
    """An instrumented run has to score exactly what the same run scores
    without the instrumentation, and the stream of random numbers is what
    decides that."""
    random.seed(12345)
    esperado = [random.random() for _ in range(5)]

    random.seed(12345)
    stats = empty_state_stats()
    obtido = []
    for passo in range(5):
        record_decision(stats, passo, passo, 0.0, 0.0, 0.0, {0: [0, 0, 0]})
        obtido.append(random.random())
    assert obtido == esperado


def test_the_trace_survives_a_json_round_trip():
    stats = empty_state_stats()
    record_decision(stats, 1, 10, 1.5, -0.25, 0.0, {0: [1.0, 2.0, 3.0]})
    assert json.loads(json.dumps(stats)) == stats


# ------------------------------------------------- the switch, in both learners

import pytest

from SimEngine.Mote import MoteDefines as d


def agente(sim_engine, monkeypatch, sf_class, learned=None):
    """A mote of the requested learner, with 6P and the parent stubbed out."""
    config = {
        'exec_numMotes'         : 4,
        'sf_class'              : sf_class,
        'factorial_combinations': ['traffic', 'queue', 'charge'],
    }
    if learned is not None:
        config['LEARNED_POLICY'] = learned
    engine = sim_engine(diff_config=config)
    mote = engine.motes[1]
    mote.sf.start()

    monkeypatch.setattr(mote.rpl, 'getPreferredParent', lambda: 'pai')
    monkeypatch.setattr(mote, 'clear_to_send_EBs_DATA', lambda: True)
    monkeypatch.setattr(mote.sf, 'sixp_interface_add', lambda **kw: None)
    monkeypatch.setattr(mote.sf, 'sixp_interface_delete', lambda **kw: None)
    return mote.sf


def decide(sf):
    """One decision, whichever signature this learner's entry point has."""
    if sf.__class__.__name__.endswith('SBRC24'):
        sf.adapt_to_traffic([d.CELLOPTION_TX])
    else:
        sf.adapt_to_traffic([d.CELLOPTION_TX], None, 'cells')


@pytest.fixture(params=['Qlearning', 'QlearningSBRC24'])
def sf_class(request):
    return request.param


def test_learning_is_on_unless_someone_turns_it_off(sim_engine, monkeypatch,
                                                    sf_class):
    """Every run recorded so far was a learned run, and stays one."""
    sf = agente(sim_engine, monkeypatch, sf_class)
    assert sf.LEARNED_POLICY is True


def test_the_control_arm_never_consults_the_table(sim_engine, monkeypatch,
                                                  sf_class):
    """Same key, same meaning, in both learners."""
    sf = agente(sim_engine, monkeypatch, sf_class, learned=False)

    def nao_deveria(state):
        raise AssertionError('the control arm read the Q-table')
    monkeypatch.setattr(sf, 'return_best_q_action', nao_deveria)

    for _ in range(20):
        decide(sf)

    assert sf.QLEARNING_STATS['STATE_VISITS']


def test_the_control_arm_still_learns_from_what_it_did(sim_engine, monkeypatch,
                                                       sf_class):
    """The table goes on being updated, so what the two arms differ in is
    acting on it and nothing else."""
    sf = agente(sim_engine, monkeypatch, sf_class, learned=False)
    monkeypatch.setattr(sf, 'return_best_q_action', lambda state: 0)

    for _ in range(20):
        decide(sf)

    assert any(
        registro['td_error'] != 0
        for registro in sf.QLEARNING_STATS['DECISIONS'].values()
    )


def test_the_control_arm_reports_no_greedy_choices(sim_engine, monkeypatch,
                                                   sf_class):
    """Otherwise the greedy share of a control run would read as policy."""
    sf = agente(sim_engine, monkeypatch, sf_class, learned=False)
    for _ in range(20):
        decide(sf)

    for linha in sf.QLEARNING_STATS['STATE_ACTION'].values():
        assert 'greedy' not in linha


def test_the_learned_arm_takes_the_action_its_table_prefers(sim_engine,
                                                            monkeypatch,
                                                            sf_class):
    sf = agente(sim_engine, monkeypatch, sf_class, learned=True)
    monkeypatch.setattr(sf, 'return_best_q_action', lambda state: 1)
    if sf.__class__.__name__.endswith('SBRC24'):
        sf.EPSLON_THRESHOLD = 1.0       # below the threshold is the greedy phase
    else:
        sf.MIN_EPSLON = sf.MAX_EPSLON = 0.0

    for _ in range(5):
        decide(sf)

    assert sf.last_action == 1


def test_every_decision_leaves_a_record_in_both(sim_engine, monkeypatch,
                                                sf_class):
    sf = agente(sim_engine, monkeypatch, sf_class)
    for _ in range(10):
        decide(sf)

    decisoes = sf.QLEARNING_STATS['DECISIONS']
    # the first decision has no reward yet: the reward of an action is only
    # knowable from the state it leads to
    assert len(decisoes) == 9
    for registro in decisoes.values():
        assert sorted(registro.keys()) == [
            'asn', 'delta_q', 'policy_changes', 'policy_code', 'reward',
            'td_error'
        ]


def test_the_run_ends_with_its_table_on_record(sim_engine, monkeypatch,
                                               sf_class):
    sf = agente(sim_engine, monkeypatch, sf_class)
    for _ in range(10):
        decide(sf)

    gravada = sf.QLEARNING_STATS['Q_TABLE']
    assert gravada == dict(
        (str(linha), list(valores)) for linha, valores in sf.Q_table.items()
    )


def test_the_trace_puts_both_learners_on_the_same_clock(sim_engine, monkeypatch,
                                                        sf_class):
    """They decide on different triggers, so step numbers are not comparable
    across the two and ASN is what is."""
    sf = agente(sim_engine, monkeypatch, sf_class)
    for _ in range(10):
        decide(sf)

    asns = [r['asn'] for r in sf.QLEARNING_STATS['DECISIONS'].values()]
    assert asns and all(asn == sf.engine.getAsn() for asn in asns)


# ------------------------------------------------------ the policy as one int

def test_the_policy_travels_as_one_integer():
    """Rows in numerical order, each a digit in base the number of actions."""
    from SimEngine.Mote.scheduling_functions.state_visits import policy_code
    # row 0 prefers 2, row 1 prefers 0, row 2 prefers 1
    tabela = {0: [0, 0, 1.0], 1: [1.0, 0, 0], 2: [0, 1.0, 0]}
    assert policy_code(tabela) == 2 + 0 * 3 + 1 * 9


def test_the_code_distinguishes_every_policy_it_can_hold():
    """Otherwise two different policies would look like no change of mind."""
    from SimEngine.Mote.scheduling_functions.state_visits import policy_code
    import itertools
    vistos = set()
    for acoes in itertools.product(range(3), repeat=3):
        tabela = dict(
            (linha, [1.0 if i == a else 0.0 for i in range(3)])
            for linha, a in enumerate(acoes)
        )
        vistos.add(policy_code(tabela))
    assert len(vistos) == 27


def test_a_level_row_reads_as_the_lowest_action_like_argmax():
    from SimEngine.Mote.scheduling_functions.state_visits import policy_code
    assert policy_code({0: [0.0, 0.0, 0.0]}) == 0
    assert policy_code({}) == 0


# ------------------------------------- the learning rate, wired into both

def taxas(sf):
    """The rate each update actually used, recovered from what was recorded."""
    saida = []
    for registro in sf.QLEARNING_STATS['DECISIONS'].values():
        if registro['td_error']:
            saida.append(registro['delta_q'] / registro['td_error'])
    return saida


def test_a_constant_rate_is_what_runs_get_unless_asked(sim_engine, monkeypatch,
                                                       sf_class):
    """Every result measured so far used a constant rate and still does."""
    sf = agente(sim_engine, monkeypatch, sf_class)
    assert sf.ALFA_DECAY_TAU == 0
    for _ in range(15):
        decide(sf)
    usadas = taxas(sf)
    assert usadas
    for alfa in usadas:
        assert alfa == pytest.approx(sf.ALFA)


def test_a_decaying_rate_shrinks_as_a_cell_is_revisited(sim_engine, monkeypatch,
                                                        sf_class):
    sf = agente(sim_engine, monkeypatch, sf_class)
    sf.ALFA_DECAY_TAU = 1
    for _ in range(30):
        decide(sf)

    usadas = taxas(sf)
    assert usadas
    assert max(usadas) <= sf.ALFA + 1e-12
    assert min(usadas) < sf.ALFA           # some cell was revisited
    assert sf.ALFA_VISITS                  # counted per state-action pair
