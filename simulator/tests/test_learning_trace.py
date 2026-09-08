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
                        td_error=-0.25, q_table=tabela)
    assert sorted(stats['DECISIONS'].keys()) == ['1', '2', '3']
    assert stats['DECISIONS']['2'] == {
        'asn': 200, 'reward': 1.5, 'td_error': -0.25, 'policy_changes': 0
    }


def test_the_temporal_difference_is_kept_rather_than_discarded():
    """It was computed on every update in both learners and never stored."""
    stats = empty_state_stats()
    record_decision(stats, 1, asn=10, reward=0.0, td_error=3.75,
                    q_table={0: [0, 0, 0]})
    assert stats['DECISIONS']['1']['td_error'] == 3.75


def test_a_changed_preference_is_counted_and_an_unchanged_one_is_not():
    stats = empty_state_stats()
    tabela = {0: [0.0, 0.0, 0.0], 1: [0.0, 0.0, 0.0]}

    record_decision(stats, 1, 10, 0.0, 0.0, tabela)
    assert stats['DECISIONS']['1']['policy_changes'] == 0

    tabela[0][2] = 1.0                      # row 0 now prefers action 2
    record_decision(stats, 2, 20, 0.0, 0.0, tabela)
    assert stats['DECISIONS']['2']['policy_changes'] == 1

    tabela[0][2] = 2.0                      # bigger, but still action 2
    record_decision(stats, 3, 30, 0.0, 0.0, tabela)
    assert stats['DECISIONS']['3']['policy_changes'] == 0


def test_the_first_decision_has_nothing_to_have_changed_from():
    """Eight rows appearing at once is not eight changes of mind."""
    stats = empty_state_stats()
    record_decision(stats, 1, 10, 0.0, 0.0, dict(
        (linha, [0, 0, 1]) for linha in range(8)
    ))
    assert stats['DECISIONS']['1']['policy_changes'] == 0


def test_the_table_written_is_the_table_in_memory():
    """The learned policy did not survive the run at all before this."""
    tabela = {0: [1.0, 2.0, 3.0], 5: [-1.0, 0.0, 0.0]}
    stats = empty_state_stats()
    record_decision(stats, 1, 10, 0.0, 0.0, tabela)
    assert stats['Q_TABLE'] == {'0': [1.0, 2.0, 3.0], '5': [-1.0, 0.0, 0.0]}
    assert stats['GREEDY_POLICY'] == {'0': 2, '5': 1}


def test_the_table_is_the_last_one_and_not_every_one():
    """It is overwritten each decision, so it costs a table and not a history."""
    stats = empty_state_stats()
    record_decision(stats, 1, 10, 0.0, 0.0, {0: [1.0, 0.0, 0.0]})
    record_decision(stats, 2, 20, 0.0, 0.0, {0: [0.0, 0.0, 9.0]})
    assert stats['Q_TABLE'] == {'0': [0.0, 0.0, 9.0]}


def test_a_snapshot_does_not_alias_the_live_table():
    tabela = {0: [0.0, 0.0, 0.0]}
    stats = empty_state_stats()
    record_decision(stats, 1, 10, 0.0, 0.0, tabela)
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
        record_decision(stats, passo, passo, 0.0, 0.0, {0: [0, 0, 0]})
        obtido.append(random.random())
    assert obtido == esperado


def test_the_trace_survives_a_json_round_trip():
    stats = empty_state_stats()
    record_decision(stats, 1, 10, 1.5, -0.25, {0: [1.0, 2.0, 3.0]})
    assert json.loads(json.dumps(stats)) == stats
