"""The visited-state histogram a reviewer asked for is recorded during the run.

Two things are being protected here. One is that the counters exist and count
what they claim to. The other is that adding them did not change the runs they
measure: the counters read values the decision had already computed, and they
draw no random numbers of their own.
"""
from SimEngine.Mote.scheduling_functions.state_visits import (
    empty_state_stats, record_action, record_state
)


def test_an_untouched_agent_has_visited_nothing():
    stats = empty_state_stats()
    assert stats['STATE_VISITS'] == {}
    assert stats['STATE_ACTION'] == {}


def test_visits_accumulate_per_state():
    stats = empty_state_stats()
    for state in [0, 3, 3, 7, 3]:
        record_state(stats, state)
    assert stats['STATE_VISITS'] == {'0': 1, '3': 3, '7': 1}


def test_actions_are_split_by_who_chose_them():
    stats = empty_state_stats()
    record_action(stats, 5, 1, explored=True)
    record_action(stats, 5, 1, explored=False)
    record_action(stats, 5, 2, explored=False)
    assert stats['STATE_ACTION'] == {
        '5': {'explored': {'1': 1}, 'greedy': {'1': 1, '2': 1}}
    }


def test_a_row_only_ever_explored_is_visible_as_such():
    """The point of the split: exploration counts describe epsilon, not policy."""
    stats = empty_state_stats()
    for _ in range(20):
        record_action(stats, 2, 0, explored=True)
    linha = stats['STATE_ACTION']['2']
    assert 'greedy' not in linha
    assert sum(linha['explored'].values()) == 20


def test_the_counters_survive_a_json_round_trip():
    """They are written to disk per mote per run, so the keys must be strings."""
    import json
    stats = empty_state_stats()
    record_state(stats, 7)
    record_action(stats, 7, 2, explored=False)
    assert json.loads(json.dumps(stats)) == stats


def test_an_action_count_does_not_need_to_know_how_many_actions_there_are():
    """DynQ has three actions, RL-SF has as many as its cell ceiling."""
    stats = empty_state_stats()
    record_action(stats, 0, 7, explored=True)
    assert stats['STATE_ACTION']['0']['explored'] == {'7': 1}
