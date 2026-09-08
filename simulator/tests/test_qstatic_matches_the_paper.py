"""Q-static does what sections/q_static.tex says it does.

The static learner is the paper's baseline and it currently beats DynQ at
every network size, so what it actually computes decides what that result
means. An audit found the code departing from the manuscript's own equations
in three places; these hold it to them.
"""
import pytest

from SimEngine.Mote.scheduling_functions.QlearningSBRC24 import (
    SchedulingFunctionQlearningSBRC24 as QStatic
)


def reward(discrete_state):
    """Equation 11 on a bare instance, with no simulator behind it."""
    return QStatic.__dict__['compute_reward'](QStatic, discrete_state)


# ---------------------------------------------------------------- Equation 11

def test_the_desirable_state_pays_three():
    """S_f is low traffic, low queue, high remaining charge."""
    assert reward((0, 0, 1)) == 3


def test_every_other_state_is_minus_traffic_minus_queue_plus_charge():
    esperado = {
        (0, 0, 0): 0,
        (0, 1, 0): -1,
        (0, 1, 1): 0,
        (1, 0, 0): -1,
        (1, 0, 1): 0,
        (1, 1, 0): -2,
        (1, 1, 1): -1,
    }
    for estado, valor in esperado.items():
        assert reward(estado) == valor, estado


def test_a_full_queue_lowers_the_reward():
    """The bug this replaces read the bits in the wrong order and paid the
    agent for filling its queue."""
    assert reward((0, 1, 1)) < reward((0, 0, 1))
    assert reward((1, 1, 0)) < reward((1, 0, 0))


def test_a_flatter_battery_lowers_the_reward():
    assert reward((1, 1, 0)) < reward((1, 1, 1))
    assert reward((0, 1, 0)) < reward((0, 1, 1))


def test_more_traffic_lowers_the_reward():
    assert reward((1, 0, 0)) < reward((0, 0, 0))
    assert reward((1, 1, 1)) < reward((0, 1, 1))


def test_the_three_bits_are_read_in_the_order_they_are_produced():
    """discretize_variables returns [traffic, queue, charge]. Reading them in
    any other order silently inverts two of the three terms."""
    assert reward((1, 0, 0)) == reward((0, 1, 0))   # traffic and queue are symmetric
    assert reward((0, 0, 1)) != reward((1, 0, 0))   # charge is not


# ------------------------------------------------------- the Bellman update

class _Table(object):
    """A learner reduced to its Q-table and the two constants of Equation 3."""

    ALFA = 0.5
    BETA = 0.9

    def __init__(self):
        self.Q_table = dict((linha, [0.0, 0.0, 0.0]) for linha in range(8))

    return_best_q_value = QStatic.__dict__['return_best_q_value']
    compute_q_table = QStatic.__dict__['compute_q_table']


def test_the_update_lands_on_the_row_the_action_was_taken_in():
    """q_static.tex: "updates the Q(s,a) value for the current state s and
    action a". It used to write into the row of s'."""
    tabela = _Table()
    tabela.compute_q_table(curr_state=3, next_state=5, action=1, reward=2.0)
    assert tabela.Q_table[3][1] == pytest.approx(1.0)
    assert tabela.Q_table[5] == [0.0, 0.0, 0.0]


def test_only_the_action_that_was_taken_moves():
    tabela = _Table()
    tabela.compute_q_table(curr_state=3, next_state=5, action=1, reward=2.0)
    assert tabela.Q_table[3][0] == 0.0
    assert tabela.Q_table[3][2] == 0.0


def test_the_next_row_is_used_as_the_bootstrap():
    tabela = _Table()
    tabela.Q_table[5] = [0.0, 4.0, 0.0]
    tabela.compute_q_table(curr_state=3, next_state=5, action=0, reward=1.0)
    # 0 + 0.5 * (1 + 0.9*4 - 0) = 2.3
    assert tabela.Q_table[3][0] == pytest.approx(2.3)


def test_the_reward_is_supplied_rather_than_re_derived():
    """Deriving it here re-ran the discretisation, which writes to the moving
    average buffers."""
    tabela = _Table()
    tabela.compute_q_table(curr_state=0, next_state=0, action=0, reward=-2.0)
    assert tabela.Q_table[0][0] == pytest.approx(-1.0)


# --------------------------------------------------------------- the row map

def test_the_row_is_the_three_bits_read_as_binary():
    mapear = QStatic.__dict__['map_discrete_state_to_number']
    assert mapear(QStatic, (0, 0, 0)) == 0
    assert mapear(QStatic, (0, 0, 1)) == 1
    assert mapear(QStatic, (1, 0, 0)) == 4
    assert mapear(QStatic, (1, 1, 1)) == 7


def test_the_desirable_state_is_row_one():
    """The code used to spell S_f as the literal 1."""
    mapear = QStatic.__dict__['map_discrete_state_to_number']
    assert mapear(QStatic, QStatic.DESIRABLE_STATE) == 1
