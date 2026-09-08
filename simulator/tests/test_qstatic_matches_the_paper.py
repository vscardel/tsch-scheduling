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


# ----------------------------------------------------------- the charge factor

class _Radio(object):
    def __init__(self, idle_listen=0):
        self.stats = {
            'idle_listen': idle_listen, 'tx_data_rx_ack': 0,
            'rx_data_tx_ack': 0, 'tx_data': 0, 'rx_data': 0, 'sleep': 0,
        }


class _Battery(object):
    """A learner reduced to the charge factor and what it reads."""

    INITIAL_REMAINING_BATTERY = QStatic.INITIAL_REMAINING_BATTERY
    TAU_CHARGE = 0.5

    def __init__(self, idle_listen=0):
        self.mote = type('M', (), {'radio': _Radio(idle_listen)})()

    _spent_charge = QStatic.__dict__['_spent_charge']
    _compute_charge = QStatic.__dict__['_compute_charge']
    _compute_average_energy_ratio = staticmethod(lambda valor: valor)
    discretize_energy = QStatic.__dict__['discretize_energy']


def test_a_fresh_mote_has_a_full_battery():
    assert _Battery(idle_listen=0)._compute_charge() == pytest.approx(1.0)


def test_the_battery_falls_as_charge_is_drawn():
    antes = _Battery(idle_listen=1000)._compute_charge()
    depois = _Battery(idle_listen=2000)._compute_charge()
    assert depois < antes < 1.0


def test_the_battery_never_goes_negative():
    """The simulator will happily run a mote past its own battery."""
    esgotado = _Battery(idle_listen=10 ** 12)
    assert esgotado._compute_charge() == 0.0


def test_the_charge_factor_does_not_depend_on_resynchronisation():
    """It used to divide charge since boot by the time since the last sync,
    and tsch resets that on every received frame, so the value inflated and
    grew and the third state bit was pinned to one for the whole run.

    _Battery carries a radio and nothing else: no tsch, no engine. Reaching
    for either would raise here rather than return a number.
    """
    bateria = _Battery(idle_listen=1000)
    assert not hasattr(bateria, 'engine')
    assert not hasattr(bateria.mote, 'tsch')
    assert 0.0 < bateria._compute_charge() < 1.0


def test_the_bit_can_be_zero_and_one():
    """Pinned at one, the state space is four rows out of eight."""
    cheia = _Battery(idle_listen=0)
    vazia = _Battery(idle_listen=10 ** 12)
    assert cheia.discretize_energy(cheia._compute_charge()) == 1
    assert vazia.discretize_energy(vazia._compute_charge()) == 0


def test_the_threshold_is_a_setting_not_a_constant():
    """The manuscript never publishes tau_C, so it must be tunable."""
    import inspect
    fonte = inspect.getsource(QStatic.__init__)
    assert "'QSTATIC_TAU_CHARGE'" in fonte
