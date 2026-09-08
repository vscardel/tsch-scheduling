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


# ------------------------------------------------------------- cell removal

class _Cell(object):
    def __init__(self, options, num_tx=0, num_tx_ack=0, num_rx=0, slot=0):
        self.options = options
        self.num_tx = num_tx
        self.num_tx_ack = num_tx_ack
        self.num_rx = num_rx
        self.slot_offset = slot
        self.channel_offset = 0


class _Remover(object):
    SLOTFRAME_HANDLE = 1

    def __init__(self, cells, smart):
        self.cells = cells
        self.SMART_CELL_REMOVAL = smart
        tsch = type('T', (), {'get_cells': lambda _s, mac, handle: cells})()
        rpl = type('R', (), {'getPreferredParent': lambda _s: 'parent'})()
        self.mote = type('M', (), {'tsch': tsch, 'rpl': rpl})()

    _is_unused_cell = QStatic.__dict__['_is_unused_cell']
    _get_cells_to_delete = QStatic.__dict__['_get_cells_to_delete']


TX = ['TX']
RX = ['RX']


def test_a_tx_cell_can_be_offered_for_deletion_at_all():
    """cell_option is a list and was compared against the bare 'TX' string, so
    every call fell through to the RX branch and asked for TX cells that had
    received something. Removing a TX cell was a no-op."""
    ociosa = _Cell(TX, num_tx=10, num_tx_ack=1)
    assert _Remover([ociosa], smart=True)._get_cells_to_delete(TX) != []


def test_the_smart_rule_keeps_the_busy_tx_cells():
    ocupada = _Cell(TX, num_tx=10, num_tx_ack=9, slot=1)
    ociosa = _Cell(TX, num_tx=10, num_tx_ack=1, slot=2)
    saida = _Remover([ocupada, ociosa], smart=True)._get_cells_to_delete(TX)
    assert [c['slotOffset'] for c in saida] == [2]


def test_the_smart_rule_keeps_the_rx_cells_that_received_something():
    usada = _Cell(RX, num_rx=5, slot=1)
    ociosa = _Cell(RX, num_rx=0, slot=2)
    saida = _Remover([usada, ociosa], smart=True)._get_cells_to_delete(RX)
    assert [c['slotOffset'] for c in saida] == [2]


def test_a_tx_cell_that_never_transmitted_is_kept_and_does_not_divide():
    """No ratio to judge it by, and num_tx stays out of the denominator."""
    nova = _Cell(TX, num_tx=0, num_tx_ack=0)
    assert _Remover([nova], smart=True)._get_cells_to_delete(TX) == []


def test_off_by_default_every_occupied_cell_is_a_candidate():
    """The manuscript gives the utilisation rule to DynQ alone, so the
    baseline offers the whole set the way the simulator does."""
    ocupada = _Cell(TX, num_tx=10, num_tx_ack=9, slot=1)
    ociosa = _Cell(TX, num_tx=10, num_tx_ack=1, slot=2)
    saida = _Remover([ocupada, ociosa], smart=False)._get_cells_to_delete(TX)
    assert sorted(c['slotOffset'] for c in saida) == [1, 2]


def test_cells_of_the_other_direction_are_never_offered():
    saida = _Remover([_Cell(RX, num_rx=0)], smart=True)._get_cells_to_delete(TX)
    assert saida == []


def test_the_switch_is_off_unless_a_config_turns_it_on():
    import inspect
    fonte = inspect.getsource(QStatic.__init__)
    assert "'QSTATIC_SMART_CELL_REMOVAL', False" in fonte


# ------------------------------------------------------------------- guards

class _Sixp(object):
    def __init__(self):
        self.pedidos = []

    def send_request(self, **kwargs):
        self.pedidos.append(kwargs)


class _Negotiator(object):
    DEFAULT_CELL_LIST_LEN = 5

    def __init__(self, occupied=None):
        self.mote = type('M', (), {'sixp': _Sixp()})()
        self.retry_count = {'parent': 1}
        self.occupied = occupied if occupied is not None else []

    _create_available_cell_list = lambda self, n: [{'slotOffset': 1}]
    _create_occupied_cell_list = lambda self, **kw: self.occupied
    _create_add_request_callback = lambda self, *a: None
    _create_delete_request_callback = lambda self, *a: None

    sixp_interface_add = QStatic.__dict__['sixp_interface_add']
    _request_deleting_cells = QStatic.__dict__['_request_deleting_cells']


def test_an_action_that_moves_no_cell_sends_no_6p():
    """N_insert is zero in state 000, which the manuscript intends. The
    request still cost a transaction and moved nothing."""
    agente = _Negotiator()
    agente.sixp_interface_add(
        preferred_parent='parent', num_cells=0, cell_option=TX
    )
    assert agente.mote.sixp.pedidos == []


def test_an_action_that_moves_cells_still_sends_6p():
    agente = _Negotiator()
    agente.sixp_interface_add(
        preferred_parent='parent', num_cells=2, cell_option=TX
    )
    assert len(agente.mote.sixp.pedidos) == 1
    assert agente.mote.sixp.pedidos[0]['numCells'] == 2


def test_a_delete_retry_with_nothing_left_gives_up_instead_of_asserting():
    """The 6P timeout retries, and by then the cells can be gone. The same
    assertion killed runs in DynQ and in RL-SF before it was guarded there."""
    agente = _Negotiator(occupied=[])
    agente._request_deleting_cells('parent', 1, TX)
    assert agente.retry_count['parent'] == -1
    assert agente.mote.sixp.pedidos == []


# ------------------------------------------------- surviving a resynchronisation

class _Learner(object):
    num_states = 8

    def __init__(self):
        self.Q_table = {}
        self.EPISODE = 0
        self.EPSLON = None
        self.MIN_EPSLON = 0.1
        self.mote = type('M', (), {
            'tsch': type('T', (), {'delete_slotframe': lambda _s, h: None})()
        })()
        self.SLOTFRAME_HANDLE = 1

    initialize_q_table = QStatic.__dict__['initialize_q_table']
    stop = QStatic.__dict__['stop']


def test_a_resynchronisation_does_not_wipe_what_was_learned():
    """tsch calls start() on every resync, and start() initialises the table."""
    agente = _Learner()
    agente.initialize_q_table(3, 3)
    agente.Q_table[5][1] = 4.2
    agente.initialize_q_table(3, 3)
    assert agente.Q_table[5][1] == 4.2


def test_initialising_still_creates_every_row():
    agente = _Learner()
    agente.initialize_q_table(3, 3)
    assert sorted(agente.Q_table) == list(range(8))
    assert agente.Q_table[0] == [0, 0, 0]


def test_stopping_does_not_send_the_agent_back_to_full_exploration():
    """EPISODE was reset alongside EPSLON, and epsilon is recomputed from the
    episode count on the next decision, so the reset undid the line above it."""
    agente = _Learner()
    agente.EPISODE = 400
    agente.stop()
    assert agente.EPISODE == 400
    assert agente.EPSLON == agente.MIN_EPSLON
