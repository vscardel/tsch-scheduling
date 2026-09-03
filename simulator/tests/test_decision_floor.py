"""A mote must not be able to silence itself.

A decision is otherwise only triggered by a cell being added or removed, which
is the output of the policy. Choosing to idle, or choosing the action that moves
no cell, changes nothing, fires no event, and the mote never decides again.
Measured, 63% of motes ended on an action with no effect against 49% expected,
p=4e-06, and ten times the run length bought only three times the episodes.

The event stays the main trigger. This is a floor under it.
"""
from __future__ import absolute_import

import pytest

from SimEngine.Mote import MoteDefines as d


@pytest.fixture
def agent(sim_engine):
    engine = sim_engine(
        diff_config = {
            'exec_numMotes': 4,
            'sf_class'     : 'Qlearning',
        }
    )
    mote = engine.motes[1]
    mote.sf.start()
    return mote


def armar(agent, monkeypatch):
    """Count decisions and give the mote a parent to talk to."""
    decisoes = []
    monkeypatch.setattr(agent.sf, 'adapt_to_traffic',
                        lambda opt, cell, op: decisoes.append(op))
    monkeypatch.setattr(agent.rpl, 'getPreferredParent', lambda: 'parent')
    return decisoes


def test_nothing_happens_before_the_floor_is_reached(agent, monkeypatch):
    decisoes = armar(agent, monkeypatch)
    agent.sf.slotframes_since_decision = 0

    for _ in range(agent.sf.MAX_SLOTFRAMES_BETWEEN_DECISIONS - 1):
        agent.sf.indication_slotframe_window_ending(1)

    assert decisoes == []


def test_the_floor_forces_a_decision(agent, monkeypatch):
    decisoes = armar(agent, monkeypatch)
    agent.sf.slotframes_since_decision = 0

    for _ in range(agent.sf.MAX_SLOTFRAMES_BETWEEN_DECISIONS):
        agent.sf.indication_slotframe_window_ending(1)

    assert decisoes == ['timer']


def test_a_decision_resets_the_count(agent, monkeypatch):
    monkeypatch.setattr(agent.rpl, 'getPreferredParent', lambda: 'parent')
    monkeypatch.setattr(agent, 'clear_to_send_EBs_DATA', lambda: True)
    monkeypatch.setattr(agent.sf, 'sixp_interface_add', lambda **kw: None)
    monkeypatch.setattr(agent.sf, 'sixp_interface_delete', lambda **kw: None)

    agent.sf.slotframes_since_decision = 999
    agent.sf.adapt_to_traffic([d.CELLOPTION_TX], None, 'insertion')

    assert agent.sf.slotframes_since_decision == 0


def test_the_root_never_decides_on_the_timer(agent, monkeypatch):
    raiz = agent.engine.motes[0]   # ja iniciada no boot
    decisoes = []
    monkeypatch.setattr(raiz.sf, 'adapt_to_traffic',
                        lambda opt, cell, op: decisoes.append(op))

    raiz.sf.slotframes_since_decision = 0
    for _ in range(raiz.sf.MAX_SLOTFRAMES_BETWEEN_DECISIONS + 5):
        raiz.sf.indication_slotframe_window_ending(1)

    assert decisoes == []


def test_a_mote_without_a_parent_waits(agent, monkeypatch):
    decisoes = armar(agent, monkeypatch)
    monkeypatch.setattr(agent.rpl, 'getPreferredParent', lambda: None)
    agent.sf.slotframes_since_decision = 0

    for _ in range(agent.sf.MAX_SLOTFRAMES_BETWEEN_DECISIONS + 5):
        agent.sf.indication_slotframe_window_ending(1)

    assert decisoes == []


def test_the_timer_yields_to_a_transaction_already_in_flight(agent, monkeypatch):
    # 6P allows one transaction per pair, and the cell being negotiated will
    # trigger a decision by itself when it lands
    decisoes = armar(agent, monkeypatch)
    monkeypatch.setattr(agent.sf, '_sixp_busy_with', lambda vizinho: True)
    agent.sf.slotframes_since_decision = 0

    for _ in range(agent.sf.MAX_SLOTFRAMES_BETWEEN_DECISIONS + 5):
        agent.sf.indication_slotframe_window_ending(1)

    assert decisoes == []


def test_a_wider_window_advances_the_count_by_its_width(agent, monkeypatch):
    decisoes = armar(agent, monkeypatch)
    agent.sf.slotframes_since_decision = 0

    agent.sf.indication_slotframe_window_ending(
        agent.sf.MAX_SLOTFRAMES_BETWEEN_DECISIONS
    )

    assert decisoes == ['timer']


def test_the_floor_can_be_set_from_the_configuration(sim_engine):
    engine = sim_engine(
        diff_config = {
            'exec_numMotes'                    : 4,
            'sf_class'                         : 'Qlearning',
            'MAX_SLOTFRAMES_BETWEEN_DECISIONS' : 7,
        }
    )
    mote = engine.motes[1]
    mote.sf.start()

    assert mote.sf.MAX_SLOTFRAMES_BETWEEN_DECISIONS == 7
