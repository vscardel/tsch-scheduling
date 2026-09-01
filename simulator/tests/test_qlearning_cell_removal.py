"""Only cells that are going to waste may be removed from the slotframe.

Section 6.3 of the paper removes TX cells whose acknowledged fraction is below
80% and RX cells that received nothing. The code selected the opposite set: TX
cells at or above 80% and RX cells that had received something, so every removal
took away the cells that were working. It also divided by num_tx without
checking it, which raises ZeroDivisionError on a cell that never transmitted.

On top of that, the TX branch was guarded by "cell_option == d.CELLOPTION_TX"
while every caller passes the list [d.CELLOPTION_TX]. The guard was never true,
so TX removals fell through to the RX rule, asked a TX cell how much it had
received, and got nothing back. These tests pass the list the callers pass.
"""
from __future__ import absolute_import

import pytest

from SimEngine.Mote import MoteDefines as d


class StubCell(object):
    """Enough of a cell for the removal criterion to look at."""

    def __init__(self, options, num_tx=0, num_tx_ack=0, num_rx=0):
        self.options        = options
        self.num_tx         = num_tx
        self.num_tx_ack     = num_tx_ack
        self.num_rx         = num_rx
        self.channel_offset = 1
        self.slot_offset    = 2


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


def candidates(agent, monkeypatch, cells, cell_option):
    monkeypatch.setattr(agent.rpl, 'getPreferredParent', lambda: None)
    monkeypatch.setattr(agent.tsch, 'get_cells', lambda mac, handle: cells)
    return agent.sf._get_unused_cells(cell_option)


def test_tx_cells_below_the_threshold_are_the_candidates(agent, monkeypatch):
    poor    = StubCell([d.CELLOPTION_TX], num_tx=10, num_tx_ack=5)
    healthy = StubCell([d.CELLOPTION_TX], num_tx=10, num_tx_ack=9)

    chosen = candidates(agent, monkeypatch, [poor, healthy], [d.CELLOPTION_TX])

    assert len(chosen) == 1
    assert chosen[0]['num_tx'] == 10
    assert chosen[0]['num_tx_ack'] == 5


def test_a_cell_exactly_at_the_threshold_is_kept(agent, monkeypatch):
    at_threshold = StubCell([d.CELLOPTION_TX], num_tx=10, num_tx_ack=8)

    assert candidates(
        agent, monkeypatch, [at_threshold], [d.CELLOPTION_TX]
    ) == []


def test_a_tx_cell_that_never_transmitted_is_a_candidate(agent, monkeypatch):
    # this used to raise ZeroDivisionError
    never_used = StubCell([d.CELLOPTION_TX], num_tx=0, num_tx_ack=0)

    chosen = candidates(agent, monkeypatch, [never_used], [d.CELLOPTION_TX])

    assert len(chosen) == 1


def test_rx_cells_that_received_nothing_are_the_candidates(agent, monkeypatch):
    silent = StubCell([d.CELLOPTION_RX], num_rx=0)
    busy   = StubCell([d.CELLOPTION_RX], num_rx=4)

    chosen = candidates(agent, monkeypatch, [silent, busy], [d.CELLOPTION_RX])

    assert len(chosen) == 1
    assert chosen[0]['channelOffset'] == silent.channel_offset


def test_cells_of_another_option_are_left_alone(agent, monkeypatch):
    rx_cell = StubCell([d.CELLOPTION_RX], num_rx=0)

    assert candidates(agent, monkeypatch, [rx_cell], [d.CELLOPTION_TX]) == []


def test_the_reported_ack_count_is_the_ack_count(agent, monkeypatch):
    # the dictionary used to report num_tx under both keys
    poor = StubCell([d.CELLOPTION_TX], num_tx=9, num_tx_ack=1)

    chosen = candidates(agent, monkeypatch, [poor], [d.CELLOPTION_TX])

    assert chosen[0]['num_tx'] != chosen[0]['num_tx_ack']
    assert chosen[0]['num_tx_ack'] == 1


def test_the_last_dedicated_cell_is_never_removed(agent, monkeypatch):
    only_cell = StubCell([d.CELLOPTION_TX], num_tx=10, num_tx_ack=1)
    monkeypatch.setattr(agent.rpl, 'getPreferredParent', lambda: None)
    monkeypatch.setattr(agent.tsch, 'get_cells', lambda mac, handle: [only_cell])

    sent = []
    monkeypatch.setattr(agent.sixp, 'send_request', lambda **kw: sent.append(kw))

    agent.sf.sixp_interface_delete(
        num_cells        = 3,
        preferred_parent = None,
        cell_option      = [d.CELLOPTION_TX]
    )

    assert sent == []


def test_no_more_cells_go_than_the_agent_asked_for(agent, monkeypatch):
    cells = [
        StubCell([d.CELLOPTION_TX], num_tx=10, num_tx_ack=1) for _ in range(5)
    ]
    monkeypatch.setattr(agent.rpl, 'getPreferredParent', lambda: None)
    monkeypatch.setattr(agent.tsch, 'get_cells', lambda mac, handle: cells)

    sent = []
    monkeypatch.setattr(agent.sixp, 'send_request', lambda **kw: sent.append(kw))

    agent.sf.sixp_interface_delete(
        num_cells        = 2,
        preferred_parent = None,
        cell_option      = [d.CELLOPTION_TX]
    )

    assert len(sent) == 1
    assert sent[0]['numCells'] == 2
    assert len(sent[0]['cellList']) == 2


def test_the_request_never_takes_the_whole_schedule(agent, monkeypatch):
    # every cell is a candidate and the agent asks for more than there are
    cells = [
        StubCell([d.CELLOPTION_TX], num_tx=10, num_tx_ack=0) for _ in range(4)
    ]
    monkeypatch.setattr(agent.rpl, 'getPreferredParent', lambda: None)
    monkeypatch.setattr(agent.tsch, 'get_cells', lambda mac, handle: cells)

    sent = []
    monkeypatch.setattr(agent.sixp, 'send_request', lambda **kw: sent.append(kw))

    agent.sf.sixp_interface_delete(
        num_cells        = 10,
        preferred_parent = None,
        cell_option      = [d.CELLOPTION_TX]
    )

    assert len(sent[0]['cellList']) == 3
