"""A cell must be judged on the chances it has had.

The utilisation term counted a cell as idle from the moment it was negotiated,
before its slot had come round even once. Adding a cell therefore lowered the
reward immediately, while the gain that cell brings arrives later and is
discounted by BETA.

Measured over 7976 randomly drawn actions, this was 90% of the reward gap
between inserting a cell and doing nothing, and motes that happened to insert
more ended with lower reward and lower latency. The reward was pointing away
from the network it was meant to describe.
"""
from __future__ import absolute_import

import pytest

from SimEngine.Mote import MoteDefines as d


class StubCell(object):
    def __init__(self, num_tx=0, created_asn=0):
        self.options     = [d.CELLOPTION_TX]
        self.num_tx      = num_tx
        self.num_tx_ack  = num_tx
        self.num_rx      = 0
        self.created_asn = created_asn


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


def slotframe(agent):
    return agent.sf.settings.tsch_slotframeLength


def test_a_cell_that_has_not_had_its_turn_is_not_judged(agent):
    now = agent.sf.engine.getAsn()
    fresh = StubCell(num_tx=0, created_asn=now)

    assert agent.sf._cell_had_its_turn(fresh) is False


def test_a_cell_is_judged_once_its_slot_came_round(agent):
    now = agent.sf.engine.getAsn()
    old = StubCell(num_tx=0, created_asn=now - slotframe(agent))

    assert agent.sf._cell_had_its_turn(old) is True


def test_a_cell_without_a_recorded_age_is_judged(agent):
    unknown = StubCell(num_tx=0)
    unknown.created_asn = None

    assert agent.sf._cell_had_its_turn(unknown) is True


def test_adding_a_cell_no_longer_lowers_utilization(agent):
    now = agent.sf.engine.getAsn()
    established = [
        StubCell(num_tx=5, created_asn=now - 10 * slotframe(agent))
        for _ in range(3)
    ]

    before = agent.sf._reward_utilization(established)
    after = agent.sf._reward_utilization(
        established + [StubCell(num_tx=0, created_asn=now)]
    )

    assert before == 1.0
    assert after == before


def test_a_cell_that_had_its_turn_and_stayed_idle_still_counts(agent):
    now = agent.sf.engine.getAsn()
    cells = [
        StubCell(num_tx=5, created_asn=now - 10 * slotframe(agent)),
        StubCell(num_tx=0, created_asn=now - 10 * slotframe(agent)),
    ]

    assert agent.sf._reward_utilization(cells) == pytest.approx(0.5)


def test_utilization_is_zero_when_no_cell_has_had_a_turn(agent):
    now = agent.sf.engine.getAsn()
    fresh = [StubCell(num_tx=0, created_asn=now) for _ in range(2)]

    assert agent.sf._reward_utilization(fresh) == 0.0
    assert agent.sf._reward_utilization([]) == 0.0


def test_a_real_cell_records_when_it_entered_the_schedule(agent):
    slotframe_handle = agent.sf.SLOTFRAME_HANDLE
    agent.tsch.addCell(
        slotOffset       = 40,
        channelOffset    = 3,
        neighbor         = None,
        cellOptions      = [d.CELLOPTION_TX],
        slotframe_handle = slotframe_handle,
    )

    cell = agent.tsch.get_cell(40, 3, None, slotframe_handle)

    assert cell.created_asn == agent.sf.engine.getAsn()
    assert agent.sf._cell_had_its_turn(cell) is False
