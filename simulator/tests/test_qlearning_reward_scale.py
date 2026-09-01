"""The four reward terms have to live on one scale.

Throughput and utilisation were shares in [0, 1], the latency term was a packet
count and the energy term a cell count that reached 22. A weight then carries
the units as well as the preference, which is the whole reason the energy weight
had to be 0.01 while the others were near one. These tests pin each term to its
own range so the weights can be compared.
"""
from __future__ import absolute_import

import pytest

from SimEngine.Mote import MoteDefines as d


class StubCell(object):
    def __init__(self, num_tx=0, num_tx_ack=0):
        self.options    = [d.CELLOPTION_TX]
        self.num_tx     = num_tx
        self.num_tx_ack = num_tx_ack


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


def test_the_weights_start_out_equal(agent):
    sf = agent.sf
    assert (
        sf.W_THROUGHPUT == sf.W_UTILIZATION == sf.W_LATENCY == sf.W_ENERGY == 1.0
    )


def test_a_weight_can_be_overridden(sim_engine):
    engine = sim_engine(
        diff_config = {
            'exec_numMotes': 4,
            'sf_class'     : 'Qlearning',
            'W_ENERGY'     : 0.25,
        }
    )
    mote = engine.motes[1]
    mote.sf.start()

    assert mote.sf.W_ENERGY == 0.25
    assert mote.sf.W_THROUGHPUT == 1.0


def test_utilization_is_a_share_and_not_a_yes_or_no(agent):
    # one of three cells has transmitted. Integer division floored this to 0.
    cells = [StubCell(num_tx=5), StubCell(), StubCell()]

    assert agent.sf._reward_utilization(cells) == pytest.approx(1 / 3.0)


def test_utilization_spans_zero_to_one(agent):
    assert agent.sf._reward_utilization([StubCell(), StubCell()]) == 0.0
    assert agent.sf._reward_utilization(
        [StubCell(num_tx=1), StubCell(num_tx=9)]
    ) == 1.0
    assert agent.sf._reward_utilization([]) == 0.0


def test_throughput_is_the_mean_share_of_the_cells_that_transmitted(agent):
    cells = [
        StubCell(num_tx=10, num_tx_ack=10),
        StubCell(num_tx=10, num_tx_ack=6),
        StubCell(),                          # no share to speak of, left out
    ]

    assert agent.sf._reward_throughput(cells) == pytest.approx(0.8)
    assert agent.sf._reward_throughput([StubCell()]) == 0.0


def test_energy_is_the_share_of_the_slotframe_claimed(agent):
    length = agent.sf.settings.tsch_slotframeLength
    cells  = [StubCell() for _ in range(10)]

    assert agent.sf._reward_energy(cells) == pytest.approx(10 / float(length))
    assert agent.sf._reward_energy([]) == 0.0
    assert 0.0 <= agent.sf._reward_energy(cells) <= 1.0


def test_latency_is_the_share_of_the_queue_drained(agent):
    size = agent.sf.settings.tsch_tx_queue_size
    queue = agent.tsch.txQueue

    # the first reading has nothing to compare against
    del queue[:]
    queue.extend([None] * 4)
    assert agent.sf._reward_latency() == 0.0

    # three packets left the queue
    del queue[:]
    queue.extend([None])
    assert agent.sf._reward_latency() == pytest.approx(3 / float(size))

    # two packets arrived
    queue.extend([None, None])
    assert agent.sf._reward_latency() == pytest.approx(-2 / float(size))


def test_every_term_stays_inside_its_range(agent):
    size  = agent.sf.settings.tsch_tx_queue_size
    cells = [StubCell(num_tx=4, num_tx_ack=1), StubCell()]

    assert 0.0 <= agent.sf._reward_throughput(cells)  <= 1.0
    assert 0.0 <= agent.sf._reward_utilization(cells) <= 1.0
    assert 0.0 <= agent.sf._reward_energy(cells)      <= 1.0

    del agent.tsch.txQueue[:]
    agent.tsch.txQueue.extend([None] * size)
    agent.sf._reward_latency()
    del agent.tsch.txQueue[:]
    assert agent.sf._reward_latency() == pytest.approx(1.0)


def test_the_reward_is_the_weighted_sum_of_the_four(agent, monkeypatch):
    sf = agent.sf
    monkeypatch.setattr(agent.rpl, 'getPreferredParent', lambda: None)
    monkeypatch.setattr(agent.tsch, 'get_cells', lambda mac, handle: [])
    monkeypatch.setattr(sf, '_reward_throughput',  lambda cells: 0.5)
    monkeypatch.setattr(sf, '_reward_utilization', lambda cells: 0.25)
    monkeypatch.setattr(sf, '_reward_latency',     lambda: -0.5)
    monkeypatch.setattr(sf, '_reward_energy',      lambda cells: 0.1)

    sf.W_THROUGHPUT, sf.W_UTILIZATION = 1.0, 2.0
    sf.W_LATENCY, sf.W_ENERGY         = 3.0, 4.0

    # 0.5 + 2*0.25 + 3*(-0.5) - 4*0.1
    assert sf.compute_reward() == pytest.approx(-0.9)


def test_the_terms_are_recorded_for_later(agent, monkeypatch):
    monkeypatch.setattr(agent.rpl, 'getPreferredParent', lambda: None)
    monkeypatch.setattr(agent.tsch, 'get_cells', lambda mac, handle: [])

    agent.sf.compute_reward()

    recorded = agent.sf.QLEARNING_STATS['REWARD_TERMS']
    assert len(recorded) == 1
    entry = list(recorded.values())[0]
    assert sorted(entry) == [
        'energy', 'latency', 'reward', 'throughput', 'utilization'
    ]
