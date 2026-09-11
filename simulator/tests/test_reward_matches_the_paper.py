"""The reward is Equation 16 of the paper, term by term.

The revised manuscript keeps the published reward, so the code has to compute
exactly what the equation says: a mean acknowledged share over the cells that
transmitted, the share of cells used at least once, the change in queue length
since the previous decision, and the number of negotiated cells as the energy
cost, with the published weights 0.8, 0.2, 0.8 and 0.01.
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


def with_cells(agent, monkeypatch, cells):
    monkeypatch.setattr(agent.tsch, 'get_cells', lambda *a, **k: cells)


def test_the_weights_are_the_published_ones_by_default(agent):
    assert agent.sf.W_THROUGHPUT  == 0.8
    assert agent.sf.W_UTILIZATION == 0.2
    assert agent.sf.W_LATENCY     == 0.8
    assert agent.sf.W_ENERGY      == 0.01


def test_a_weight_can_be_overridden(sim_engine):
    engine = sim_engine(diff_config={
        'exec_numMotes': 4, 'sf_class': 'Qlearning', 'W_ENERGY': 0.5,
    })
    assert engine.motes[1].sf.W_ENERGY == 0.5


def test_throughput_is_the_mean_share_of_the_cells_that_transmitted(agent, monkeypatch):
    with_cells(agent, monkeypatch, [
        StubCell(num_tx=4, num_tx_ack=2),   # 0.5
        StubCell(num_tx=2, num_tx_ack=2),   # 1.0
        StubCell(num_tx=0, num_tx_ack=0),   # never transmitted: left out
    ])
    agent.sf.compute_reward()
    terms = agent.sf.QLEARNING_STATS['REWARD_TERMS'].values()[-1] \
        if hasattr(dict, 'iteritems') else \
        list(agent.sf.QLEARNING_STATS['REWARD_TERMS'].values())[-1]
    assert terms['throughput'] == pytest.approx(0.75)


def test_utilization_is_the_share_of_cells_used_at_least_once(agent, monkeypatch):
    with_cells(agent, monkeypatch, [
        StubCell(num_tx=7), StubCell(num_tx=1), StubCell(num_tx=0), StubCell(num_tx=0),
    ])
    agent.sf.compute_reward()
    terms = list(agent.sf.QLEARNING_STATS['REWARD_TERMS'].values())[-1]
    assert terms['utilization'] == pytest.approx(0.5)


def test_no_cells_means_zero_throughput_and_utilization(agent, monkeypatch):
    with_cells(agent, monkeypatch, [])
    agent.sf.compute_reward()
    terms = list(agent.sf.QLEARNING_STATS['REWARD_TERMS'].values())[-1]
    assert terms['throughput'] == 0.0
    assert terms['utilization'] == 0.0
    assert terms['energy'] == 0.0


def test_latency_is_how_much_the_queue_drained_since_the_last_decision(agent, monkeypatch):
    with_cells(agent, monkeypatch, [])
    agent.tsch.txQueue[:] = [{}, {}, {}]
    agent.sf.compute_reward()                       # first decision: nothing to compare
    first = list(agent.sf.QLEARNING_STATS['REWARD_TERMS'].values())[-1]
    assert first['latency'] == 0.0
    agent.tsch.txQueue[:] = [{}]
    agent.sf.RECORDED_STEP += 1
    agent.sf.compute_reward()
    second = list(agent.sf.QLEARNING_STATS['REWARD_TERMS'].values())[-1]
    assert second['latency'] == 2.0                 # three packets became one
    agent.tsch.txQueue[:] = [{}, {}, {}, {}]
    agent.sf.RECORDED_STEP += 1
    agent.sf.compute_reward()
    third = list(agent.sf.QLEARNING_STATS['REWARD_TERMS'].values())[-1]
    assert third['latency'] == -3.0                 # and a growing queue is punished


def test_energy_is_the_number_of_negotiated_cells(agent, monkeypatch):
    with_cells(agent, monkeypatch, [StubCell(num_tx=1)] * 5)
    agent.sf.compute_reward()
    terms = list(agent.sf.QLEARNING_STATS['REWARD_TERMS'].values())[-1]
    assert terms['energy'] == 5.0


def test_the_reward_is_the_published_weighted_sum(agent, monkeypatch):
    with_cells(agent, monkeypatch, [
        StubCell(num_tx=2, num_tx_ack=1), StubCell(num_tx=0),
    ])
    agent.tsch.txQueue[:] = []
    reward = agent.sf.compute_reward()
    # throughput 0.5, utilization 0.5, latency 0 (first decision), energy 2
    assert reward == pytest.approx(0.8 * 0.5 + 0.2 * 0.5 + 0.8 * 0.0 - 0.01 * 2)
