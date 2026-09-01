"""The traffic and charge factors must report what changed, not the history.

Both are built from radio counters that only ever grow. Each one used to store
the difference it had just computed instead of the counter it read, so the next
difference was taken against the wrong baseline and the factor drifted upwards
for the whole run. A factor that only grows is above its own moving average
almost always, which pins its discretised bit at 1 and takes the state variable
out of the agent's state.
"""
from __future__ import absolute_import

import pytest

from SimEngine.Mote import MoteDefines as d

RADIO_COUNTERS = [
    'idle_listen',
    'tx_data_rx_ack',
    'rx_data_tx_ack',
    'tx_data',
    'rx_data',
    'sleep',
]


@pytest.fixture
def agent(sim_engine):
    """A mote running the Q-learning SF, with its radio counters zeroed."""
    engine = sim_engine(
        diff_config = {
            'exec_numMotes': 4,
            'sf_class'     : 'Qlearning',
        }
    )
    mote = engine.motes[1]
    mote.sf.start()
    for name in RADIO_COUNTERS:
        mote.radio.stats[name] = 0
    mote.sf.prev_rx_ack = 0
    mote.sf.charge = 0
    mote.sf.old_charge = 0
    return mote


def test_traffic_is_the_change_in_the_counter(agent):
    counter  = [0, 3, 3, 10, 11]
    expected = [0, 3, 0,  7,  1]

    measured = []
    for value in counter:
        agent.radio.stats['rx_data_tx_ack'] = value
        measured.append(agent.sf._compute_traffic())

    assert measured == expected


def test_a_steady_arrival_rate_gives_a_steady_traffic(agent):
    # five packets acknowledged per call, twenty calls. Under the old code this
    # climbed with every call instead of staying at five.
    measured = []
    for step in range(1, 21):
        agent.radio.stats['rx_data_tx_ack'] = 5 * step
        measured.append(agent.sf._compute_traffic())

    assert measured == [5] * 20


def test_charge_is_what_was_spent_since_the_last_call(agent):
    agent.radio.stats['tx_data'] = 1
    first = agent.sf._compute_charge()
    assert first == d.CHARGE_TxData_uC

    # nothing happened in between, so nothing was spent
    assert agent.sf._compute_charge() == 0

    agent.radio.stats['rx_data'] = 2
    assert agent.sf._compute_charge() == 2 * d.CHARGE_RxData_uC


def test_charge_does_not_re_add_the_counters(agent):
    # the radio does not move at all, so the total charge has to stay put
    agent.radio.stats['idle_listen'] = 100
    agent.sf._compute_charge()
    total = agent.sf.charge

    for _ in range(10):
        agent.sf._compute_charge()

    assert agent.sf.charge == total


def test_the_total_charge_is_the_sum_over_the_counters(agent):
    agent.radio.stats['idle_listen']    = 3
    agent.radio.stats['tx_data_rx_ack'] = 5
    agent.radio.stats['rx_data_tx_ack'] = 7
    agent.radio.stats['tx_data']        = 11
    agent.radio.stats['rx_data']        = 13
    agent.radio.stats['sleep']          = 17

    expected = (
        3  * d.CHARGE_IdleListen_uC +
        5  * d.CHARGE_TxDataRxAck_uC +
        7  * d.CHARGE_RxDataTxAck_uC +
        11 * d.CHARGE_TxData_uC +
        13 * d.CHARGE_RxData_uC +
        17 * d.CHARGE_Sleep_uC
    )

    assert agent.sf._compute_charge() == expected
