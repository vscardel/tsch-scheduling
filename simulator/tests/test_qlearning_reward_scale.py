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


def zerar_radio(agent):
    for nome in ('idle_listen', 'tx_data_rx_ack', 'rx_data_tx_ack',
                 'tx_data', 'rx_data', 'sleep'):
        agent.radio.stats[nome] = 0


def test_energy_is_the_share_of_the_charge_that_could_have_been_drawn(agent):
    sf = agent.sf
    zerar_radio(agent)
    sf.reward_charge = 0
    sf.reward_charge_asn = sf.engine.getAsn() - 100

    # cem slots dormindo nao gastam nada
    assert sf._reward_energy() == 0.0

    # dez transmissoes com confirmacao, ao longo de cem slots
    sf.reward_charge_asn = sf.engine.getAsn() - 100
    agent.radio.stats['tx_data_rx_ack'] = 10
    assert sf._reward_energy() == pytest.approx(10 / 100.0)


def test_energy_prices_transmitting_above_listening(agent):
    sf = agent.sf

    zerar_radio(agent)
    sf.reward_charge = 0
    sf.reward_charge_asn = sf.engine.getAsn() - 1000
    agent.radio.stats['tx_data_rx_ack'] = 10
    transmitindo = sf._reward_energy()

    zerar_radio(agent)
    sf.reward_charge = 0
    sf.reward_charge_asn = sf.engine.getAsn() - 1000
    agent.radio.stats['idle_listen'] = 10
    escutando = sf._reward_energy()

    # 54.5 contra 6.4. Contar celulas cobrava o mesmo pelos dois.
    assert transmitindo > escutando
    assert transmitindo / escutando == pytest.approx(
        d.CHARGE_TxDataRxAck_uC / d.CHARGE_IdleListen_uC
    )


def test_reading_the_charge_does_not_disturb_the_state_factor(agent):
    # _compute_charge feeds the state and subtracts what it has already read.
    # Sharing that reading with the reward would give each of them a slice of
    # the same consumption.
    sf = agent.sf
    zerar_radio(agent)
    sf.charge = 0
    sf.old_charge = 0
    sf.reward_charge = 0
    sf.reward_charge_asn = sf.engine.getAsn() - 100

    agent.radio.stats['tx_data'] = 4
    sf._reward_energy()

    assert sf._compute_charge() == pytest.approx(4 * d.CHARGE_TxData_uC)


def test_latency_is_how_long_the_packets_have_waited(agent):
    sf = agent.sf
    now = sf.engine.getAsn()
    slotframe = sf.settings.tsch_slotframeLength
    referencia = float(sf.settings.tsch_tx_queue_size * slotframe)
    queue = agent.tsch.txQueue

    del queue[:]
    assert sf._reward_latency() == 1.0            # nada esperando

    queue.append({u'app': {u'timestamp': now}})
    assert sf._reward_latency() == 1.0            # acabou de chegar

    del queue[:]
    queue.append({u'app': {u'timestamp': now - slotframe}})
    assert sf._reward_latency() == pytest.approx(1.0 - slotframe / referencia)


def test_an_older_packet_scores_worse_than_a_fresh_one(agent):
    sf = agent.sf
    now = sf.engine.getAsn()
    slotframe = sf.settings.tsch_slotframeLength
    queue = agent.tsch.txQueue

    del queue[:]
    queue.append({u'app': {u'timestamp': now - slotframe}})
    novo = sf._reward_latency()

    del queue[:]
    queue.append({u'app': {u'timestamp': now - 3 * slotframe}})
    velho = sf._reward_latency()

    assert velho < novo


def test_a_packet_without_a_stamp_is_skipped(agent):
    # control packets go through the queue and carry no stamp
    sf = agent.sf
    now = sf.engine.getAsn()
    slotframe = sf.settings.tsch_slotframeLength
    queue = agent.tsch.txQueue

    del queue[:]
    queue.append({u'type': u'6P'})
    assert sf._reward_latency() == 1.0

    queue.append({u'app': {u'timestamp': now - slotframe}})
    com_controle = sf._reward_latency()

    del queue[:]
    queue.append({u'app': {u'timestamp': now - slotframe}})
    assert sf._reward_latency() == pytest.approx(com_controle)


def test_a_very_old_packet_does_not_push_the_term_below_zero(agent):
    sf = agent.sf
    now = sf.engine.getAsn()
    referencia = (
        sf.settings.tsch_tx_queue_size * sf.settings.tsch_slotframeLength
    )
    queue = agent.tsch.txQueue
    del queue[:]
    # muito mais velho que a referencia, o termo trava em zero e nao vira negativo
    queue.append({u'app': {u'timestamp': now - 10 * referencia}})

    assert sf._reward_latency() == 0.0


def test_every_term_stays_inside_its_range(agent):
    sf = agent.sf
    cells = [StubCell(num_tx=4, num_tx_ack=1), StubCell()]
    queue = agent.tsch.txQueue
    now = sf.engine.getAsn()

    assert 0.0 <= sf._reward_throughput(cells)  <= 1.0
    assert 0.0 <= sf._reward_utilization(cells) <= 1.0

    for idade in (0, sf.settings.tsch_slotframeLength, now):
        del queue[:]
        queue.append({u'app': {u'timestamp': max(0, now - idade)}})
        assert 0.0 <= sf._reward_latency() <= 1.0

    zerar_radio(agent)
    sf.reward_charge = 0
    sf.reward_charge_asn = now - 50
    assert 0.0 <= sf._reward_energy() <= 1.0


def test_the_reward_is_the_weighted_sum_of_the_four(agent, monkeypatch):
    sf = agent.sf
    monkeypatch.setattr(agent.rpl, 'getPreferredParent', lambda: None)
    monkeypatch.setattr(agent.tsch, 'get_cells', lambda mac, handle: [])
    monkeypatch.setattr(sf, '_reward_throughput',  lambda cells: 0.5)
    monkeypatch.setattr(sf, '_reward_utilization', lambda cells: 0.25)
    monkeypatch.setattr(sf, '_reward_latency',     lambda: 0.5)
    monkeypatch.setattr(sf, '_reward_energy',      lambda: 0.1)

    sf.W_THROUGHPUT, sf.W_UTILIZATION = 1.0, 2.0
    sf.W_LATENCY, sf.W_ENERGY         = 3.0, 4.0

    # 0.5 + 2*0.25 + 3*0.5 - 4*0.1
    assert sf.compute_reward() == pytest.approx(2.1)


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
