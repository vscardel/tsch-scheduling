"""Changes to the environment, and the guarantee that they cost nothing when
nobody asks for them.

Every experiment in this project ran in a network that never changed, so
nothing rewarded an agent for readapting and no metric could tell an agent
that tracks a moving environment from one that found a decent operating point
and sat on it. That is the gap these fill, and the reviewers' objection is
precisely about non-stationarity.

The property protected hardest here is the negative one: with no disturbance
declared, the run is exactly the run it was before. Three times in this
revision a change that looked inert turned out not to be.
"""
from __future__ import absolute_import

import random

import pytest

from SimEngine import disturbances


class _Settings(object):
    tsch_slotframeLength = 101
    exec_numSlotframesPerRun = 100
    app_pkPeriod = 60
    phy_numChans = 16
    disturbances = None


class _Mote(object):
    def __init__(self, mote_id):
        self.id = mote_id


class _Matrix(object):
    """Every pair connected on every channel, and a record of what changed."""

    def __init__(self, ids, canais):
        self.pdr = {}
        for src in ids:
            for dst in ids:
                for canal in canais:
                    self.pdr[(src, dst, canal)] = 1.0
        self.mudados = []

    def get_pdr(self, src, dst, canal):
        return self.pdr[(src, dst, canal)]

    def set_pdr_both_directions(self, a, b, canal, valor):
        self.pdr[(a, b, canal)] = valor
        self.pdr[(b, a, canal)] = valor
        self.mudados.append((a, b, canal, valor))


class _Connectivity(object):
    def __init__(self, matrix):
        self.matrix = matrix


class _Engine(object):
    """The engine reduced to the calendar and the two things a disturbance
    reaches into."""

    def __init__(self, num_motes=6, declared=None, seed=1):
        self.settings = _Settings()
        self.settings.disturbances = declared
        self.motes = [_Mote(i) for i in range(num_motes)]
        self.random_seed = seed
        self.asn = 0
        self.agendado = []
        canais = disturbances._channels(self.settings)
        self.connectivity = _Connectivity(
            _Matrix([m.id for m in self.motes], canais)
        )

    def scheduleAtAsn(self, asn, cb, uniqueTag, intraSlotOrder):
        assert asn > self.asn
        self.agendado.append((asn, cb, uniqueTag))

    def fire(self):
        for _, cb, _ in sorted(self.agendado, key=lambda e: e[0]):
            cb()


# ------------------------------------------------- nothing asked, nothing done

def test_no_disturbances_declared_schedules_nothing():
    """The run has to be exactly the run it was before this existed."""
    engine = _Engine(declared=None)
    assert disturbances.schedule(engine) == []
    assert engine.agendado == []


def test_an_empty_list_schedules_nothing():
    engine = _Engine(declared=[])
    assert disturbances.schedule(engine) == []
    assert engine.agendado == []


# --------------------------------------------------------------- when it fires

def test_the_fraction_becomes_an_asn_in_the_run():
    engine = _Engine(declared=[
        {'asn_fraction': 0.4, 'kind': 'traffic', 'factor': 0.5}
    ])
    agendadas = disturbances.schedule(engine)
    fim = 101 * 100
    assert agendadas[0][0] == int(fim * 0.4)
    assert 0 < agendadas[0][0] < fim


def test_the_same_fraction_lands_at_the_same_place_at_any_length():
    """One configuration has to describe the same experiment whatever length
    it is run at, which is why fractions and not absolute ASNs."""
    curto = _Settings()
    longo = _Settings()
    longo.exec_numSlotframesPerRun = 400
    fracao = 0.7
    assert (
        disturbances.asn_of(curto, fracao) / float(disturbances.run_length(curto))
        == pytest.approx(
            disturbances.asn_of(longo, fracao) / float(disturbances.run_length(longo)),
            abs=1e-4
        )
    )


def test_a_disturbance_outside_the_run_is_dropped():
    """At zero there is nothing yet to disturb, and at one it never fires."""
    engine = _Engine(declared=[
        {'asn_fraction': 0.0, 'kind': 'traffic', 'factor': 0.5},
        {'asn_fraction': 1.0, 'kind': 'traffic', 'factor': 0.5},
        {'asn_fraction': 1.5, 'kind': 'traffic', 'factor': 0.5},
    ])
    assert disturbances.schedule(engine) == []


def test_several_disturbances_come_back_in_order():
    engine = _Engine(declared=[
        {'asn_fraction': 0.7, 'kind': 'traffic', 'factor': 2.0},
        {'asn_fraction': 0.4, 'kind': 'traffic', 'factor': 0.5},
    ])
    agendadas = disturbances.schedule(engine)
    assert [a for a, _ in agendadas] == sorted(a for a, _ in agendadas)


# ------------------------------------------------------------------ traffic

def test_traffic_multiplies_the_sending_period():
    """A factor below one is a surge: the period shrinks, so packets come
    more often."""
    engine = _Engine(declared=[
        {'asn_fraction': 0.4, 'kind': 'traffic', 'factor': 0.5}
    ])
    disturbances.schedule(engine)
    assert engine.settings.app_pkPeriod == 60
    engine.fire()
    assert engine.settings.app_pkPeriod == 30


def test_traffic_touches_nothing_but_the_period():
    """app._schedule_transmission reads the period afresh for every packet,
    so this needs to change no calendar entry of its own."""
    engine = _Engine(declared=[
        {'asn_fraction': 0.4, 'kind': 'traffic', 'factor': 2.0}
    ])
    disturbances.schedule(engine)
    antes = len(engine.agendado)
    engine.fire()
    assert len(engine.agendado) == antes
    assert engine.connectivity.matrix.mudados == []


# ------------------------------------------------------------ link quality

def test_link_quality_degrades_the_share_asked_for():
    engine = _Engine(num_motes=6, declared=[
        {'asn_fraction': 0.4, 'kind': 'link_quality', 'share': 0.5, 'pdr': 0.4}
    ])
    disturbances.schedule(engine)
    engine.fire()

    canais = disturbances._channels(engine.settings)
    pares = set(
        (a, b) for a, b, _, _ in engine.connectivity.matrix.mudados
    )
    # six motes give fifteen unordered pairs, half of them is eight
    assert len(pares) == 8
    assert len(engine.connectivity.matrix.mudados) == 8 * len(canais)


def test_link_quality_degrades_both_directions_on_every_channel():
    engine = _Engine(num_motes=4, declared=[
        {'asn_fraction': 0.4, 'kind': 'link_quality', 'share': 1.0, 'pdr': 0.2}
    ])
    disturbances.schedule(engine)
    engine.fire()

    matriz = engine.connectivity.matrix
    for canal in disturbances._channels(engine.settings):
        for src in range(4):
            for dst in range(4):
                if src != dst:
                    assert matriz.get_pdr(src, dst, canal) == 0.2


def test_a_dead_link_is_not_counted_against_the_share():
    """Degrading a link that was already dead changes nothing and would eat
    into the share that was asked for."""
    engine = _Engine(num_motes=4, declared=[
        {'asn_fraction': 0.4, 'kind': 'link_quality', 'share': 1.0, 'pdr': 0.2}
    ])
    for canal in disturbances._channels(engine.settings):
        engine.connectivity.matrix.pdr[(0, 1, canal)] = 0.0
        engine.connectivity.matrix.pdr[(1, 0, canal)] = 0.0

    disturbances.schedule(engine)
    engine.fire()
    pares = set((a, b) for a, b, _, _ in engine.connectivity.matrix.mudados)
    assert (0, 1) not in pares
    assert len(pares) == 5          # six pairs less the dead one


# ------------------------------------------------------ the random stream

def test_disturbing_draws_nothing_from_the_simulation_stream():
    """A perturbed arm and an unperturbed one have to see the same topology
    and the same draws everywhere else, or the comparison between them is
    meaningless by construction."""
    random.seed(999)
    esperado = [random.random() for _ in range(5)]

    random.seed(999)
    engine = _Engine(num_motes=6, declared=[
        {'asn_fraction': 0.4, 'kind': 'traffic', 'factor': 0.5},
        {'asn_fraction': 0.7, 'kind': 'link_quality', 'share': 0.5, 'pdr': 0.4},
    ])
    disturbances.schedule(engine)
    engine.fire()
    obtido = [random.random() for _ in range(5)]
    assert obtido == esperado


def test_the_same_seed_degrades_the_same_links():
    def rodar(seed):
        engine = _Engine(num_motes=8, seed=seed, declared=[
            {'asn_fraction': 0.4, 'kind': 'link_quality',
             'share': 0.3, 'pdr': 0.4}
        ])
        disturbances.schedule(engine)
        engine.fire()
        return set((a, b) for a, b, _, _ in engine.connectivity.matrix.mudados)

    assert rodar(7) == rodar(7)
    assert rodar(7) != rodar(8)
