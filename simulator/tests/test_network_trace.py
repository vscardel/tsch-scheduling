"""How the network was doing over time, and what recovery means.

The .kpi files give one number per quantity for a whole run, which cannot say
how far delivery fell after a disturbance or how long it took to come back.
Those two are the metrics the non-stationary framing turns on, so they get a
time series of their own.
"""
from __future__ import absolute_import

import pytest

from SimEngine import network_trace as nt


def trace_de(pares, every=10):
    """A trace from (sent, received) cumulative pairs, ten slotframes apart."""
    t = nt.empty()
    t['every'] = every
    for i, (enviados, recebidos) in enumerate(pares):
        nt.sample(t, i * 101 * every, enviados, recebidos)
    return t


# ------------------------------------------------------------- the series

def test_a_fresh_trace_holds_nothing():
    t = nt.empty()
    assert t['asn'] == [] and t['sent'] == [] and t['received'] == []


def test_the_ratio_comes_from_differencing_the_counters():
    """Cumulative counters are recorded so the reader picks its own window."""
    t = trace_de([(0, 0), (100, 70), (200, 130)])
    serie = nt.delivery_series(t)
    assert [round(v, 4) for _, v in serie] == [0.7, 0.6]


def test_the_first_window_is_dropped():
    """Almost nothing has been sent yet, so the ratio there is decided by
    whether a single packet happened to arrive."""
    t = trace_de([(0, 0), (10, 10)])
    assert len(nt.delivery_series(t)) == 1


def test_a_window_that_sent_nothing_has_no_ratio():
    t = trace_de([(0, 0), (50, 40), (50, 40), (100, 80)])
    serie = nt.delivery_series(t)
    assert serie[1][1] is None
    assert serie[0][1] == pytest.approx(0.8)


# ---------------------------------------------------------- how far it fell

def test_the_drop_is_measured_against_the_level_before():
    antes = [(100 * i, 80 * i) for i in range(1, 12)]      # 80%
    depois = [(1100 + 100 * i, 880 + 40 * i) for i in range(1, 6)]  # 40%
    t = trace_de([(0, 0)] + antes + depois)
    serie = nt.delivery_series(t)
    asn = t['asn'][12]
    assert nt.drop(serie, asn) == pytest.approx(0.5, abs=0.01)


def test_no_fall_reads_as_zero_and_not_as_instant_recovery():
    """It says the disturbance never reached this configuration."""
    t = trace_de([(100 * i, 80 * i) for i in range(20)])
    serie = nt.delivery_series(t)
    assert nt.drop(serie, t['asn'][10]) == 0.0


# ------------------------------------------------------ how long to come back

def test_recovery_is_the_span_until_delivery_holds_again():
    antes = [(100 * i, 80 * i) for i in range(1, 12)]
    caido = [(1100 + 100, 880 + 40)]
    voltou = [(1200 + 100 * i, 920 + 80 * i) for i in range(1, 5)]
    t = trace_de([(0, 0)] + antes + caido + voltou)
    serie = nt.delivery_series(t)
    asn = t['asn'][12]
    span = nt.recovery(serie, asn)
    assert span is not None
    assert span > 0


def test_one_lucky_window_is_not_a_recovery():
    """A single window can cross the line by chance, and calling that a
    recovery would flatter every configuration equally."""
    antes = [(100 * i, 80 * i) for i in range(1, 12)]
    depois = [
        (1200, 920),    # 40%, caiu
        (1300, 1000),   # 80%, cruzou uma vez
        (1400, 1040),   # 40%, caiu de novo
        (1500, 1080),   # 40%
    ]
    t = trace_de([(0, 0)] + antes + depois)
    serie = nt.delivery_series(t)
    assert nt.recovery(serie, t['asn'][12]) is None


def test_never_coming_back_is_the_answer_and_not_an_error():
    """For an agent that cannot readapt, that is the result."""
    antes = [(100 * i, 80 * i) for i in range(1, 12)]
    depois = [(1100 + 100 * i, 880 + 20 * i) for i in range(1, 8)]
    t = trace_de([(0, 0)] + antes + depois)
    serie = nt.delivery_series(t)
    assert nt.recovery(serie, t['asn'][12]) is None


def test_a_disturbance_before_any_data_has_no_level_to_return_to():
    t = trace_de([(0, 0), (100, 80)])
    serie = nt.delivery_series(t)
    assert nt.recovery(serie, 0) is None
    assert nt.drop(serie, 0) is None
