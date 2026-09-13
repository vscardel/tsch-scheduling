"""How the network was doing, over time, rather than only at the end.

The .kpi files describe a whole run with one number per quantity. That is
enough to say which scheduler ended up better and useless for saying how a
scheduler responded to something. With disturbances in the run, the question
becomes how far delivery fell and how long it took to come back, and neither
can be read from an average taken over the entire run.

So a small time series per run: how many application packets have been sent
and delivered, sampled every few slotframes. Cumulative counters rather than
per-window rates, so the reader can choose its own window afterwards without
the recording having committed to one.

A caveat that belongs with the numbers. Sends and deliveries are counted when
they happen, and a packet is delivered some time after it is sent, so a ratio
taken over a short window is blurred by that delay. The default sampling
period is ten slotframes, about ten seconds, against an end to end latency
around one second, which keeps the blur small; a reader taking a narrower
window than the recording should not.

Counting draws no random numbers and changes no decision, so a run with the
trace scores exactly what it scores without it.
"""
from __future__ import absolute_import


DEFAULT_EVERY_SLOTFRAMES = 10


def empty():
    return {'asn': [], 'sent': [], 'received': [], 'every': None}


def sample(trace, asn, sent, received):
    """Record where the counters stand now."""
    trace['asn'].append(asn)
    trace['sent'].append(sent)
    trace['received'].append(received)


def delivery_series(trace):
    """Delivery ratio per window, from the cumulative counters.

    The first window is dropped: at that point almost nothing has been sent
    and the ratio is dominated by whether one packet happened to arrive.
    """
    saida = []
    for i in range(1, len(trace['asn'])):
        enviados = trace['sent'][i] - trace['sent'][i - 1]
        recebidos = trace['received'][i] - trace['received'][i - 1]
        if enviados <= 0:
            saida.append((trace['asn'][i], None))
        else:
            saida.append((trace['asn'][i], recebidos / float(enviados)))
    return saida


def recovery(serie, asn_disturbance, tolerance=0.95, baseline_windows=10):
    """How long delivery took to come back after a disturbance.

    The level to come back to is the mean of the windows just before the
    disturbance, and coming back means reaching that level times the
    tolerance and staying at or above it for three windows running. Three
    because a single window can cross by chance and calling that a recovery
    would flatter every configuration equally.

    Returns the ASN span, or None if it never came back, which is itself the
    result for an agent that cannot readapt.

    **This is a comparative measure, not a verdict.** A disturbance that
    permanently changes what the network can deliver, more traffic through the
    same schedule being the obvious case, moves the achievable ratio, and then
    no agent returns to the old level however well it adapts. So the number to
    read is one arm's recovery against another's on the same disturbance and
    the same seeds; "did not recover" on its own says as much about the
    disturbance as about the agent.
    """
    antes = [
        valor for asn, valor in serie
        if valor is not None and asn < asn_disturbance
    ][-baseline_windows:]
    if not antes:
        return None
    alvo = (sum(antes) / len(antes)) * tolerance

    depois = [(asn, v) for asn, v in serie if asn >= asn_disturbance]
    seguidas = 0
    for asn, valor in depois:
        if valor is None:
            continue
        if valor >= alvo:
            seguidas += 1
            if seguidas == 3:
                return asn - asn_disturbance
        else:
            seguidas = 0
    return None


def drop(serie, asn_disturbance, baseline_windows=10, after_windows=10):
    """How far delivery fell, as a share of the level before.

    Zero means it did not fall at all, which says the disturbance did not
    reach this configuration rather than that it recovered instantly.
    """
    antes = [
        valor for asn, valor in serie
        if valor is not None and asn < asn_disturbance
    ][-baseline_windows:]
    depois = [
        valor for asn, valor in serie
        if valor is not None and asn >= asn_disturbance
    ][:after_windows]
    if not antes or not depois:
        return None
    nivel = sum(antes) / len(antes)
    if nivel <= 0:
        return None
    return max(0.0, (nivel - min(depois)) / nivel)
