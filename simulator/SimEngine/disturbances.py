"""Changes to the environment, part way through a run.

Every experiment in this project so far ran in a network that never changed:
the traffic rate was fixed, the links were fixed, and nothing happened between
the first slotframe and the last. That makes a whole class of question
unanswerable. Nothing rewards an agent for readapting, so the hyperparameter
search had no way to prefer parameters that keep the agent able to readapt,
and no metric could tell an agent that tracks a moving environment from one
that simply found a decent operating point and sat on it.

That matters here more than it would elsewhere, because the reviewers'
objection is precisely about non-stationarity, and because the textbook answer
to it, Sutton and Barto's "Tracking a Nonstationary Problem", only means
anything if there is something to track.

Two kinds, both from the same idea: knock the network and watch what the agent
does about it.

  traffic       every mote starts sending more often, or less
  link_quality  a share of the links degrades, the way an interferer would

Positions are given as a fraction of the run rather than an absolute ASN, so
one configuration describes the same experiment whatever length it is run at.

Nothing here draws from the simulation's random stream. It keeps its own
generator, seeded from the run's seed, so a perturbed arm and an unperturbed
one see the same topology and the same sequence of random draws everywhere
else. Without that the two arms would not be comparable, and comparing them is
the entire point.

With no disturbances declared, nothing is scheduled and the run is exactly the
run it was before this module existed.
"""
from __future__ import absolute_import

import random

from . import Mote
from .Mote import MoteDefines as d


# an arbitrary prime, so this generator does not walk in step with the one the
# simulation draws from even though both are seeded from the same number
SEED_OFFSET = 104729


def run_length(settings):
    """The ASN the run ends at."""
    return (
        settings.tsch_slotframeLength * settings.exec_numSlotframesPerRun
    )


def asn_of(settings, fraction):
    """Where in the run a fraction falls."""
    return int(run_length(settings) * fraction)


def schedule(engine):
    """Put every declared disturbance on the engine's calendar.

    Returns what was scheduled, in ASN order, which is what the tests read and
    what a caller can log.
    """
    declared = getattr(engine.settings, 'disturbances', None)
    if not declared:
        return []

    fim = run_length(engine.settings)
    rng = random.Random((engine.random_seed or 0) + SEED_OFFSET)

    agendadas = []
    for indice, spec in enumerate(declared):
        asn = asn_of(engine.settings, spec['asn_fraction'])
        if not 0 < asn < fim:
            # a disturbance at the very start has nothing to disturb, and one
            # at or past the end never fires
            continue
        engine.scheduleAtAsn(
            asn            = asn,
            cb             = _callback(engine, spec, rng),
            uniqueTag      = (u'Disturbance', u'{0}'.format(indice)),
            intraSlotOrder = d.INTRASLOTORDER_ADMINTASKS,
        )
        agendadas.append((asn, spec))
    return sorted(agendadas)


def _callback(engine, spec, rng):
    def aplicar():
        APPLY[spec['kind']](engine, spec, rng)
    return aplicar


def _apply_traffic(engine, spec, rng):
    """Every mote's sending period is multiplied, so a factor below one is a
    surge and above one is a lull.

    app._schedule_transmission reads settings.app_pkPeriod afresh for every
    packet, so this takes effect from each mote's next transmission without
    touching anything already on the calendar.
    """
    engine.settings.app_pkPeriod = (
        engine.settings.app_pkPeriod * spec['factor']
    )


def _apply_link_quality(engine, spec, rng):
    """A share of the connected links drops to a worse delivery ratio.

    Both directions and every channel, which is what an interferer sitting
    between two motes looks like from the outside. Links that were already
    dead are left alone: degrading them would change nothing and would eat
    into the share that was asked for.
    """
    vivos = _live_links(engine)
    if not vivos:
        return
    quantos = int(round(len(vivos) * spec['share']))
    for par in rng.sample(vivos, min(quantos, len(vivos))):
        for channel in _channels(engine.settings):
            engine.connectivity.matrix.set_pdr_both_directions(
                par[0], par[1], channel, spec['pdr']
            )


def _live_links(engine):
    """Every unordered pair of motes with any connectivity between them."""
    ids = sorted(mote.id for mote in engine.motes)
    canais = _channels(engine.settings)
    saida = []
    for posicao, src in enumerate(ids):
        for dst in ids[posicao + 1:]:
            if any(
                engine.connectivity.matrix.get_pdr(src, dst, canal) > 0
                for canal in canais
            ):
                saida.append((src, dst))
    return saida


def _channels(settings):
    return d.TSCH_HOPPING_SEQUENCE[:settings.phy_numChans]


APPLY = {
    'traffic'     : _apply_traffic,
    'link_quality': _apply_link_quality,
}
