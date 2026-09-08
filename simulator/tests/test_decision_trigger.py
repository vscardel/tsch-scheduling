"""What makes DynQ decide, and that changing it is opt in.

DynQ decides when a cell is added or removed, which is the tight feedback
loop of section 6.4, plus a slotframe floor. Q-static decides every 100 cells
that elapse. Measured over the factorial, that is 136 decisions per mote
against 1048, and the gap tracks the whole latency and 6P difference between
the two. These check the switch that lets the comparison be run, and above
all that the default changes nothing.
"""
import pytest


class _Cell(object):
    def __init__(self, minimal=False):
        self.minimal = minimal


class _Agent(object):
    """Only the parts of DynQ that the counting path touches."""

    def __init__(self, trigger='event', every=100, parent='parent'):
        self.DECISION_TRIGGER = trigger
        self.CELLS_BETWEEN_DECISIONS = every
        self.cells_since_decision = 0
        self.parent = parent
        self.busy = False
        self.decisoes = []

    _is_minimal_cell = staticmethod(lambda cell: cell.minimal)

    class _Rpl(object):
        def __init__(self, agent):
            self.agent = agent

        def getPreferredParent(self):
            return self.agent.parent

    @property
    def mote(self):
        agent = self

        class _Mote(object):
            dagRoot = False
            rpl = _Agent._Rpl(agent)
        return _Mote()

    def _sixp_busy_with(self, neighbor):
        return self.busy

    def adapt_to_traffic(self, cellopt, cell, op):
        self.decisoes.append(op)

    from SimEngine.Mote.scheduling_functions.Qlearning import (
        SchedulingFunctionQlearning as _Q
    )
    _count_cell_towards_next_decision = (
        _Q.__dict__['_count_cell_towards_next_decision']
    )


def elapse(agent, n, minimal=False):
    for _ in range(n):
        agent._count_cell_towards_next_decision(_Cell(minimal))


def test_the_default_never_decides_on_elapsed_cells():
    """Every result collected so far was run this way and must not move."""
    agent = _Agent(trigger='event')
    elapse(agent, 1000)
    assert agent.decisoes == []
    assert agent.cells_since_decision == 0


def test_the_cell_trigger_decides_once_every_n_cells():
    agent = _Agent(trigger='cells', every=100)
    elapse(agent, 99)
    assert agent.decisoes == []
    elapse(agent, 1)
    assert agent.decisoes == ['cells']
    elapse(agent, 100)
    assert len(agent.decisoes) == 2


def test_minimal_cells_do_not_count():
    """They are shared and not negotiated, so they say nothing about demand."""
    agent = _Agent(trigger='cells', every=10)
    elapse(agent, 50, minimal=True)
    assert agent.decisoes == []


def test_a_mote_with_no_parent_does_not_decide():
    agent = _Agent(trigger='cells', every=10, parent=None)
    elapse(agent, 10)
    assert agent.decisoes == []


def test_a_transaction_in_flight_postpones_the_decision():
    """6P allows one per peer, and the cell being negotiated triggers its own."""
    agent = _Agent(trigger='cells', every=10)
    agent.busy = True
    elapse(agent, 10)
    assert agent.decisoes == []


def test_the_counter_restarts_after_a_decision():
    agent = _Agent(trigger='cells', every=10)
    elapse(agent, 10)
    assert agent.cells_since_decision == 0
