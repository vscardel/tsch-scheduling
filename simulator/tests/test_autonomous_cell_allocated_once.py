"""A mote gets one autonomous RX cell, not two.

Both learners called allocate_autonomous_rx_cell unconditionally and then again
in the non-root branch. SlotFrame.add appends without deduplicating, so a
non-root mote carried the same cell twice at the same slot and channel offset.

It matters beyond tidiness. _action_slot notifies the scheduling function for
every cell at the ASN, so each elapsed autonomous slot counted twice towards
RX_CELLS_PASSED, and Q-static decides every MAX_RX_CELLS_PASSED of those: its
RX decision period was half what the configuration asked for. get_autonomous_rx_cell
also asserts there is exactly one.
"""
import inspect

import pytest

from SimEngine.Mote.scheduling_functions.Qlearning import (
    SchedulingFunctionQlearning
)
from SimEngine.Mote.scheduling_functions.QlearningSBRC24 import (
    SchedulingFunctionQlearningSBRC24
)


@pytest.mark.parametrize('classe', [
    SchedulingFunctionQlearning, SchedulingFunctionQlearningSBRC24
])
def test_start_allocates_the_autonomous_rx_cell_exactly_once(classe):
    fonte = inspect.getsource(classe.start)
    assert fonte.count('allocate_autonomous_rx_cell()') == 1


@pytest.mark.parametrize('classe', [
    SchedulingFunctionQlearning, SchedulingFunctionQlearningSBRC24
])
def test_the_root_still_gets_one(classe):
    """It is allocated before the branch, so it is not inside the else."""
    fonte = inspect.getsource(classe.start)
    antes, _, depois = fonte.partition('dagRoot')
    assert 'allocate_autonomous_rx_cell()' in antes
    assert 'allocate_autonomous_rx_cell()' not in depois


@pytest.mark.parametrize('classe', [
    SchedulingFunctionQlearning, SchedulingFunctionQlearningSBRC24
])
def test_only_a_non_root_builds_a_q_table(classe):
    fonte = inspect.getsource(classe.start)
    _, _, depois = fonte.partition('dagRoot')
    assert 'initialize_q_table' in depois
