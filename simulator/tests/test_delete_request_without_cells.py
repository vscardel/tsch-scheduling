"""A 6P delete retry with nothing left to delete must not kill the run.

Found at 200 motes: DynQ without its utilisation-aware removal crashed both
attempts on `assert len(cell_list) > 0` inside _request_deleting_cells,
reached from the 6P timeout handler. The timeout retries the request, and by
then the cells can be gone, either because the transaction that timed out was
applied at the far end or because the agent removed them since.

The same assertion, reached the same way, had already been guarded in RL-SF.
"""
import types

import pytest

from SimEngine.Mote.scheduling_functions.Qlearning import (
    SchedulingFunctionQlearning
)


class _FakeSF(object):
    """Only the parts of the scheduling function this path touches."""
    DEFAULT_CELL_LIST_LEN = 5
    MAX_RETRY = 3

    def __init__(self, cells):
        self.cells = cells
        self.retry_count = {'parent': 1}
        self.pedidos = []

    def _create_occupied_cell_list(self, neighbor, cell_options, cell_list_len):
        return self.cells

    def sixp_interface_delete_request(self, *args, **kwargs):
        self.pedidos.append(args)

    _create_delete_request_callback = lambda self, *a: None
    _request_deleting_cells = (
        SchedulingFunctionQlearning.__dict__['_request_deleting_cells']
    )


def test_an_empty_cell_list_gives_up_instead_of_asserting():
    sf = _FakeSF(cells=[])
    sf._request_deleting_cells('parent', 1, ['tx'])
    assert sf.retry_count['parent'] == -1
    assert sf.pedidos == []


def test_giving_up_matches_what_the_retry_limit_does():
    """-1 is the value the MAX_RETRY branch writes, so the rest of the code
    already knows what it means."""
    sf = _FakeSF(cells=[])
    sf._request_deleting_cells('parent', 1, ['tx'])
    assert sf.retry_count['parent'] == -1
