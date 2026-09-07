"""Running one cell of the factorial, and running it twice without collision.

A control arm needs the same state model and the same code path as the cell it
is being compared against, differing only in the hyperparameters. That means
running one cell rather than all nine, and writing its results somewhere the
cell itself will not be overwritten by.
"""
import pytest

from runExperiments import cell_name


def test_a_cell_is_named_after_the_factors_it_switched_on():
    assert cell_name(['traffic', 'queue', 'charge']) == 'traffic_queue_charge'
    assert cell_name(['queue']) == 'queue'


def test_the_empty_cell_is_the_baseline():
    """No factors means no learner: that cell runs MSF."""
    assert cell_name([]) == 'baseline'


def test_the_static_learner_keeps_its_own_name():
    assert cell_name(['qlearningSBRC24']) == 'qlearningSBRC24'


def test_every_cell_of_the_factorial_has_a_distinct_name():
    import itertools
    factors = ['traffic', 'queue', 'charge']
    cells = []
    for combination in itertools.product([0, 1], repeat=3):
        cells.append(cell_name(
            [factors[i] for i, on in enumerate(combination) if on]
        ))
    cells.append(cell_name(['qlearningSBRC24']))
    assert len(set(cells)) == len(cells) == 9
