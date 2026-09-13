"""The 2^k factorial measures the state factors, and nothing else.

Two things contaminated the published main effects, and neither is visible in
the numbers they produce. The empty cell of the design ran MSF, so each effect
compared a state factor against another scheduling function: the latency
effect of the traffic factor read -0.805 s with that cell in the arithmetic
and -0.005 s without it. And each cell ran the hyperparameters its own
optimisation found, so an effect also mixed in a different learning rate.
"""
import pytest

from factorial_per_metric import (
    EMPTY_CELL, EMPTY_CELL_MSF, FACTORS, cell_name, cells
)


def test_the_empty_cell_is_the_stateless_learner_by_default():
    assert EMPTY_CELL == 'sem_estado'
    assert cell_name([]) == EMPTY_CELL
    assert EMPTY_CELL in [n for n, _ in cells()]


def test_the_msf_cell_has_to_be_asked_for_by_name():
    """It reproduces the published arithmetic, so it stays reachable."""
    nomes = [n for n, _ in cells(EMPTY_CELL_MSF)]
    assert EMPTY_CELL_MSF in nomes
    assert EMPTY_CELL not in nomes


def test_the_design_has_one_cell_per_combination():
    assert len(cells()) == 2 ** len(FACTORS)
    assert len(set(n for n, _ in cells())) == 2 ** len(FACTORS)


def test_no_cell_of_the_design_is_another_scheduler():
    """The Q-static cell runExperiments adds is not part of the 2^k.

    runExperiments inserts a qlearningSBRC24 cell alongside the eight, which
    is useful to run and must not enter a main effect: it is a different
    learner, not a combination of state factors.
    """
    nomes = [n for n, _ in cells()]
    assert 'qlearningSBRC24' not in nomes
    assert EMPTY_CELL_MSF not in nomes


def test_each_factor_splits_the_design_in_half():
    """A main effect is the mean with minus the mean without, so an
    unbalanced split would weight the halves differently without saying so."""
    for fator in FACTORS:
        com = [n for n, f in cells() if fator in f]
        sem = [n for n, f in cells() if fator not in f]
        assert len(com) == len(sem) == 2 ** (len(FACTORS) - 1)


def test_a_cell_name_lists_its_factors_in_a_fixed_order():
    """The folder name has to match what runExperiments wrote."""
    assert cell_name(['traffic', 'queue', 'charge']) == 'traffic_queue_charge'
    assert cell_name(['queue', 'charge']) == 'queue_charge'


# ------------------------------------- the same hyperparameters everywhere

def test_the_anchor_gives_every_cell_the_same_core_values():
    """Without it each cell keeps its own optimised parameters, and a main
    effect mixes the state factor with a different learning rate."""
    from scenario import load_anchor
    import json
    import os
    caminho = 'anchor_escolhido.json'
    assert os.path.exists(caminho), 'a configuracao escolhida tem que existir'
    ancora = load_anchor(caminho)
    assert 'dynq' in ancora
    for chave in ('ALFA', 'BETA', 'MIN_EPSLON'):
        assert chave in ancora['dynq'], chave


def test_the_scenario_is_declared_in_one_place():
    """The sweep, the factorial and the final comparison read the same list,
    so a run cannot silently disturb the network differently."""
    import scenario
    import runSensitivity
    import runComparison
    import runExperiments
    for modulo in (runSensitivity, runComparison, runExperiments):
        assert modulo.DISTURBANCES is scenario.DISTURBANCES
    assert len(scenario.DISTURBANCES) == 2
    assert [d['asn_fraction'] for d in scenario.DISTURBANCES] == [0.40, 0.70]
