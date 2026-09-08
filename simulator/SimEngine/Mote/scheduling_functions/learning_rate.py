"""How big a step each Q-table update takes.

Watkins and Dayan give two conditions for Q-learning to converge to the
optimal policy. The first is on the learning rate: the sum of the rates must
diverge and the sum of their squares must converge. A constant rate satisfies
the first half and fails the second, and the consequence is visible rather
than theoretical: the estimate never settles, it tracks, moving by alpha times
the noise of the latest sample forever. The published DynQ configuration uses
0.79, so its table is very nearly rewritten on every visit.

alpha / (1 + n/tau) falls like c/n, so it passes both halves, with n counted
per state-action pair, which is how the condition is stated. It is the one
change that lets the convergence criterion say yes to anything.

A tau of zero keeps the rate constant, which is the default, so every run made
before this existed is reproduced exactly.

Nothing here draws a random number.
"""


def step_size(alfa, tau, visits, key):
    """The rate for this update, counting the update as it goes.

    The count used is of updates already applied to this cell, so the first
    update to a cell takes the full rate however small tau is.
    """
    n = visits.get(key, 0)
    visits[key] = n + 1
    if not tau:
        return alfa
    return alfa / (1.0 + n / float(tau))
