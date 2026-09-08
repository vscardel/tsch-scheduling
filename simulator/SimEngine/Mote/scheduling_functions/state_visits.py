"""What the agent saw, what it did, and what it learned from it.

A reviewer asks for the distribution of visited states. Several more ask
whether the Q-table has converged, and by what criterion. Nothing in the
output answered either: the .kpi files describe the network rather than the
agent, the temporal difference was computed and thrown away, and the Q-table
was never written anywhere, so the learned policy did not survive the run.

The counts and the trace only exist while the run happens, so they are kept
here and written out with the rest of the per-mote statistics.

  STATE_VISITS   how often the agent was in the row
  STATE_ACTION   which action it took there, split by whether the Q-table
                 chose it or a coin did
  DECISIONS      one record per decision: reward, temporal difference, the
                 ASN it happened at, and how many rows changed their
                 preferred action
  Q_TABLE        the table itself, overwritten each decision, so the last
                 write is the policy the run ended with
  GREEDY_POLICY  the preferred action of each row, same overwriting

The split in STATE_ACTION matters as much as the histogram. A row entered
only while exploring has counts that describe epsilon and not the policy, and
a single total cannot tell the two apart.

ASN is what puts the two learners on one time axis. They decide on different
triggers, so their step counts are not comparable and their step numbers are
not either.

Nothing here draws a random number, so an instrumented run scores exactly what
the same run scores without the instrumentation.

Everything is keyed by string and counted in plain dicts, so the structure
survives a JSON round trip unchanged and does not need to know how many
actions a given learner has.
"""


def empty_state_stats():
    return {
        'STATE_VISITS': {},
        'STATE_ACTION': {},
        'DECISIONS': {},
        'Q_TABLE': {},
        'GREEDY_POLICY': {},
    }


def record_state(stats, state):
    visitas = stats['STATE_VISITS']
    chave = str(state)
    visitas[chave] = visitas.get(chave, 0) + 1


def record_action(stats, state, action, explored):
    linha = stats['STATE_ACTION'].setdefault(str(state), {})
    contagem = linha.setdefault('explored' if explored else 'greedy', {})
    chave = str(action)
    contagem[chave] = contagem.get(chave, 0) + 1


def greedy_action(values):
    """The row's preferred action, first index on a tie, as argmax gives."""
    melhor = 0
    for indice in range(1, len(values)):
        if values[indice] > values[melhor]:
            melhor = indice
    return melhor


def greedy_policy(q_table):
    """The preferred action of every row, keyed by row number as a string."""
    return dict(
        (str(linha), greedy_action(valores))
        for linha, valores in q_table.items()
        if len(valores) > 0
    )


def record_decision(stats, step, asn, reward, td_error, q_table):
    """One record for the decision that has just been scored and learned from.

    Called from where the table is updated, so the table it snapshots is the
    table after the update and the count of changed rows is the effect of this
    decision alone. On the first decision there is nothing to compare against
    and the count is zero.
    """
    politica = greedy_policy(q_table)
    anterior = stats['GREEDY_POLICY']
    trocas = sum(
        1 for linha, acao in politica.items()
        if linha in anterior and anterior[linha] != acao
    )

    stats['DECISIONS'][str(step)] = {
        'asn'            : asn,
        'reward'         : reward,
        'td_error'       : td_error,
        'policy_changes' : trocas,
    }
    stats['GREEDY_POLICY'] = politica
    stats['Q_TABLE'] = dict(
        (str(linha), list(valores)) for linha, valores in q_table.items()
    )
