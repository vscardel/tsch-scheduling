"""Which rows of the Q-table an agent actually reaches.

A reviewer asks for the distribution of visited states. Nothing in the output
answers it today: the .kpi files describe the network rather than the agent,
and a Q-table saved at the end says what a row holds, not how often the agent
stood in it. The counts only exist while the run happens, so they are kept
here and written out with the rest of the per-mote statistics.

Two counters, both keyed by the row number, both small enough to write for
every mote of every run:

  STATE_VISITS  how often the agent was in the row
  STATE_ACTION  which action it took there, split by whether the Q-table chose
                it or a coin did

The split matters as much as the histogram. A row entered only while exploring
has counts that describe epsilon and not the policy, and a single total cannot
tell the two apart.

Everything is keyed by string and counted in plain dicts, so the structure
survives a JSON round trip unchanged and does not need to know how many
actions a given learner has.
"""


def empty_state_stats():
    return {'STATE_VISITS': {}, 'STATE_ACTION': {}}


def record_state(stats, state):
    visitas = stats['STATE_VISITS']
    chave = str(state)
    visitas[chave] = visitas.get(chave, 0) + 1


def record_action(stats, state, action, explored):
    linha = stats['STATE_ACTION'].setdefault(str(state), {})
    contagem = linha.setdefault('explored' if explored else 'greedy', {})
    chave = str(action)
    contagem[chave] = contagem.get(chave, 0) + 1
