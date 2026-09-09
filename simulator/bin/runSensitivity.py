"""One factor at a time, around each learner's published configuration.

Two reviewers asked for this and nothing in the manuscript answers either.
Reviewer 1 calls the energy weight of 0.01 negligible and asks what the reward
does without it, and says the moving average window of 10 is used without
justification. That window is not a detail: it is what the dynamic learner
discretises with, so it is the thing that separates it from the static one.

The design is the conventional one on purpose. Every arm is the published
configuration with a single setting changed, on the same seeds, so each arm
pairs run by run against its own baseline and the difference is attributable
to that one setting.

Three factors go beyond what was asked, because the convergence question needs
them: the learning rate, tuned to 0.79 in DynQ and therefore nearly
memoryless; the knobs that govern how much the agent explores and how far
ahead it looks; and a decaying learning rate, which is the only setting here
that can make the convergence criterion say yes, since every constant rate
fails the Watkins condition alike.

Each learner is swept on the knobs it actually has. The energy weight is not
in the static learner, whose reward is Equation 11 and has no weights, and the
phase threshold is not in the dynamic one, which tosses a coin per decision.
Sweeping a knob a method does not have would produce a row that means nothing.

Usage:
    python runSensitivity.py --learners dynq qstatic
    python runSensitivity.py --factors alfa,janela --dry-run
"""
from __future__ import print_function

import argparse
import json
import os

from runExperiments import search_space


FACTORS_STATE = ['traffic', 'queue', 'charge']

# The values of tau are small on purpose. Measured on the runs of 2026-09-08,
# the median cell of the table is updated five times at the published length,
# so a tau of 50 would trim the rate by a tenth and a tau of 200 would do
# nothing at all.
DECAY_GRID = [1, 3, 10]

# Three sets of factors, and which one is swept is a command line choice.
#
#   core     the Q-learning parameters, over the ranges runExperiments defines
#            from the literature. These are chosen first, because the task
#            parameters are only meaningful around a given operating point.
#   tarefa   what the agent is being asked to do: the reward weights, the
#            moving average window, the charge threshold. Swept afterwards,
#            around the core values that won.
#   legado   the factors of the sweep of 2026-09-08, kept so that run can be
#            reproduced. It answered R1.6 and R1.7 for the configuration the
#            reviewers read.
#
# Every core grid point means something, rather than being an even division
# of the range. A reviewer reading "alpha in {0.01, 0.1, 0.3}" sees the two
# values the neighbouring 6TiSCH papers use and the ceiling of the range;
# reading "0.0825, 0.155, 0.2275" sees arithmetic. Three points per factor
# also keeps the sweep inside its time budget, and each one is either cited or
# an endpoint of the range the ranges test pins.
CORE_GRIDS = [
    #  label             setting              points, and what each one is
    ('alfa',    'ALFA',              [0.01, 0.1, 0.3]),
    #                                 Access 2024, ICEIEC 2022, ceiling
    ('beta',    'BETA',              [0.4, 0.7, 0.95]),
    #                                 floor, middle, both neighbours
    ('epsilon', 'MIN_EPSLON',        [0.01, 0.1, 0.15]),
    #                                 floor, both neighbours, ceiling
    ('limiar',  'EPSLON_THRESHOLD',  [0.3, 0.6, 0.9]),
    #                                 published, and the two the sweep of
    #                                 2026-09-08 found better without
    #                                 disturbances
]

# The published values that fall outside the literature box, run anyway as
# one extra arm each so the curve of their own factor shows what the box
# costs. They are references, not candidates: the configuration the revised
# paper adopts stays inside the box even if one of these wins, and if one wins
# the report says by how much.
#
#   DynQ's alpha of 0.786 is 2.6 times the ceiling of the box. The probe of
#   2026-09-09 already measured it as a tie with 0.1 on the score, with and
#   without disturbances, and worse on 6P overhead.
#   Q-static's beta of 0.05 is below the floor of 0.40, which makes the agent
#   nearly myopic. Nothing has measured it against a box value yet, so its
#   own sweep does that here.
CORE_REFERENCES = {
    'dynq'   : [('ALFA', 0.7863156043331933),
                ('MIN_EPSLON', 0.19096869607843248)],
    'qstatic': [('BETA', 0.05)],
}

# Left at its published value in the core sweep. It governs how quickly
# epsilon reaches its floor rather than where the floor is, the published
# 0.0151 is already inside the range, and spending five arms per learner on it
# would cost more than the question is worth right now.
CORE_LEFT_ALONE = ['EPSLON_DECAY_RATE']


def core_factors(sf_class):
    """One factor per core parameter the learner actually has.

    The learner's own search space decides which: DynQ has no phase threshold
    to cross, so sweeping one there would produce a row that means nothing.
    """
    tem = set(nome for nome, _ in search_space(sf_class))
    return [
        (label, setting, pontos)
        for label, setting, pontos in CORE_GRIDS
        if setting in tem
    ]


LEARNERS = {
    'dynq': {
        'sf_class'  : 'Qlearning',
        'parameters': 'traffic_queue_charge',
        'tarefa'    : [
            ('w_energy',      'W_ENERGY',                [0.0, 0.01, 0.5, 2.0]),
            ('w_latency',     'W_LATENCY',               [0.0, 0.5, 2.0, 5.0]),
            ('w_throughput',  'W_THROUGHPUT',            [0.0, 0.5, 2.0, 5.0]),
            ('w_utilization', 'W_UTILIZATION',           [0.0, 0.5, 2.0, 5.0]),
            ('janela',        'SLOTFRAME_INTERVAL_SIZE', [1, 3, 20, 50]),
        ],
        'legado'    : [
            ('w_energy', 'W_ENERGY',                [0.0, 0.01, 0.1, 5.0]),
            ('janela',   'SLOTFRAME_INTERVAL_SIZE', [1, 3, 20, 50]),
            ('alfa',     'ALFA',                    [0.05, 0.2, 0.5, 0.95]),
            ('epsilon',  'MIN_EPSLON',              [0.0, 0.05, 0.4, 0.8]),
            ('beta',     'BETA',                    [0.0, 0.2, 0.7, 0.9]),
            ('decaimento', 'ALFA_DECAY_TAU',        DECAY_GRID),
        ],
    },
    'qstatic': {
        'sf_class'  : 'QlearningSBRC24',
        'parameters': 'qlearningSBRC24',
        # the window does nothing here: the discretisation reads the raw
        # value against a fixed threshold, which is what makes it the static
        # learner, so its only task parameter is the charge threshold
        'tarefa'    : [
            ('tau_charge', 'QSTATIC_TAU_CHARGE', [0.2, 0.35, 0.65, 0.8]),
        ],
        'legado'    : [
            ('janela',   'SLOTFRAME_INTERVAL_SIZE', [1, 3, 20, 50]),
            ('alfa',     'ALFA',                    [0.01, 0.2, 0.5, 0.9]),
            ('beta',     'BETA',                    [0.0, 0.3, 0.6, 0.9]),
            ('limiar',   'EPSLON_THRESHOLD',        [0.0, 0.1, 0.6, 1.0]),
            ('decaimento', 'ALFA_DECAY_TAU',        DECAY_GRID),
        ],
    },
}


def factors_of(learner, grupo):
    """The factor table for one learner and one group.

    The core group is derived from the ranges rather than written out, so the
    grid and the ranges cannot drift apart.
    """
    if grupo == 'core':
        return core_factors(LEARNERS[learner]['sf_class'])
    return LEARNERS[learner][grupo]


def load_parameters(nome):
    with open('./{0}_parameters.json'.format(nome)) as f:
        return json.load(f)


def load_anchor(path):
    """Per learner, the settings the arms of this sweep depart from.

    A coordinate sweep measures sensitivity around one point, so the point has
    to be the one being defended. Two reasons it is no longer the published
    configuration. The published alpha of the dynamic learner, 0.786, sits
    outside the range the 6TiSCH literature uses, and the probe of 2026-09-09
    measured it as a tie with 0.1 both with disturbances and without, so the
    citable value costs nothing on the score. And the sweep of the task
    parameters runs after this one, anchored on whatever this one chooses.

    The file is read as {learner: {SETTING: value}} and applied last, so it
    wins over the published parameters.
    """
    if not path:
        return {}
    with open(path) as f:
        return json.load(f)


# The two disturbances of the probe, in the positions it validated: 40% of the
# run for the agent to settle before the first, and 30% after the last to
# measure the recovery. Both together cost about 0.077 of score on the
# published DynQ, ten runs out of ten.
DISTURBANCES = [
    {'asn_fraction': 0.40, 'kind': 'traffic', 'factor': 0.5},
    {'asn_fraction': 0.70, 'kind': 'link_quality', 'share': 0.2, 'pdr': 0.4},
]


def baseline_settings(base, learner, anchor=None):
    """The configuration of one learner, as its arms depart from it.

    The published parameters, unless an anchor overrides some of them.
    """
    regular = json.loads(json.dumps(base))['settings']['regular']
    spec = LEARNERS[learner]
    regular.update(load_parameters(spec['parameters']))
    regular['sf_class'] = spec['sf_class']
    if spec['sf_class'] == 'Qlearning':
        regular['factorial_combinations'] = FACTORS_STATE
        regular['STATE_SIZE'] = 2 ** len(FACTORS_STATE)
    regular.update((anchor or {}).get(learner, {}))
    return regular


def arm_name(learner, factor, valor):
    """A folder name that survives being a float."""
    texto = ('%g' % valor).replace('.', 'p').replace('-', 'm')
    return '{0}_{1}_{2}'.format(learner, factor, texto)


def plan(base, learners, grupo='core', factors=None, anchor=None):
    """Every arm to run: one baseline per learner, then one per swept value.

    A value equal to the published one is not run again under another name. It
    is the baseline, and an arm compared against itself would report a
    difference of exactly zero on every metric, which in this project has
    twice meant a parameter that never arrived.
    """
    arms = []
    for learner in learners:
        regular = baseline_settings(base, learner, anchor)
        arms.append(('{0}_base'.format(learner), learner, None, None))
        escolhidos = [
            f for f in factors_of(learner, grupo)
            if not factors or f[0] in factors
        ]
        rotulo = dict((f[1], f[0]) for f in escolhidos)
        referencias = [
            (rotulo[setting], setting, valor)
            for setting, valor in CORE_REFERENCES.get(learner, [])
            if grupo == 'core' and setting in rotulo
        ]
        for factor, setting, valores in escolhidos:
            publicado = regular.get(setting, 0)
            extras = [v for f, s_, v in referencias if s_ == setting]
            for valor in list(valores) + extras:
                if valor == publicado:
                    continue
                arms.append(
                    (arm_name(learner, factor, valor), learner, setting, valor)
                )
    if factors:
        conhecidos = set()
        for nome in LEARNERS:
            conhecidos.update(f[0] for f in factors_of(nome, grupo))
        faltando = set(factors) - conhecidos
        if faltando:
            raise ValueError('no such factor: {0}'.format(', '.join(faltando)))
    return arms


def build_config(base, arm, num_motes, num_runs, num_cpus, slotframes,
                 anchor=None, disturbed=False):
    label, learner, setting, valor = arm
    settings = json.loads(json.dumps(base))
    regular = baseline_settings(base, learner, anchor)
    if setting is not None:
        regular[setting] = valor
    regular['exec_numSlotframesPerRun'] = slotframes
    regular['disturbances'] = DISTURBANCES if disturbed else []
    settings['settings']['regular'] = regular
    settings['settings']['combination']['exec_numMotes'] = [num_motes]
    settings['execution'] = {'numRuns': num_runs, 'numCPUs': num_cpus}
    settings['log_directory_name'] = '{0}_n{1}'.format(label, num_motes)
    return settings


def manifest(arms, grupo='core', anchor=None, disturbed=False):
    """What each arm changed, so the report can group the sweeps by factor.

    The anchor and the disturbances go on every entry rather than in a key of
    their own, because the report walks the entries and an entry that is not
    an arm would have no factor to group by.
    """
    saida = {}
    comum = {'group': grupo, 'disturbed': bool(disturbed)}
    for label, learner, setting, valor in arms:
        entrada = dict(comum)
        entrada['anchor'] = (anchor or {}).get(learner, {})
        entrada['learner'] = learner
        if setting is None:
            entrada.update({'factor': 'baseline', 'setting': None,
                            'value': None, 'baseline': label})
            saida[label] = entrada
            continue
        factor = [
            f[0] for f in factors_of(learner, grupo) if f[1] == setting
        ][0]
        entrada.update({'factor': factor, 'setting': setting, 'value': valor,
                        'baseline': '{0}_base'.format(learner)})
        saida[label] = entrada
    return saida


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--motes', type=int, default=50)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--cpus', type=int, default=10)
    parser.add_argument('--slotframes', type=int, default=3750)
    parser.add_argument('--learners', nargs='+', default=['dynq', 'qstatic'])
    parser.add_argument('--group', default='core',
                        choices=['core', 'tarefa', 'legado'],
                        help='which set of factors to sweep')
    parser.add_argument('--factors', help='comma separated labels, default all')
    parser.add_argument('--arms', help='comma separated arm names to run')
    parser.add_argument('--anchor',
                        help='JSON of {learner: {SETTING: value}}, the point '
                             'the arms depart from, default the published one')
    parser.add_argument('--disturbances', action='store_true',
                        help='run every arm in the disturbed scenario')
    parser.add_argument('--dry-run', action='store_true',
                        help='write the configs and the manifest, run nothing')
    args = parser.parse_args()

    pedidos = (
        [f.strip() for f in args.factors.split(',')] if args.factors else None
    )
    with open('config.json') as f:
        base = json.load(f)
    anchor = load_anchor(args.anchor)

    arms = plan(base, args.learners, args.group, pedidos, anchor)
    if args.arms:
        querido = [a.strip() for a in args.arms.split(',')]
        arms = [a for a in arms if a[0] in querido]

    if args.arms:
        # The supervisor runs one arm per container, so writing the manifest
        # here would leave the last arm's name as the whole manifest, and the
        # report would silently describe a sweep of one. The manifest of the
        # full plan is written by the dry run that precedes the sweep.
        print('{0} bracos pedidos, manifesto preservado'.format(len(arms)))
    else:
        with open('sensitivity_manifest.json', 'w') as f:
            json.dump(
                manifest(arms, args.group, anchor, args.disturbances),
                f, indent=2
            )
        print(
            '{0} bracos, manifesto em sensitivity_manifest.json'.format(
                len(arms)
            )
        )
    for learner in sorted(anchor):
        print('  ancora {0}: {1}'.format(learner, anchor[learner]))
    print('  perturbacoes: {0}'.format(
        len(DISTURBANCES) if args.disturbances else 0))

    for arm in arms:
        settings = build_config(
            base, arm, args.motes, args.runs, args.cpus, args.slotframes,
            anchor, args.disturbances
        )
        nome = settings['log_directory_name']
        config_name = 'config_{0}.json'.format(nome)
        with open(config_name, 'w') as f:
            json.dump(settings, f, indent=4)
        if args.dry_run:
            mudou = ('{0}={1}'.format(arm[2], arm[3]) if arm[2]
                     else ('ancora' if anchor else 'publicado'))
            print('  {0:28s} {1}'.format(nome, mudou))
            continue

        print('=== {0} ==='.format(nome))
        os.system('python2 runSim.py --config {0}'.format(config_name))
        pasta = os.path.join(
            'simData', nome, 'exec_numMotes_{0}'.format(args.motes)
        )
        os.system('python2 compute_kpis.py --subfolder {0}'.format(pasta))
        print('=== {0} pronto ==='.format(nome))


if __name__ == '__main__':
    main()
