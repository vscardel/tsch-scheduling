import argparse
import json
import os
import glob
import shutil
import random
import itertools
import matplotlib.pyplot as plt
import numpy as np
import time

from skopt import gp_minimize

# the scenario and the anchor, shared by the sweep, the factorial and
# the final comparison so none of them can declare a different one
from scenario import DISTURBANCES, load_anchor
from skopt.plots import plot_convergence

MAX_FUNCTION_VALUE = 1
ALL_SCORES = []

def convert_types(obj):
    if isinstance(obj, dict):
        return {k: convert_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_types(i) for i in obj]
    elif isinstance(obj, np.integer): 
        return int(obj)
    elif isinstance(obj, np.floating): 
        return float(obj)
    else:
        return obj

# The ranges the core Q-learning parameters are chosen from, and why they are
# these ranges.
#
# The environment is non-stationary: every node learns while its neighbours
# learn, and traffic and link quality move underneath all of them. Sutton and
# Barto, "Tracking a Nonstationary Problem", is explicit about what that means
# for the learning rate. With a constant step size the second convergence
# condition is not met, so "the estimates never completely converge but
# continue to vary in response to the most recently received rewards", and
# they add that this "is actually desirable in a nonstationary environment".
# Decaying the rate towards zero would buy a guarantee that does not apply
# here anyway, since it assumes a stationary MDP, and would cost the agent the
# ability to notice change.
#
# So the rate stays constant and the question is only how large. Two 6TiSCH
# Q-learning schedulers from the same group answer it, and one of them is the
# RL-SF baseline in this repository:
#
#   Pratama and Chung, ICEIEC 2022          alpha 0.1   beta 0.95  eps floor 0.1
#   Pratama, Chung and Fawwaz, Access 2024  alpha 0.01  beta 0.95  eps floor 0.1
#
# The published DynQ configuration sits at alpha 0.786, eight to eighty times
# larger, which is a table very nearly rewritten on every visit. The ranges
# below cover both neighbours and exclude that, which is deliberate: a value
# inside them can be defended by citation, and the old one could only be
# defended by the search that produced it.
#
# Q-static still decides whether to explore by comparing epsilon against
# EPSLON_THRESHOLD, so the threshold is a real parameter for it. DynQ is
# epsilon-greedy, so searching it there would spend evaluations on a dimension
# that changes nothing.
SEARCH_SPACE = {
    'Qlearning': [
        ("ALFA",              (0.01, 0.30)),
        ("BETA",              (0.40, 0.95)),
        ("EPSLON_DECAY_RATE", (0.005, 0.30)),
        ("MIN_EPSLON",        (0.01, 0.15)),
    ],
    'QlearningSBRC24': [
        ("ALFA",              (0.01, 0.30)),
        ("BETA",              (0.40, 0.95)),
        ("EPSLON_DECAY_RATE", (0.005, 0.30)),
        ("MIN_EPSLON",        (0.01, 0.15)),
        ("EPSLON_THRESHOLD",  (0.30, 0.90)),
    ],
    # RL-SF gets the same budget and the same number of dimensions as DynQ, so
    # neither method is the only one that was tuned. Its three reward weights
    # are left out for the same reason DynQ's are: searching the reward changes
    # what the agent is being asked to do, not how well it does it.
    'RLSF': [
        ("RLSF_ALFA",          (0.01, 0.30)),
        ("RLSF_BETA",          (0.40, 0.95)),
        ("RLSF_EPSILON_DECAY", (0.95, 0.9999)),
        ("RLSF_EPSILON_END",   (0.01, 0.15)),
    ],
}

def search_space(sched_function):
    """The names and ranges the search varies for a scheduling function."""
    return SEARCH_SPACE.get(sched_function, SEARCH_SPACE['QlearningSBRC24'])


DEFAULT_NUM_EVALUATIONS   = 40
DEFAULT_NUM_RANDOM_STARTS = 10


def optimisation_budget(num_evaluations, num_random_starts):
    """How many evaluations the search gets, and how many of those are random.

    skopt samples at random for its first n_random_starts evaluations and only
    then fits the surrogate. If n_calls equals that number every evaluation is
    a draw, and the search is a random search wearing the name of Bayesian
    optimisation. Returns the pair, or raises if the budget buys no guided
    evaluation at all.
    """
    n_random_starts = num_random_starts or DEFAULT_NUM_RANDOM_STARTS
    n_calls = num_evaluations or DEFAULT_NUM_EVALUATIONS
    if n_calls <= n_random_starts:
        raise ValueError(
            'n_calls ({0}) must exceed n_random_starts ({1}), otherwise no '
            'evaluation is guided by the model and this is a random '
            'search.'.format(n_calls, n_random_starts)
        )
    return n_calls, n_random_starts

parameters_position = []

metrics_vector_position = [
   "latency",
   "join_time",
   "network_lifetime",
   "packet_delivery_ratio",
]

#hotspot scoring function

kpis_weights = {
    'latency': 0.25,
    'pdr': 0.25,
    'lifetime': 0.25,
    'join_time': 0.25
}
kpis_tresholds = {
    'latency': 1.5, #s
    'pdr': 0.95, #%|
    'lifetime': 1, #y
    'join_time': 1000,#s
}

metrics = {
    'latencies': [],
    'pdrs': [],
    'join_times': [],
    'lifetimes': []
}

def smooth_threshold_above(x, T, k=1):
    return 1 / (1 + np.exp(-k * (x - T)))

def smooth_threshold_below(x, T, k=1):
    return (1 - smooth_threshold_above(x, T, k))
########################

def remove_results_folder(subfolder):
    shutil.rmtree(subfolder)
    time.sleep(5)

def load_config():
    settings = None
    with open('config.json', 'r') as f:
        json_string = f.read() 
        settings = json.loads(json_string)
    return settings

def evaluations_path(output_folder):
    return './{0}_evaluations.json'.format(output_folder)


def record_evaluation(output_folder, parameters, value):
    """Append one finished evaluation, so a lost run resumes instead of restarting.

    Every evaluation is a full set of simulations, so a run that dies two
    thirds of the way through used to throw away hours. Two configurations
    never finished at all: each attempt met the same accumulated risk of
    stopping, and starting over reset the progress but not the risk.
    """
    registro = load_evaluations(output_folder)
    registro.append({
        'x': [float(p) for p in parameters],
        'y': float(value),
    })
    with open(evaluations_path(output_folder), 'w') as f:
        json.dump(registro, f)


def load_evaluations(output_folder):
    caminho = evaluations_path(output_folder)
    if not os.path.exists(caminho):
        return []
    try:
        with open(caminho, 'r') as f:
            return json.load(f)
    except Exception:
        # a half-written file is worth less than starting the record over
        return []


def cell_name(factor_combination, empty_is_learner=False):
    """The folder a cell of the factorial writes to."""
    if factor_combination:
        return '_'.join(factor_combination)
    return 'sem_estado' if empty_is_learner else 'baseline'


def load_optimal_parameters(factor_combination):
    """Every hyperparameter the file holds, by name.

    It used to return a positional list built from parameters_position, which
    is the search space of whatever -sf the run was given. The factorial runs
    with -sf Qlearning, which has four hyperparameters, so the Q-static cell
    silently lost its fifth: EPSLON_THRESHOLD never left the file, and that
    cell ran on config.json's 0.58 instead of the 0.30 the optimisation found.
    Reading by name cannot drop a parameter the file bothered to record.
    """
    with open('./{0}_parameters.json'.format(factor_combination), 'r') as f:
        return json.load(f)

def configure_settings(settings, parameters):
    settings['settings']['combination']['exec_numMotes'] = args.combinations
    settings['execution']['numCPUs'] = args.num_cpus
    settings['execution']['numRuns'] = args.num_runs
    settings['settings']['regular']['sf_class'] = args.sched_function
    settings['settings']['regular']['conn_class'] = args.conn_class
    settings['settings']['regular']['exec_numSlotframesPerRun'] = args.num_slots
    factor_combinations = args.factor_combinations
    if not factor_combinations:
        settings['settings']['regular']['factorial_combinations'] = ['traffic', 'queue', 'charge']
    else:
        settings['settings']['regular']['factorial_combinations'] = factor_combinations.split(',')
    settings['log_directory_name']= args.output_folder
    settings['get_sync_node_info'] = args.sync_required
    settings['settings']['regular']['disturbances'] = (
        DISTURBANCES if getattr(args, 'disturbances', False) else []
    )

    # Configure simulator with the parameters. The optimiser hands over a
    # positional list, in the order of parameters_position; a parameters file
    # hands over a mapping, and every key in it is applied.
    if parameters:
        if isinstance(parameters, dict):
            settings['settings']['regular'].update(parameters)
        else:
            for position, parameter_name in enumerate(parameters_position):
                settings['settings']['regular'][parameter_name] = parameters[position]
    return settings


def save_curr_run_config(config_name, settings):
    with open(config_name, 'w') as f:
        json.dump(settings, f, indent=4)

def load_kpis(folder_path, num_motes):
    """Every run's KPIs, merged and keyed by run id.

    This used to open output_cpu0.dat.kpi and nothing else. runSim gives each
    core its own output file, so with N cores it read one run and silently
    discarded the other N-1: the score that drove the optimisation and the
    factorial analysis was computed from a single run while the config asked
    for ten, and the runs that cost the most time were the ones thrown away.

    It also explains why every experiment so far had to be pinned to one core
    to be trustworthy, which is the slowest way to run any of them.
    """
    kpis = {}
    arquivos = sorted(glob.glob(os.path.join(folder_path, '*.dat.kpi')))
    for caminho in arquivos:
        try:
            with open(caminho, 'r') as f:
                for run_id, run in json.loads(f.read()).items():
                    kpis[run_id] = run
        except Exception as e:
            print(e)
            print("Something went wrong reading {0}".format(caminho))
    return kpis or None

def compute_run_lifetime(run_kpis):
    """Mean battery lifetime over the motes of one run, in years.

    A mote whose lifetime could not be estimated reports a string rather than a
    number, and counts as zero.
    """
    lifetimes = []
    for mote, mote_kpis in run_kpis.items():
        if mote == 'global-stats':
            continue
        lifetime = mote_kpis.get('lifetime_AA_years')
        if not isinstance(lifetime, (int, float)):
            lifetime = 0
        lifetimes.append(lifetime)
    if not lifetimes:
        return 0.0
    return sum(lifetimes) / float(len(lifetimes))


def compute_average_lifetime(kpis):
    """Mean battery lifetime over every run."""
    per_run = [compute_run_lifetime(kpis[run]) for run in kpis]
    if not per_run:
        return 0.0
    return sum(per_run) / float(len(per_run))
        

def compute_score(kpis):
    scores = []
    for run in kpis:
        try:
            #seconds
            latency = kpis[run]['global-stats']['e2e-upstream-latency'][0]['mean']
            #convert join_time to seconds
            join_time = kpis[run]['global-stats']["joining-time"][0]['mean'] / 100
            #years
            # this run's lifetime, not the average over every run. Taking the
            # average here gave all the runs the same value, so a quarter of the
            # score was a constant and could not tell one run from another.
            network_lifetime = compute_run_lifetime(kpis[run])
            packet_delivery_ratio = kpis[run]['global-stats']['e2e-upstream-delivery'][0]['value']

            metrics_vector = [
                (latency, 'latency'), 
                (join_time, 'join_time'), 
                (network_lifetime, 'lifetime'), 
                (packet_delivery_ratio, 'pdr')
            ]
            score = 0.0
            for metric in metrics_vector:
                metric_value, metric_name = metric[0],metric[1]
                if metric_name == 'latency':
                    score += kpis_weights[metric_name] * smooth_threshold_above(metric_value, kpis_tresholds[metric_name], k = 1)
                if metric_name == 'join_time':
                    score += kpis_weights[metric_name] * smooth_threshold_above(metric_value, kpis_tresholds[metric_name],k=0.004)
                elif metric_name =='pdr':
                    score += kpis_weights[metric_name] * smooth_threshold_below(metric_value, kpis_tresholds[metric_name], k = 0.2)
                elif metric_name == 'lifetime':
                    score += kpis_weights[metric_name] * smooth_threshold_below(metric_value, kpis_tresholds[metric_name], k = 1.5)
            scores.append(score)
        except Exception as e:
            print(e)
            print('Failed to calculate score. Returning MAX VALUE.')
            scores.append(1)
            return None
    return scores

# to be called by the gp_minimize function
def efficience_function(parameters): 
    global args, ALL_SCORES, metrics

    config_name = 'config_{0}.json'.format(args.output_folder)
    settings = load_config()
    settings = configure_settings(settings, parameters)
    save_curr_run_config(config_name, settings)
    settings = convert_types(settings)

    # Run simulator
    os.system('python2 runSim.py --config {0}'.format(config_name))
    curr_output_folder_path = os.path.join(
        'simData',
        args.output_folder,
        'exec_numMotes_{0}'.format(args.combinations[0])
    )
    os.system('python2 compute_kpis.py --subfolder {0}'.format(curr_output_folder_path))
    time.sleep(3)

    # Get results
    num_motes = settings['settings']['combination']['exec_numMotes'][0]
    kpis = load_kpis(curr_output_folder_path, num_motes)
                
    if kpis:
        scores = compute_score(kpis)
        mean_scores = sum(scores)/float(len(scores))
        remove_results_folder(curr_output_folder_path)
        if mean_scores:
            ALL_SCORES.append(mean_scores)
            record_evaluation(args.output_folder, parameters, mean_scores)
            return mean_scores
        record_evaluation(args.output_folder, parameters, MAX_FUNCTION_VALUE)
        return MAX_FUNCTION_VALUE
    record_evaluation(args.output_folder, parameters, MAX_FUNCTION_VALUE)
    return MAX_FUNCTION_VALUE

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-nc','--num_cpus', type=int, help='Number of cpu cores', required=True)
    parser.add_argument('-nr','--num_runs', type=int, help='Number of times each combination of motes will be run', required=True)
    parser.add_argument('-cb','--combinations',type=int, nargs='+', help='Combination of number of Motes', required=True)
    parser.add_argument('-sf','--sched_function', help='Scheduling function alias', required=True)
    parser.add_argument('-app','--application', help='Application Type', required=True)
    parser.add_argument('-of','--output_folder', help='Output folder name', required=True)
    parser.add_argument('-ne','--num_evaluations', type=int, help='Number of evaluations of efficiency function', required=False)
    parser.add_argument('-af','--aquisition_function', type=str, help='The aquisition function used in gp_minimize', required=False)
    parser.add_argument('-sr','--sync_required', type=bool, help='if sync info is obtained in simulation', required=False)
    parser.add_argument('-nrs','--num_random_starts', type=int, help='num random starts of gp.minimize', required=False)
    parser.add_argument('-rs','--random_state', type=int, default=1, help='seed of the search itself, so the optimisation reproduces', required=False)
    parser.add_argument('-cc','--conn_class', type=str, help='connectivity_matrix', required=True)
    parser.add_argument('-nslots','--num_slots', type=int, help='number of slotframes (time) of simulation', required=True)
    parser.add_argument('-is_min','--experiment_type', type=str, help='determines the time of experiment (minimization or 2^k)', required=True)

    parser.add_argument(
        '--anchor',
        help=(
            'JSON of {learner: {SETTING: value}}. When given, every cell of '
            'the factorial runs these core values instead of its own '
            'optimised parameters file. The factorial needs that: with each '
            'cell separately optimised, a main effect mixes the state factor '
            'with a different learning rate, and the design no longer '
            'measures what it says it measures.'
        )
    )
    parser.add_argument(
        '--disturbances', action='store_true',
        help='run every cell in the disturbed scenario'
    )
    parser.add_argument(
        '--empty-cell-learner', action='store_true',
        help=(
            'run the empty cell as the learner with no state factors, under '
            'the name sem_estado, instead of as MSF under the name baseline. '
            'The factorial needs this: with MSF in that cell, a main effect '
            'compares a state factor against another scheduling function.'
        )
    )
    parser.add_argument(
        '--cells',
        help=(
            'run only these cells of the factorial, by folder name, comma '
            'separated. Without it every cell runs, which is what the '
            'factorial itself wants and what a control arm does not.'
        )
    )
    parser.add_argument(
        '--tag',
        help=(
            'suffix for the output folder and the parameters file, so a '
            'control arm can reuse a cell without overwriting it. With '
            '--tag control, cell traffic_queue_charge writes to '
            'traffic_queue_charge_control and reads its hyperparameters from '
            'traffic_queue_charge_control_parameters.json, falling back to '
            'the cell\'s own file.'
        )
    )
    parser.add_argument(
        '-fc', '--factor_combinations',
        help='List of factor combinations',
        type=str,
        required=False
    )


    args = parser.parse_args()

    # which parameters this scheduling function has, and in what order. Both the
    # minimization and the 2^k path read the order from here.
    espaco = search_space(args.sched_function)
    parameters_position[:] = [nome for nome, _ in espaco]

    if args.experiment_type == 'minimization':
        print('searching {0} parameters: {1}'.format(
            len(parameters_position), ', '.join(parameters_position)))

        try:
            n_calls, n_random_starts = optimisation_budget(
                args.num_evaluations, args.num_random_starts
            )
        except ValueError as erro:
            raise SystemExit(str(erro))
        print('{0} evaluations, {1} of them random, {2} guided'.format(
            n_calls, n_random_starts, n_calls - n_random_starts))

        anteriores = load_evaluations(args.output_folder)
        if anteriores:
            x0 = [e['x'] for e in anteriores]
            y0 = [e['y'] for e in anteriores]
            n_calls = max(1, n_calls - len(anteriores))
            n_random_starts = max(0, n_random_starts - len(anteriores))
            print('resuming with {0} evaluations already done, {1} to go'.format(
                len(anteriores), n_calls))
        else:
            x0 = y0 = None

        res = gp_minimize(efficience_function,
                        [faixa for _, faixa in espaco],
                        n_calls          = n_calls,
                        n_random_starts  = n_random_starts,
                        acq_func         = args.aquisition_function or 'gp_hedge',
                        random_state     = args.random_state,
                        x0               = x0,
                        y0               = y0,
                    )

        time.sleep(30)

        print('Optimal set of parameters\n')
        parameters_to_save = {}
        for i,paramater in enumerate(parameters_position):
            current_value = res.x[i]
            print('{0}: {1}\n'.format(paramater, current_value))
            parameters_to_save[paramater] = current_value

        with open('./{0}_parameters.json'.format(args.output_folder), 'w') as f:
            json.dump(parameters_to_save, f)

        print('Best value: {0}'.format(min(res.func_vals)))

        ys = ALL_SCORES
        xs = [i+1 for i in range(len(ALL_SCORES))]  # Evaluation numbers (x-axis)

        my_plot = plt.plot(xs, ys ,color='red', linewidth=2)
        plt.scatter(xs, ys, color='red', s=50, edgecolors='black', zorder=3)
        plt.title("Optimization Convergence")
        plt.xlabel("Number of Evaluations")
        plt.ylabel("Minimum Objective Function Value")
        plt.xticks(range(1, len(ALL_SCORES)+1, 1))  
        random_num = random.randint(1, 50)
        plt.savefig("all_values_convergence{0}.png".format(random_num))

        ax = plot_convergence(res)
        ax.set_title("Optimization Convergence")
        ax.set_xlabel("Number of Evaluations")
        ax.set_ylabel("Minimum Objective Function Value")
        plt.savefig("convergence_plot{0}.png".format(random_num))

    else:
        print('lets do the 2^k factorial experiment')
        ancora = load_anchor(args.anchor) if args.anchor else None
        if ancora is not None:
            print('  every cell on the chosen core values: {0}'.format(ancora))
        print('  disturbances: {0}'.format(
            len(DISTURBANCES) if args.disturbances else 0))

        # build all possibilities of factors
        factors = ['traffic', 'queue', 'charge']
        all_combinations = []
        combinations = list(itertools.product([0, 1], repeat=3))
        for combination in combinations:
            current_combination = []
            for i,index in enumerate(combination):
                if index:
                    current_combination.append(factors[i])
            all_combinations.append(current_combination)

        all_combinations.sort(key=len)
        all_combinations.reverse()
        all_combinations.insert(0,['qlearningSBRC24'])

        pedidas = (
            [c.strip() for c in args.cells.split(',') if c.strip()]
            if args.cells else None
        )
        if pedidas:
            conhecidas = set(
                cell_name(c, args.empty_cell_learner) for c in all_combinations
            )
            desconhecidas = [c for c in pedidas if c not in conhecidas]
            if desconhecidas:
                raise ValueError(
                    'no such cell: {0}. The cells are {1}'.format(
                        ', '.join(desconhecidas), ', '.join(sorted(conhecidas))
                    )
                )
            all_combinations = [
                c for c in all_combinations
                if cell_name(c, args.empty_cell_learner) in pedidas
            ]

        # run each combination
        for factor_combination in all_combinations:

            cell = cell_name(factor_combination, args.empty_cell_learner)
            output_folder = '{0}_{1}'.format(cell, args.tag) if args.tag else cell

            # The hyperparameters of this cell. With an anchor they are the
            # same in every cell, which is what lets a main effect be read as
            # the effect of the state factor. Without one, each cell keeps
            # the parameters its own optimisation found, which is how the
            # published factorial ran.
            if ancora is not None:
                aprendiz = ('qstatic'
                            if factor_combination == ['qlearningSBRC24']
                            else 'dynq')
                parameters_list = dict(ancora.get(aprendiz, {}))
            elif not factor_combination:
                parameters_list = []
            else:
                parameters_list = load_optimal_parameters(
                    output_folder if args.tag and os.path.exists(
                        './{0}_parameters.json'.format(output_folder)
                    ) else cell
                )

            config_name = 'config_{0}.json'.format(output_folder)
            settings = load_config()
            settings = configure_settings(settings, parameters_list)
            settings['log_directory_name'] = output_folder
            settings['settings']['regular']['factorial_combinations'] = factor_combination
            settings['settings']['regular']['STATE_SIZE'] = 2**(len(factor_combination))
            
            #baseline runs MSF, unless the empty cell was asked to learn
            if not factor_combination and not args.empty_cell_learner:
                settings['settings']['regular']['sf_class'] = 'MSF'
            elif factor_combination == ['qlearningSBRC24']:
                settings['settings']['regular']['sf_class'] = 'QlearningSBRC24'

            save_curr_run_config(config_name, settings)
            settings = convert_types(settings)   

            os.system('python2 runSim.py --config {0}'.format(config_name))

            curr_output_folder_path = os.path.join(
                'simData',
                output_folder,
                'exec_numMotes_{0}'.format(args.combinations[0])
            )

            os.system('python2 compute_kpis.py --subfolder {0}'.format(curr_output_folder_path))
            os.system('python2 plot.py --inputfolder {0}'.format(curr_output_folder_path))

            import time 
            time.sleep(10)

            kpis = load_kpis(curr_output_folder_path, args.combinations[0])
            scores = compute_score(kpis)
            final_results = {
                'score': scores, 
                # to be computed
                'comulative reward': None
            }
            with open(os.path.join(curr_output_folder_path, 'final_results.json'), 'w') as f:
                json.dump(final_results, f, indent=4)
            time.sleep(2)
