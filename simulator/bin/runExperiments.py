import argparse
import json
import os
import shutil
from skopt import gp_minimize
from skopt.plots import plot_convergence
import matplotlib.pyplot as plt
import numpy as np
import random

MAX_FUNCTION_VALUE = 1
NUM_RUNS = 0
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

# parameters_position = [
#    "ALFA",
#    "BETA",
#    "SLOTFRAME_INTERVAL_SIZE",
#    "EPSLON_DECAY_RATE",
#    "MIN_EPSLON",
#    "MAX_TX_CELLS_PASSED",
#    "MAX_RX_CELLS_PASSED",
#    "EPSLON_THRESHOLD"
# ]

parameters_position = [
   "ALFA",
   "BETA",
   "EPSLON_DECAY_RATE",
   "MIN_EPSLON",
   "EPSLON_THRESHOLD"
]

metrics_vector_position = [
   "latency",
   "join_time",
   "network_lifetime",
   "packet_delivery_ratio",
]

max_values = {
   "latency": -1,
   "join_time": -1,
   "network_lifetime": -1,
   "packet_delivery_ratio": -1,
}

min_values = {
   "latency": 1000000,
   "join_time": 1000000,
   "network_lifetime": 1000000,
   "packet_delivery_ratio": 1000000,
}

weigths = {
    "latency": 0.2,
   "join_time": 0.2,
   "network_lifetime": 0.2,
   "packet_delivery_ratio": 0.2,
   "average_sync_nodes_in_simulation": 0.2
}

def update_min_max_values(name, value):
    if value > max_values[name]:
        max_values[name] = value
    
    if value < min_values[name]:
        min_values[name] = value

def remove_results_folder(args):
    shutil.rmtree(        
            os.path.join(
            'simData', 
            args.output_folder
        )
    )

def normalize_metric(name,value):
    if max_values[name] == min_values[name]:
        return 1
    return (value - min_values[name]) / (max_values[name] - min_values[name])

def normalize(metrics):
    normalized_metrics = [0] * len(metrics_vector_position)
    for position,name in enumerate(metrics_vector_position):
        normalized_metrics[position] = normalize_metric(name, metrics[position])
    return normalized_metrics

def efficience_function(
        parameters, 
        experiment_type = 'minimization',
        factorial_combinations = []
    ):
    global NUM_RUNS, ALL_SCORES
    # import ipdb;
    # ipdb.set_trace()
    global args

    settings = None

    config_name = 'config_{0}.json'.format(args.output_folder)
    try:
        with open(config_name,'r') as f:
            json_string = f.read()
            settings = json.loads(json_string)
    except IOError as err:
        with open('config.json', 'r') as f:
            json_string = f.read() 
        with open(config_name, 'w') as f:
            f.write(json_string)
        settings = json.loads(json_string)

    settings['settings']['combination']['exec_numMotes'] = args.combinations
    settings['execution']['numCPUs'] = args.num_cpus
    settings['execution']['numRuns'] = args.num_runs
    settings['settings']['regular']['sf_class'] = args.sched_function
    settings['settings']['regular']['conn_class'] = args.conn_class
    settings['settings']['regular']['exec_numSlotframesPerRun'] = args.num_slots
    settings['log_directory_name']= args.output_folder
    settings['get_sync_node_info'] = args.sync_required

    if experiment_type != 'minimization':
        settings['factorial_combinations'] = factorial_combinations

    # Configure simulator with the parameters
    for position, parameter_name in enumerate(parameters_position):
        parameter_value = parameters[position]
        settings['settings']['regular'][parameter_name] = parameter_value

    settings = convert_types(settings)

    with open('config.json', 'w') as f:
        f.write(json.dumps(settings, indent=4))

    # Run simulator
    os.system('python2 runSim.py')

    # Get results
    num_motes = settings['settings']['combination']['exec_numMotes'][0]

    kpis, sync_info = None, None
    for tentativa in range(3):
        try:
            with open(
                os.path.join(
                    'simData', 
                    args.output_folder, 
                    'exec_numMotes_{0}.dat.kpi'.format(num_motes)
                )
            , 'r') as f:
                json_string = f.read()
                kpis = json.loads(json_string)
            with open(
                os.path.join(
                    'simData', 
                    args.output_folder,
                    'exec_numMotes_{0}'.format(num_motes), 
                    'run_0/sync_info.json'
                )
            , 'r') as f:
                json_string = f.read()
                sync_info = json.loads(json_string)                
                break
        except Exception as e:
            print(e)
            print("Something went wrong reading KPIs on try {0}".format(tentativa))
                
    if kpis and sync_info:


        latency = kpis['0']['global-stats']['e2e-upstream-latency'][0]['mean']
        join_time = kpis['0']['global-stats']["joining-time"][0]['mean']
        network_lifetime = kpis['0']['global-stats']['network_lifetime'][0]['min']
        packet_delivery_ratio = kpis['0']['global-stats']['e2e-upstream-delivery'][0]['value']
        sync_list = [v for k,v in sync_info.items()]

        update_min_max_values('latency',latency)
        update_min_max_values('join_time',join_time)
        update_min_max_values('network_lifetime',network_lifetime)
        update_min_max_values('packet_delivery_ratio',packet_delivery_ratio)

        #first run is only to update min_max values
        if NUM_RUNS == 0:
            NUM_RUNS = NUM_RUNS + 1
            remove_results_folder(args)
            ALL_SCORES.append(MAX_FUNCTION_VALUE)
            return MAX_FUNCTION_VALUE

        # import ipdb;
        # ipdb.set_trace()
        try:
            metrics_vector = [latency, join_time, network_lifetime, packet_delivery_ratio]
            #normalize results
            normalized_results = normalize(metrics_vector)

            score = (1 - normalized_results[0] )* weigths['latency'] + \
                    (1 - normalized_results[1]) * weigths['join_time']  + \
                    normalized_results[2] * weigths['network_lifetime']  + \
                    normalized_results[3] * weigths['packet_delivery_ratio'] 
                    
        except Exception as e:
            print(e)
            print('Failed to calculate score. Returning infinity.')
            remove_results_folder(args)
            return MAX_FUNCTION_VALUE
        
        # Remove results folder
        remove_results_folder(args)
        NUM_RUNS = NUM_RUNS + 1

        if score != 0:
            # import ipdb;
            # ipdb.set_trace()
            ALL_SCORES.append(1-score)
            return 1 - score
        else:
            return MAX_FUNCTION_VALUE
    else:
        NUM_RUNS = NUM_RUNS + 1
        return MAX_FUNCTION_VALUE

parser = argparse.ArgumentParser()

parser.add_argument('-nc','--num_cpus', type=int, help='Number of cpu cores', required=True)
parser.add_argument('-nr','--num_runs', type=int, help='Number of times each combination of motes will be run', required=True)
parser.add_argument('-cb','--combinations',type=int, nargs='+', help='Combination of number of Motes', required=True)
parser.add_argument('-sf','--sched_function', help='Scheduling function alias', required=True)
parser.add_argument('-app','--application', help='Application Type', required=True)
parser.add_argument('-of','--output_folder', help='Output folder name', required=True)
parser.add_argument('-ne','--num_evaluations', type=int, help='Number of evaluations of efficiency function', required=False)
parser.add_argument('-af','--aquisition_function', type=str, help='The aquisition function used in gp_minimize', required=False)
parser.add_argument('-sr','--sync_required', type=bool, help='if sync info is obtained in simulation', required=True)
parser.add_argument('-nrs','--num_random_starts', type=int, help='num random starts of gp.minimize', required=False)
parser.add_argument('-cc','--conn_class', type=str, help='connectivity_matrix', required=True)
parser.add_argument('-nslots','--num_slots', type=int, help='number of slotframes (time) of simulation', required=True)
parser.add_argument('-is_min','--experiment_type', type=str, help='determines the time of experiment (minimization or 2^k)', required=True)



args = parser.parse_args()

if args.experiment_type == 'minimization':
    res = gp_minimize(efficience_function,  # The function to minimize
                    [(0.1, 0.9),  # ALFA
                    (0.1, 0.9),  # BETA
                    #    (5,10),     # SLOTFRAME_INTERVAL_SIZE
                    (0.001, 0.005), # EPSLON_DECAY_RATE
                    (0.05, 0.1),  # MIN_EPSLON
                    #    (50, 100),   # MAX_TX_CELLS_PASSED
                    #    (50, 100),   # MAX_RX_CELLS_PASSED
                    (0.5, 0.7)],   # EPSLON_THRESHOLD   
                )


    print('Optimal set of parameters\n')
    parameters_to_save = {}
    for i,paramater in enumerate(parameters_position):
        current_value = res.x[i]
        print('{0}: {1}\n'.format(paramater, current_value))
        parameters_to_save[paramater] = current_value

    with open('./optimal_set_of_paramaters.json', 'w') as f:
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
