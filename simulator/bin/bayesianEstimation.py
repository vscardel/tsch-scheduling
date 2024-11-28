import argparse
import json
import os


parameters_position = {
   "ALFA": 0,
   "BETA": 1,
   "SLOTFRAME_INTERVAL_SIZE": 2,
   "EPSLON_DECAY_RATE": 3,
   "MIN_EPSLON": 4,
   "MAX_TX_CELLS_PASSED": 5,
   "MAX_RX_CELLS_PASSED": 6,
   "DISCRETIZE_ENERGY_PARAMETER": 7,
   "LAMBDA": 8
}

def efficience_function(parameters, args):

    settings = None

    with open('config.json','r') as f:
        json_string = f.read()
        settings = json.loads(json_string)

    settings['settings']['combination']['exec_numMotes'] = args.combinations
    settings['execution']['numCPUs'] = args.num_cpus
    settings['execution']['numRuns'] = args.num_runs
    settings['settings']['regular']['sf_class'] = args.sched_function
    settings['log_directory_name']= args.output_folder

    #configure simulator with the parameters
    for parameter_name,position in parameters_position.items():
        parameter_value = parameters[position]
        settings['settings']['regular'][parameter_name] = parameter_value

    with open('config.json','w') as f:
        f.write(json.dumps(settings,indent=4))

    #run simulator
    os.system('python2 runSim.py')
    

parser = argparse.ArgumentParser()

parser.add_argument('-nc','--num_cpus', type=int, help='Number of cpu cores', required=True)
parser.add_argument('-nr','--num_runs', type=int, help='Number of times each combination of motes will be run', required=True)
parser.add_argument('-cb','--combinations',type=int, nargs='+', help='combination of number of Motes', required=True)
parser.add_argument('-sf','--sched_function', help='scheduling function alias', required=True)
parser.add_argument('-of','--output_folder', help='output folder name', required=True)

args = parser.parse_args()

parameters = [
    0.5,
    0.2,
    3,
    0.1,
    0.05,
    100,
    100,
    0.5,
    3
]
efficience = efficience_function(parameters, args)
