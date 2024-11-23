import argparse
import json
import os

parser = argparse.ArgumentParser()

parser.add_argument('-nc','--num_cpus', type=int, help='Number of cpu cores', required=True)
parser.add_argument('-nr','--num_runs', type=int, help='Numberr of times each combination of motes will be run', required=True)
parser.add_argument('-cb','--combinations',type=int, nargs='+', help='combination of number of Motes', required=True)
parser.add_argument('-sf','--sched_function', help='scheduling function alias', required=True)


args = parser.parse_args()
settings = None

with open('config.json','r') as f:
    json_string = f.read()
    settings = json.loads(json_string)
    
settings['settings']['combination']['exec_numMotes'] = args.combinations
settings['execution']['numCPUs'] = args.num_cpus
settings['execution']['numRuns'] = args.num_runs
settings['settings']['regular']['sf_class'] = args.sched_function

with open('config.json','w') as f:
    f.write(json.dumps(settings,indent=4))

print(f'Executando experimento com parâmetros num_cpus={args.num_cpus}, num_runs={args.num_runs}, combinations={args.combinations}, sched_function={args.sched_function}' )
os.system('python2 runSim.py')


