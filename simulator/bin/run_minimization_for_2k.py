import subprocess
import multiprocessing
import itertools


# Lista de nomes para o argumento --of
output_folders = [
    "traffic",
    "queue",
    "charge",
    "traffic_queue",
    "traffic_charge",
    "queue_charge",
    "traffic_queue_charge"
]


MAX_PARALLEL_JOBS = min(len(output_folders), multiprocessing.cpu_count())

# Comando base
# talvez tentar treinar com --cc random depois
base_command = [
    "python2", "runExperiments.py",
    "-nc", "1",
    "-nr", "3",
    "-cb", "50",
    "-sf", "Qlearning",
    "-app", "random",
    "-cc", "FullyMeshed",
    "-sr", "true",
    "--num_slots", "3750",
    "--experiment_type", "minimization"
]

def run_experiment(output_folder):
    factor_combinations = output_folder.split('_')  
    combination_command = ''
    for combination in factor_combinations:
        combination_command += (combination + ',')
    combination_command = combination_command[:-1]
    command = base_command + ["-of", output_folder, "-fc", combination_command] 
    print(command)
    print("Iniciando experimento com --of {0} e --fc {1}".format(output_folder, combination_command))
    subprocess.Popen(command)

if __name__ == "__main__":

    pool = multiprocessing.Pool(processes=MAX_PARALLEL_JOBS)
    pool.map(run_experiment, output_folders)
    pool.close()  
    pool.join()  

