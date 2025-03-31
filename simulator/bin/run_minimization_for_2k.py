import subprocess
import multiprocessing
import itertools


# Lista de nomes para o argumento --of
output_folders = [
    "baseline_minization",
    "traffic_minimization",
    "queue_minimization",
    "charge_minimization",
    "traffic_queue_minimization",
    "traffic_charge_minimization",
    "queue_charge_minimization",
    "traffic_queue_charge_minimization"
]


MAX_PARALLEL_JOBS = min(len(output_folders), multiprocessing.cpu_count())

# Comando base
# talvez tentar treinar com --cc random depois
base_command = [
    "python2", "runExperiments.py",
    "-nc", "1",
    "-nr", "1",
    "-cb", "50",
    "-sf", "Qlearning",
    "-app", "random",
    "-cc", "FullyMeshed",
    "-sr", "true",
    "--num_slots", "3750",
    "--experiment_type", "minimization"
]

def run_experiment(output_folder):
    """Executa um experimento com um valor especifico de --of."""
    command = base_command + ["-of", output_folder] 
    print("Iniciando experimento com --of {0}".format(output_folder))
    subprocess.Popen(command)

if __name__ == "__main__":

    pool = multiprocessing.Pool(processes=MAX_PARALLEL_JOBS)
    pool.map(run_experiment, output_folders)
    pool.close()  
    pool.join()  

