# tsch-scheduling
Repository containing two new scheduling functions for the TSCH protocol. Those functions are described in detail in the articles referenced in this project README.

# How to run a minimization Experiment
Go to /simulator/bin and run:
```
python2 runExperiments.py -nc <combinations> -nr <num_runs> -sf <scheduling function> -app <app> -of <output_folder_name>  -cc <conn_class> --num_slots <num_slots> --experiment_type minimization
```
# How to run a 2³ Factorial Experiment
Go to /simulator/bin and run:
```
python2 runExperiments.py -nc <combinations> -nr <num_runs> -sf <scheduling function> -app <app> -of <output_folder_name>  -cc <conn_class> --num_slots <num_slots> --experiment_type 2k
```
