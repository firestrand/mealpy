#!/usr/bin/env python
# Created by "Thieu" at 10:08, 02/03/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import os
from pathlib import Path
from opfunu.cec_basic import cec2014_nobias
from pandas import DataFrame

from mealpy.math_based.RandomWalk import UnbiasedRandomWalk
from mealpy.swarm_based.SPSO import SPSO2011
from joblib import Parallel, delayed

def append_or_create_csv(df: DataFrame, file_path: str):
    """
    Appends data to an existing CSV file or creates a new one if it doesn't exist.
    """
    print("Full path:", os.path.abspath(file_path))
    if os.path.exists(file_path):
        df.to_csv(file_path, mode='a', header=False, index=False)
    else:
        df.to_csv(file_path, header=True, index=False)

def run_benchmark(func_name, model_class, lb, ub, epoch, pop_size, wf, cr):
    """
    Runs the benchmark for a single function and returns results.
    """
    problem = {
        "fit_func": getattr(cec2014_nobias, func_name),
        "lb": lb,
        "ub": ub,
        "minmax": "min",
        "log_to": "console",
    }
    model = model_class(epoch, pop_size, step_size=0.05, fit_name=func_name)
    # Initialize SPSO2011 model
    #model = model_class(epoch, pop_size, wf, cr, fit_name=func_name)
    _, best_fitness = model.solve(problem)
    return func_name, model.history.list_global_best_fit, best_fitness

# Setting up parameters
PATH_RESULTS = "history/results/"
Path(PATH_RESULTS).mkdir(parents=True, exist_ok=True)

model_class = UnbiasedRandomWalk
model_name = model_class.__name__
lb1 = [-100] * 30
ub1 = [100] * 30
epoch = 100
pop_size = 40
wf = 0.8
cr = 0.9

func_names = ["F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9", "F10", "F11", "F12", "F13", "F14", "F15", "F16", "F17", "F18", "F19"]

# Running benchmarks in parallel
results = Parallel(n_jobs=-1)(delayed(run_benchmark)(func_name, model_class, lb1, ub1, epoch, pop_size, wf, cr) for func_name in func_names)

# Process and save results
error_full = {}
best_fit_full = {}
for func_name, error_data, best_fitness in results:
    error_full[func_name] = error_data
    best_fit_full[func_name] = [best_fitness]

df_err = DataFrame(error_full)
csv_path_err = f"{PATH_RESULTS}{len(lb1)}D_{model_name}_error.csv"
append_or_create_csv(df_err, csv_path_err)

df_fit = DataFrame(best_fit_full)
csv_path_fit = f"{PATH_RESULTS}{len(lb1)}D_{model_name}_best_fit.csv"
append_or_create_csv(df_fit, csv_path_fit)
