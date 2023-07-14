#!/usr/bin/env python
# Created by "Thieu" at 11:37, 02/03/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import concurrent.futures as parallel
import itertools
from pathlib import Path
from opfunu.cec_basic import cec2014_nobias
from pandas import DataFrame

from mealpy import get_all_optimizers, get_optimizer_by_name, get_all_optimizer_names
from mealpy.bio_based.BBO import OriginalBBO
from mealpy.evolutionary_based.DE import BaseDE
from mealpy.evolutionary_based.ES import OriginalES
from mealpy.evolutionary_based.GA import MultiGA
from mealpy.evolutionary_based.MA import OriginalMA
from mealpy.evolutionary_based.SHADE import OriginalSHADE
from mealpy.swarm_based.MFO import OriginalMFO
from mealpy.swarm_based.PSO import OriginalPSO

N_TRIALS = 2
LB = [-100, ] * 15
UB = [100, ] * 15
verbose = False
epoch = 10
pop_size = 50

#func_names = ["F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9", "F10", "F11", "F12", "F13", "F14", "F15", "F16", "F17", "F18", "F19"]
# Selecting Funcions which have a large stddev across multiple algorithms
func_names = ["F1", "F2", "F7", "F9", "F10", "F11", "F12", "F13", "F18"]
algo_names = ['OriginalBBO', 'OriginalBMO', 'OriginalEOA']




def find_minimum(input):
    """
    We can run multiple functions at the same time.
    """
    model_name, function_name, trial_number = input


    print(f"Start model: {model_name} || function: {function_name} || trial: {trial_number}")

    problem = {
        "fit_func": getattr(cec2014_nobias, function_name),
        "lb": LB,
        "ub": UB,
        "minmax": "min",
        "log_to": "console",
        "name": function_name
    }
    # model = BaseDE(epoch=epoch, pop_size=pop_size, wf=wf, cr=cr, name=model_name)
    model = get_optimizer_by_name(model_name)(epoch=epoch, pop_size=pop_size, name=model_name, sampling_method="LHS")
    _, best_fitness = model.solve(problem)

    print(f"Finish model: {model_name} || function: {function_name} || trial: {trial_number}")

    return {
        "model_name": model_name,
        "func_name": function_name,
        "trial_number": trial_number,
        "best_fitness": best_fitness
    }


if __name__ == '__main__':
    ## Run model
    best_fit_full = {}
    best_fit_columns = []

    #optimizers: set = get_all_optimizer_names()
    optimizers: dict = {n: get_optimizer_by_name(n) for n in algo_names}

    trial_list = range(1, N_TRIALS + 1)
    func_trials = list(itertools.product(optimizers.keys(), func_names, trial_list))

    with parallel.ProcessPoolExecutor(8) as executor:
        results = executor.map(find_minimum, func_trials)

    result_df = DataFrame(results)

    pivot_df = result_df.pivot(index='trial_number', columns=['model_name', 'func_name'], values='best_fitness')

    mean_df = pivot_df.mean().reset_index().rename(columns={0: 'mean_fitness'})
    mean_alt_df = mean_df.pivot(index='func_name', columns='model_name', values='mean_fitness')
    mean_alt_df.index = mean_alt_df.index.map(lambda x: x[0] + x[1:].zfill(2))
    mean_alt_df = mean_alt_df.sort_index()

    path_best_fit = "history/best_fit/"
    Path(path_best_fit).mkdir(parents=True, exist_ok=True)

    mean_alt_df.to_csv(f"{path_best_fit}/{len(LB)}D_mean_fit.csv", header=True, index=True)
    pivot_df.to_csv(f"{path_best_fit}/{len(LB)}D_best_fit.csv", header=True, index=True)
