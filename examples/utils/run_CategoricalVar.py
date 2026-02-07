#!/usr/bin/env python
# Created by "Thieu" at 09:53, 28/05/2025 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from mealpy import CategoricalVar, Problem


## 1. One variable
bounds = [
    CategoricalVar(valid_sets=("auto", 0.5, 1, None)),
]

## 2. Multiple variables
bounds = [
    CategoricalVar(valid_sets=((0.5, "backward", "forward"),
                          ("0.2", "None", "root"),
                          ("auto", None, "modified"),
                          ("random", 1, "roulette", 0.5, "round-robin")), name="delta")
]

## 3. Multiple variables
bounds = [
    CategoricalVar(valid_sets=(("auto", 2, 3, "backward", "forward", True),
                          (1, 0, 10, "leaf", "branch", "root", False),
                          (0.01, "auto", 0.1, "adaptive", 0.05, "modified"),
                          ("random", 0, 2, 4, "tournament", "roulette", "round-robin")), name="delta"),
    CategoricalVar(valid_sets=(100, 200, 300, 400, 500, 600, 700, 800, 900, 1000), name="epoch")
]


problem = Problem(bounds, obj_func=lambda sol: np.sum(sol**2))
print(f"Problem: {problem}")
print(f"Bounds: {problem.bounds}")

## Generate encoded solution (the real-value solution)
x = problem.generate_solution()
print(x)

x = problem.generate_solution(encoded=False)    # Real world (actual solution - decoded solution) for the problem
x1 = problem.encode_solution(x)                 # Optimizer solution (encoded solution) for the problem
x2 = problem.correct_solution(x1)               # Correct the solution (encoded and bounded solution) for the problem
x3 = problem.decode_solution(x1)                # Real world (actual solution - decoded solution) for the problem
print(f"Real value solution: {x}")
print(f"Encoded solution: {x1}")
print(f"Bounded solution: {x2}")
print(f"Real value solution after decoded: {x3}")
