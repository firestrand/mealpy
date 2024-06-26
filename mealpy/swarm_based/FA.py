#!/usr/bin/env python
# Created by "Thieu" at 22:07, 07/04/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class OriginalFA(Optimizer):
    """
    The original version of: Fireworks Algorithm (FA)

    Links:
        1. https://doi.org/10.1007/978-3-642-13495-1_44

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + max_sparks (int): parameter controlling the total number of sparks generated by the pop_size fireworks, default=100
        + p_a (float): percent (const parameter), default=0.04
        + p_b (float): percent (const parameter), default=0.8
        + max_ea (int): maximum explosion amplitude, default=40
        + m_sparks (int): number of sparks generated in each explosion generation, default=100

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.swarm_based.FA import OriginalFA
    >>>
    >>> def fitness_function(solution):
    >>>     return np.sum(solution**2)
    >>>
    >>> problem_dict1 = {
    >>>     "fit_func": fitness_function,
    >>>     "lb": [-10, -15, -4, -2, -8],
    >>>     "ub": [10, 15, 12, 8, 20],
    >>>     "minmax": "min",
    >>> }
    >>>
    >>> epoch = 1000
    >>> pop_size = 50
    >>> max_sparks = 50
    >>> p_a = 0.04
    >>> p_b = 0.8
    >>> max_ea = 40
    >>> m_sparks = 50
    >>> model = OriginalFA(epoch, pop_size, max_sparks, p_a, p_b, max_ea, m_sparks)
    >>> best_position, best_fitness = model.solve(problem_dict1)
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")

    References
    ~~~~~~~~~~
    [1] Tan, Y. and Zhu, Y., 2010, June. Fireworks algorithm for optimization. In International
    conference in swarm intelligence (pp. 355-364). Springer, Berlin, Heidelberg.
    """

    def __init__(self, epoch=10000, pop_size=100, max_sparks=100, p_a=0.04, p_b=0.8, max_ea=40, m_sparks=100, **kwargs):
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            max_sparks (int): parameter controlling the total number of sparks generated by the pop_size fireworks, default=100
            p_a (float): percent (const parameter), default=0.04
            p_b (float): percent (const parameter), default=0.8
            max_ea (int): maximum explosion amplitude, default=40
            m_sparks (int): number of sparks generated in each explosion generation, default=100
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.max_sparks = self.validator.check_int("max_sparks", max_sparks, [2, 10000])
        self.p_a = self.validator.check_float("p_a", p_a, (0, 1.0))
        self.p_b = self.validator.check_float("p_b", p_b, (0, 1.0))
        self.max_ea = self.validator.check_int("max_ea", max_ea, [2, 100])
        self.m_sparks = self.validator.check_int("m_sparks", m_sparks, [2, 10000])
        self.set_parameters(["epoch", "pop_size", "max_sparks", "p_a", "p_b", "max_ea", "m_sparks"])
        self.sort_flag = False

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        fit_list = np.array([agent[self.ID_TAR][self.ID_FIT] for agent in self.pop])
        fit_list = sorted(fit_list)

        pop_new = []
        for idx in range(0, self.pop_size):
            si = self.max_sparks * (fit_list[-1] - self.pop[idx][self.ID_TAR][self.ID_FIT] + self.EPSILON) / \
                 (self.pop_size * fit_list[-1] - np.sum(fit_list) + self.EPSILON)
            Ai = self.max_ea * (self.pop[idx][self.ID_TAR][self.ID_FIT] - fit_list[0] + self.EPSILON) / \
                 (np.sum(fit_list) - fit_list[0] + self.EPSILON)
            if si < self.p_a * self.max_sparks:
                si_ = int(round(self.p_a * self.max_sparks) + 1)
            elif si > self.p_b * self.m_sparks:
                si_ = int(round(self.p_b * self.max_sparks) + 1)
            else:
                si_ = int(round(si) + 1)

            ## Algorithm 1
            pop_new = []
            for j in range(0, si_):
                pos_new = self.pop[idx][self.ID_POS].copy()
                list_idx = np.random.choice(range(0, self.problem.n_dims), round(np.random.uniform() * self.problem.n_dims), replace=False)
                displacement = Ai * np.random.uniform(-1, 1)
                pos_new[list_idx] = pos_new[list_idx] + displacement
                pos_new = np.where(np.logical_or(pos_new < self.problem.lb, pos_new > self.problem.ub),
                                   self.problem.lb + np.abs(pos_new) % (self.problem.ub - self.problem.lb), pos_new)
                pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
                pop_new.append([pos_new, None])
                if self.mode not in self.AVAILABLE_MODES:
                    pop_new[-1][self.ID_TAR] = self.get_target_wrapper(pos_new)
            pop_new = self.update_target_wrapper_population(pop_new)

        for _ in range(0, self.m_sparks):
            idx = np.random.randint(0, self.pop_size)
            pos_new = self.pop[idx][self.ID_POS].copy()
            list_idx = np.random.choice(range(0, self.problem.n_dims), round(np.random.uniform() * self.problem.n_dims), replace=False)
            pos_new[list_idx] = pos_new[list_idx] + np.random.normal(0, 1, len(list_idx))  # Gaussian
            condition = np.logical_or(pos_new < self.problem.lb, pos_new > self.problem.ub)
            pos_true = self.problem.lb + np.abs(pos_new) % (self.problem.ub - self.problem.lb)
            pos_new = np.where(condition, pos_true, pos_new)
            pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
            pop_new.append([pos_new, None])
            if self.mode not in self.AVAILABLE_MODES:
                pop_new[-1][self.ID_TAR] = self.get_target_wrapper(pos_new)
        pop_new = self.update_target_wrapper_population(pop_new)

        ## Update the global best
        self.pop = self.get_sorted_strim_population(pop_new + self.pop, self.pop_size)
