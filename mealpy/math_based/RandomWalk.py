import numpy as np
from mealpy.optimizer import Optimizer


class UnbiasedRandomWalk(Optimizer):
    """
    An unbiased random walk optimization algorithm.

    This algorithm performs optimization by randomly moving each solution in each dimension.
    It's a simple stochastic optimization method that can be used as a baseline for comparison
    or for educational purposes.

    References:
    - Pearson, K. (1905). The problem of the random walk. Nature, 72(1867), 342-342.
    - Codling, E. A., Plank, M. J., & Benhamou, S. (2008). Random walk models in biology.
      Journal of the Royal Society Interface, 5(25), 813-834.
    """

    def __init__(self, epoch=10000, pop_size=100, step_size=1.0, **kwargs):
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            step_size (float): size of each step in the random walk, default = 1.0
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.step_size = self.validator.check_float("step_size", step_size, (0, 10.0))
        self.set_parameters(["epoch", "pop_size", "step_size"])
        self.sort_flag = False

    def initialize_variables(self):
        self.pop = self.create_population(self.pop_size)

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_new = []
        for idx in range(0, self.pop_size):
            pos_new = self.pop[idx][self.ID_POS].copy()
            for j in range(0, self.problem.n_dims):
                # Generate a random direction
                direction = np.random.uniform(-1, 1)
                # Move the solution in the generated direction
                pos_new[j] += direction * self.step_size
            pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
            pop_new.append([pos_new, None])
            if self.mode not in self.AVAILABLE_MODES:
                target = self.get_target_wrapper(pos_new)
                self.pop[idx] = self.get_better_solution([pos_new, target], self.pop[idx])
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_wrapper_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new)