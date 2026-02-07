import numpy as np
from mealpy.optimizer import Optimizer


class DichoPSO(Optimizer):
    """
    Dichotomic Particle Swarm Optimization
    Adapted from the Matlab code: http://clerc.maurice.free.fr/pso/SPSO_Dicho.zip
    (Maurice Clerc 2023-03)

    At each iteration:
    1) define two subswarms
    2) keep the "best" one
    3) replace the other by random particles
    There are several options for each step
    """

    def __init__(self, epoch=10000, pop_size=100, w=0.7213, c1=1.193, c2=1.193,
                 dicho=True, dich_dist=0, dich_option=-1, rand_option=1, **kwargs):
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.w = self.validator.check_float("w", w, (0, 1.0))
        self.c1 = self.validator.check_float("c1", c1, (0, 5.0))
        self.c2 = self.validator.check_float("c2", c2, (0, 5.0))
        self.dicho = self.validator.check_bool("dicho", dicho)
        self.dich_dist = self.validator.check_int("dich_dist", dich_dist, [0, 2])
        self.dich_option = self.validator.check_int("dich_option", dich_option, [-1, 2])
        self.rand_option = self.validator.check_int("rand_option", rand_option, [0, 2])
        self.set_parameters(["epoch", "pop_size", "w", "c1", "c2", "dicho", "dich_dist", "dich_option", "rand_option"])
        self.sort_flag = False

    def create_solution(self, lb=None, ub=None):
        position = self.generate_position(lb, ub)
        target = self.get_target_wrapper(position)
        velocity = 0.5 * (self.generate_position(lb, ub) - position)
        local_best_position = position.copy()
        local_best_target = target.copy()
        return [position, target, velocity, local_best_position, local_best_target]

    def amend_position(self, position, lb, ub):
        condition = np.logical_and(lb <= position, position <= ub)
        pos_rand = np.random.uniform(lb, ub)
        return np.where(condition, position, pos_rand)

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        nfe_epoch = 0
        pop_new = []
        for idx in range(0, self.pop_size):
            agent = self.pop[idx].copy()

            # Find the best informant
            if self.pop_size <= 3:
                best_agent = self.g_best
            else:
                idx_list = np.random.choice(range(0, self.pop_size), 3, replace=False)
                best_agent = self.get_best_solution(self.pop[idx_list])

            # Velocity update
            agent[2] = self.w * agent[2] + self.c1 * np.random.rand() * (agent[3] - agent[0]) + \
                       self.c2 * np.random.rand() * (best_agent[0] - agent[0])

            # Position update
            agent[0] = agent[0] + agent[2]

            # Amend position
            agent[0] = self.amend_position(agent[0], self.problem.lb, self.problem.ub)

            # Evaluate the new position
            agent[1] = self.get_target_wrapper(agent[0])
            nfe_epoch += 1

            # Update local best
            if self.compare_agent(agent, [None, agent[4]]):
                agent[3] = agent[0].copy()
                agent[4] = agent[1].copy()
            pop_new.append(agent)

        # Dichotomy
        if self.dicho:
            worst_idx = self.get_global_worst_solution(pop_new)[0]
            pop_new[worst_idx] = self.create_solution(self.problem.lb, self.problem.ub)
            nfe_epoch += 1

        self.pop = pop_new
        self.nfe_per_epoch = nfe_epoch

    def topology(self, pop_size, topo=0, k=3):
        if topo == 0:  # Random topology
            l = np.zeros((pop_size, pop_size))
            for i in range(pop_size):
                idx_list = np.random.choice(range(0, pop_size), k, replace=False)
                l[i, idx_list] = 1
        elif topo == 1:  # Ring topology
            l = np.zeros((pop_size, pop_size))
            for i in range(pop_size):
                l[i, (i - 1) % pop_size] = 1
                l[i, (i + 1) % pop_size] = 1
        return l