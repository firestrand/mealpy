import copy
import time
from dataclasses import field
from typing import Union, List, Optional, NamedTuple, Sequence

import numpy as np
from attr import dataclass

from mealpy import Problem
from mealpy.optimizer import Optimizer
from mealpy.utils.history import History
from mealpy.utils.logger import Logger

@dataclass
class Particle:
    """
    This class mimics the way existing PSO logic access particles through array indexing the properties in a list.
    So in code we can access via particle.position or particle[ID_POS] this enables base optimizer code which uses
    the constants for things like ID_POS on a population entity to work.
    """
    position: np.ndarray
    fitness: Union[float, List[float]]
    velocity: np.ndarray
    best_position: np.ndarray
    best_fitness: Union[float, List[float]]

    def __getitem__(self, index: Union[int, str]):
        if isinstance(index, str):
            return getattr(self, index)
        elif isinstance(index, int):
            return getattr(self, list(self.__class__.__annotations__.keys())[index])
        else:
            raise TypeError("Index must be an integer or a string")

    def __setitem__(self, index: Union[int, str], value):
        if isinstance(index, str):
            setattr(self, index, value)
        elif isinstance(index, int):
            setattr(self, list(self.__class__.__annotations__.keys())[index], value)
        else:
            raise TypeError("Index must be an integer or a string")

    def __len__(self):
        return len(self.__class__.__annotations__)

    def copy(self) -> 'Particle':
        return copy.deepcopy(self)

    @property
    def indices(self):
        return list(self.__class__.__annotations__.keys())

class SPSO2011(Optimizer):
    ID_POS = 0  # Current Position
    ID_TAR = 1  # Latest fitness
    ID_VEC = 2  # Velocity
    ID_LOP = 3  # Local best position
    ID_LOF = 4  # Local best fitness

    def __init__(self, epoch:int=10000, pop_size:int=40, c1:float=1.193147180559945,
                 w:float=0.721347520444482, k:int=3, **kwargs):
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            c1 (float): [0-2] local coefficient, default 0.5 + ln(2)
            c2 (float): [0-2] global coefficient, default 0.5 + ln(2)
            w (float): Weight, default = ω = 1/(2 * ln(2)).
        """
        super().__init__(**kwargs)
        self.links:Union[None,np.ndarray] = None
        self.update_topology:bool = True
        self.epoch:int = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size:int = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.k:int = k # Number of informants
        self.c:float = self.validator.check_float("c", c1, (0, 5.0))
        self.w:float = self.validator.check_float("w", w, (0, 2.0))
        self.p:float = 1 - pow(1 - 1. / self.pop_size, self.k)
        self.set_parameters(["epoch", "pop_size", "c", "w", "k"])

        self.v_max: Union[list, tuple, np.ndarray] = []
        self.v_min: Union[list, tuple, np.ndarray] = []

        self.pop:List[Particle] = []
        self.g_best:Optional[Particle] = None
        self.g_worst:Optional[Particle] = None
        self.problem:Optional[Problem] = None

    def generate_position(self)->np.ndarray:
        return np.random.uniform(low=self.problem.lb, high=self.problem.ub)


    def check_and_set_problem(self, problem: Union[Problem, dict])-> None:
        # sets the problem if a class or passes the dictionary to the problem constructor
        self.problem = problem if isinstance(problem, Problem) else Problem(**problem)
        self.v_max = self.problem.ub # Canonical PSO Bound
        self.v_min = -self.v_max
        # creates the logger using the problem log_to setting
        self.logger = Logger(self.problem.log_to, log_file=self.problem.log_file).create_logger(name=f"{self.__module__}.{self.__class__.__name__}")
        self.logger.info(self.problem.msg)
        self.history = History(log_to=self.problem.log_to, log_file=self.problem.log_file)

    def initialize_pop(self, starting_positions=None):
        if starting_positions is None:
            self.pop = [self.create_solution(self.problem.lb, self.problem.ub) for _ in range(self.pop_size)]
        elif type(starting_positions) in [list, np.ndarray] and len(starting_positions) == self.pop_size:
            if isinstance(starting_positions[0], np.ndarray) and len(starting_positions[0]) == self.problem.n_dims:
                self.pop = [self.create_solution(self.problem.lb, self.problem.ub, pos) for pos in starting_positions]
            else:
                raise ValueError("Starting positions should be a list of positions or 2D matrix of positions only.")
        else:
            raise ValueError("Starting positions should be a list/2D matrix of positions with same length as pop_size hyper-parameter.")

    def solve(self, problem=None, mode='single', starting_positions=None, n_workers=None, termination=None):
        """
        Args:
            problem (Problem, dict): an instance of Problem class or a dictionary

                problem = {
                    "fit_func": your objective function,
                    "lb": list of value
                    "ub": list of value
                    "minmax": "min" or "max"
                    "verbose": True or False
                    "n_dims": int (Optional)
                    "obj_weights": list weights corresponding to all objectives (Optional, default = [1, 1, ...1])
                }

            mode (str): Parallel: 'process', 'thread'; Sequential: 'swarm', 'single'.

                * 'process': The parallel mode with multiple cores run the tasks
                * 'thread': The parallel mode with multiple threads run the tasks
                * 'swarm': The sequential mode that no effect on updating phase of other agents
                * 'single': The sequential mode that effect on updating phase of other agents, default

            starting_positions(list, np.ndarray): List or 2D matrix (numpy array) of starting positions with length equal pop_size parameter
            n_workers (int): The number of workers (cores or threads) to do the tasks (effect only on parallel mode)
            termination (dict, None): The termination dictionary or an instance of Termination class

        Returns:
            list: [position, fitness value]
        """
        self.check_and_set_problem(problem)
        self.check_termination("start", termination, None)

        self.initialize_pop(starting_positions)

        self.after_initialization()

        self.before_main_loop()

        if self.problem.minmax == "min":
            error_previous = np.inf
        else:
            error_previous = -np.inf

        for epoch in range(0, self.epoch):
            time_epoch = time.perf_counter()

            ## Evolve method will be called in child class
            self.evolve(epoch)

            error = self.g_best.fitness[0]
            if self.problem.minmax == "min":
                self.update_topology = not (error < error_previous)
            else:
                self.update_topology = not (error > error_previous)
            error_previous = error

            # Update global best position, the population is sorted or not depended on algorithm's strategy
            pop_temp, self.g_best = self.update_global_best_solution(self.pop)

            time_epoch = time.perf_counter() - time_epoch
            self.track_optimize_step(self.pop, epoch + 1, time_epoch)
            if self.check_termination("end", None, epoch+1):
                break
        self.track_optimize_process()
        return self.solution[self.ID_POS], self.solution[self.ID_TAR][self.ID_FIT]

    def create_solution(self, lb: Union[list, tuple, np.ndarray]=None, ub: Union[list, tuple, np.ndarray]=None, pos: Union[list, tuple, np.ndarray]=None):
        """
        Overriding method in Optimizer class

        Returns:
            list: wrapper of solution with format [position, target, velocity, local_pos, local_fit]
        """
        if pos is None:
            pos = self.generate_position()
        position: Union[list, tuple, np.ndarray] = self.amend_position(pos, lb, ub)
        # get_target_wrapper gets the fitness for the position and multiplies by the weight defined in the problem
        target:Union[float, List[float]] = self.get_target_wrapper(position)
        velocity: Union[list, tuple, np.ndarray] = np.random.uniform(self.v_min, self.v_max)
        local_pos: Union[list, tuple, np.ndarray] = position.copy()
        local_fit:Union[float, List[float]] = target.copy()
        return Particle(position, target, velocity, local_pos, local_fit)

    #TODO: SPSO bound updates the velocity, this does not. This is called by amend_position in create_solution
    def bounded_position(self, position: Union[list, tuple, np.ndarray]=None, lb: Union[list, tuple, np.ndarray]=None, ub: Union[list, tuple, np.ndarray]=None) -> Union[list, tuple, np.ndarray]:
        condition = np.logical_and(lb <= position, position <= ub)
        pos_rand = np.random.uniform(lb, ub)
        return np.where(condition, position, pos_rand)

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        # Check for topology update
        if self.update_topology:
            # Generate random connections only for non-self connections
            random_links = (np.random.random((self.pop_size, self.pop_size)) < self.p)
            # Add self-connections
            self.links = random_links | np.eye(self.pop_size, dtype=bool)
            self.update_topology = False

        local_best_fitness = np.array([particle.best_fitness[0] for particle in self.pop])
        masked_fitness = np.where(self.links == 1, local_best_fitness[:, np.newaxis], np.inf)
        best_informant_indices = np.argmin(masked_fitness, axis=0)

        # Update the velocity and position of each agent
        for idx, agent in enumerate(self.pop):
            # get the best informant best position based on the local best fitness
            best_informant_index = best_informant_indices[idx]
            g_best = self.pop[best_informant_index]
            # Update the velocity
            velocity = self.w * agent.velocity + self.c * np.random.rand() * (agent.best_position - agent.position) + \
                       self.c * np.random.rand() * (g_best.position - agent.position)
            # Update the position
            position = agent.position + velocity
            # Handle boundary conditions
            position = self.amend_position(position, self.problem.lb, self.problem.ub)
            # Evaluate the new position
            agent.position = position
            agent.fitness = self.get_target_wrapper(position)
            # Update the local best position and fitness
            if self.problem.minmax == "min":
                if agent.fitness[0] < agent.best_fitness[0]:
                    agent.best_position = position.copy()
                    agent.best_fitness = agent.fitness.copy()
            else:
                if agent.fitness[0] > agent.best_fitness[0]:
                    agent.best_position = position.copy()
                    agent.best_fitness = agent.fitness.copy()

