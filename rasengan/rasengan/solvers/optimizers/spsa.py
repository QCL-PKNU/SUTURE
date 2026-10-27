import nevergrad as ng
import csv

from rasengan.utils import iprint
from .abstract_optimizer import Optimizer
from ..options.optimizer_option import CobylaOptimizerOption as OptimizerOption


class SpsaOptimizer(Optimizer):
    def __init__(self, *, max_iter: int = 50, save_address=None, tol=None):
        super().__init__()
        self.optimizer_option: OptimizerOption = OptimizerOption(max_iter=max_iter)
        self.cost_history = []
        self.save_address = save_address
        self.tol = tol

    def minimize(self):
        optimizer_option = self.optimizer_option
        obj_dir = optimizer_option.obj_dir
        cost_func = optimizer_option.cost_func
        cost_func_trans = self.obj_dir_trans(obj_dir, cost_func)

        num_params = optimizer_option.num_params
        budget = optimizer_option.max_iter

        iteration_count = 0

        # Nevergrad SPSA optimizer setup
        optimizer = ng.optimizers.SPSA(parametrization=num_params, budget=budget)

        # Run optimization
        for _ in range(budget):
            x = optimizer.ask()
            value = cost_func_trans(x.value)
            optimizer.tell(x, value)

            # Record cost
            true_cost = cost_func(x.value)
            self.cost_history.append(true_cost)

            iteration_count += 1
            if iteration_count % 10 == 0:
                iprint(f"iteration {iteration_count}, result: {true_cost}")

        recommendation = optimizer.provide_recommendation()
        if self.save_address:
            self.save_cost_history_to_csv()

        return recommendation.value, iteration_count

    def save_cost_history_to_csv(self):
        filename = self.save_address + ".csv"
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Iteration', 'Cost'])
            for i, cost in enumerate(self.cost_history):
                writer.writerow([i + 1, cost])
