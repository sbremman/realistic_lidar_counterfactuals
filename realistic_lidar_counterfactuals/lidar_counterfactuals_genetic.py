import time
import numpy as np
import torch
import math
from shapely.geometry import Point
import pygad
from realistic_lidar_counterfactuals import utils, models

# Constants for LiDAR dimensions and distance bounds
LIDAR_DIM = 180
MAX_DISTANCE = 3.5 / 3.5
MIN_DISTANCE = 0.5 / 3.5


class LidarCounterfactualsGenetic:
    def __init__(self,
                 ml_model,
                 output_bounds,
                 num_objects,
                 gene_space,
                 gene_add,
                 gene_multiply,
                 base_state=None,
                 num_cfs=1,
                 origin=None,
                 combination_type='closest',
                 loss_weights=[1.0, 1.0, 1.0, 1.0],
                 max_tries_per_cf=50,
                 y_loss_weight_increase_if_fail=1.1,
                 y_loss_threshold_completion=-0.1,
                 coordinate_type='cartesian',
                 cf_base_combination_type='minimum_distance'):
        """
        Initialize the counterfactual generator with configurations and parameters.

        Args:
            ml_model: The model function that generates predictions.
            output_bounds (np.ndarray): Bounds for the expected output.
            num_objects (int): Number of objects to be generated in counterfactuals.
            gene_space: Range of each gene for genetic algorithm.
            gene_add, gene_multiply: Offset and scale factors for genes.
            base_state (np.ndarray): Initial state of the LiDAR data.
            num_cfs (int): Number of counterfactuals to generate.
            origin: Origin point for calculations, default is [0.0, 0.0].
            combination_type: Type of combination used ('closest' or other).
            loss_weights (list): Weights for loss components.
            max_tries_per_cf (int): Maximum attempts to generate a counterfactual.
            y_loss_weight_increase_if_fail (float): Factor to increase y-loss if generation fails.
            y_loss_threshold_completion (float): Threshold for y-loss completion.
            coordinate_type (str): Type of coordinates ('cartesian' or 'polar').
            cf_base_combination_type (str): Combination type for counterfactuals ('minimum_distance' or 'cf_priority').
        """
        # Initial configurations
        self.origin = origin or [0.0, 0.0]
        self.base_state = base_state if base_state is not None else self._create_default_base_state()

        # Type validations
        if not isinstance(output_bounds, np.ndarray):
            raise TypeError("output_bounds should be a numpy array (ndarray).")
        if not isinstance(num_objects, int):
            raise TypeError("num_objects should be an integer.")

        # Model and fitness function parameters
        self.ml_model = ml_model
        self.test_output_bounds = output_bounds
        self.lidar_dim = LIDAR_DIM
        self.min_distance = MIN_DISTANCE
        self.max_distance = MAX_DISTANCE

        # Genetic Algorithm Parameters
        self.num_cfs = num_cfs
        self.len_object_params = len(gene_space)
        self.gene_space = gene_space * num_objects
        self.gene_multiply = np.tile(gene_multiply, num_objects)
        self.gene_add = np.tile(gene_add, num_objects)

        # Loss and fitness function parameters
        self.cf_lidar_data = []
        self.loss_weights = np.expand_dims(np.array(loss_weights), axis=1)
        self.max_tries_per_cf = max_tries_per_cf
        self.y_loss_wgt_incr_if_fail = y_loss_weight_increase_if_fail
        self.y_loss_thresh_completion = y_loss_threshold_completion

        # Assign multipolygon_func based on coordinate_type
        if coordinate_type == 'cartesian':
            self.coordinate_type = 'cartesian'
            self.multipolygon_func = utils.genes_to_multipolygon
        elif coordinate_type == 'polar':
            self.coordinate_type = 'polar'
            self.multipolygon_func = utils.genes_to_multipolygon_polar
        else:
            raise ValueError("coordinate_type should be either 'cartesian' or 'polar'.")

        # Assign cf_combination_func based on cf_base_combination_type
        self.cf_combination_func = self._get_combination_function(cf_base_combination_type)
        self.center_check_no_overlap_circle = Point(0, 0).buffer(MIN_DISTANCE)

    def _create_default_base_state(self):
        lidar_base_state = [MAX_DISTANCE] * LIDAR_DIM
        goal_angle_base_state = [0.0]
        goal_dist_base_state = [2.0]
        return np.array(lidar_base_state + goal_angle_base_state + goal_dist_base_state)

    def _validate_coordinate_type(self, coordinate_type):
        if coordinate_type == 'cartesian':
            return 'cartesian'
        elif coordinate_type == 'polar':
            return 'polar'
        else:
            raise ValueError("coordinate_type should be either 'cartesian' or 'polar'.")

    def _get_combination_function(self, cf_base_combination_type):
        if cf_base_combination_type == 'minimum_distance':
            return self._combine_base_and_cf_by_minimum
        elif cf_base_combination_type == 'cf_priority':
            return self._combine_base_and_cf_by_cf_prio
        else:
            raise ValueError("cf_base_combination_type should be either 'minimum_distance' or 'cf_priority'.")

    def _decode_chromosome(self, chromosome):
        return chromosome * self.gene_multiply + self.gene_add

    def compute_loss(self, model_input, model_output):
        y_loss = self._compute_y_loss(model_output)
        proximity_loss = self._compute_proximity_loss(model_input)
        sparsity_loss = self._compute_sparsity_loss(model_output)
        diversity_loss = self._compute_diversity_loss(model_input)
        total_loss = np.array([y_loss, proximity_loss, sparsity_loss, diversity_loss])
        return np.dot(total_loss, self.loss_weights)[0]

    def _compute_y_loss(self, output):
        y_loss = 0.0
        for i in range(len(output)):
            if not self.test_output_bounds[i][0] <= output[i] <= self.test_output_bounds[i][1]:
                y_loss -= min(abs(output[i] - self.test_output_bounds[i][0]),
                              abs(output[i] - self.test_output_bounds[i][1]))
        return y_loss

    def _compute_proximity_loss(self, model_input):
        base_sol_diff = np.abs(model_input - self.base_state)
        proximity_loss = -np.sum(base_sol_diff) / (self.lidar_dim * (self.max_distance - self.min_distance))
        return proximity_loss

    def _compute_sparsity_loss(self, output):
        return 0.0  # Placeholder for sparsity loss

    def _compute_diversity_loss(self, model_input):
        return 0.0  # Placeholder for diversity loss

    def _overlap_with_center_radius(self, multipolygon):
        try:
            return multipolygon.intersects(self.center_check_no_overlap_circle)
        except:
            return True

    def _combine_base_and_cf_by_minimum(self, base_lidar, cf_lidar):
        return np.minimum(base_lidar, cf_lidar)

    def _combine_base_and_cf_by_cf_prio(self, base_lidar, cf_lidar):
        return np.where(np.array(cf_lidar) < MAX_DISTANCE, cf_lidar, base_lidar).tolist()

    def compute_fitness(self, ga_instance, solution, solution_idx):
        decoded_solution = self._decode_chromosome(solution)
        multipolygon = self.multipolygon_func(decoded_solution)
        if self._overlap_with_center_radius(multipolygon):
            return -500.0
        lidar_data = utils.calculate_lidar_readings(multipolygon, self.origin)
        combined_arr = self.cf_combination_func(self.base_state[:LIDAR_DIM], lidar_data)
        model_input = np.concatenate((combined_arr, self.base_state[LIDAR_DIM:]))
        model_output = self.ml_model(model_input)
        return self.compute_loss(model_input, model_output)

    def get_lidar_data_from_sol(self, solution):
        decoded_solution = self._decode_chromosome(solution)
        multipolygon = utils.genes_to_multipolygon(decoded_solution)
        return utils.calculate_lidar_readings(multipolygon, self.origin)

    def calc_y_loss_from_sol(self, solution):
        lidar_data = self.get_lidar_data_from_sol(solution)
        combined_array = self.cf_combination_func(self.base_state[:LIDAR_DIM], lidar_data)
        model_input = np.concatenate((combined_array, self.base_state[LIDAR_DIM:]))
        model_output = self.ml_model(model_input)
        return self._compute_y_loss(model_output)

    def generate_counterfactuals(self):
        solutions, solution_fitnesses, solution_indices = [], [], []
        for curr_cf in range(self.num_cfs):
            cf_generation_terminated = False
            best_solution, best_solution_fitness, best_solution_idx = None, -math.inf, None
            y_loss_best_solution = -math.inf
            attempts = 0
            while not cf_generation_terminated:
                ga_instance = pygad.GA(
                    num_generations=100,
                    num_parents_mating=10,
                    fitness_func=self.compute_fitness,
                    sol_per_pop=100,
                    num_genes=len(self.gene_space),
                    gene_space=self.gene_space,
                    parent_selection_type="tournament",
                    keep_parents=10,
                    crossover_type="single_point",
                    mutation_type="random",
                    mutation_percent_genes=20,
                    stop_criteria=["saturate_10", "reach_0"]
                )

                ga_instance.run()
                solution, solution_fitness, solution_idx = ga_instance.best_solution()

                if solution_fitness > best_solution_fitness:
                    best_solution = solution
                    best_solution_fitness = solution_fitness
                    best_solution_idx = solution_idx
                    y_loss_best_solution = self.calc_y_loss_from_sol(best_solution)

                if y_loss_best_solution >= self.y_loss_thresh_completion or attempts > self.max_tries_per_cf:
                    cf_generation_terminated = True
                    if y_loss_best_solution >= self.y_loss_thresh_completion:
                        print(f"Solution found! y_loss_best_solution: {y_loss_best_solution}, cf_num {curr_cf + 1}")
                    else:
                        print(
                            f"Solution not found with {attempts} attempts. y_loss_best_solution: {y_loss_best_solution}")
                else:
                    print(f"Solution not found, y_loss_best_solution: {y_loss_best_solution}")
                    self.loss_weights[0] *= self.y_loss_wgt_incr_if_fail
                    attempts += 1

            solutions.append(best_solution)
            solution_fitnesses.append(best_solution_fitness)
            solution_indices.append(best_solution_idx)
            sol_lidar_data = self.get_lidar_data_from_sol(best_solution)
            self.cf_lidar_data.append(sol_lidar_data)

        return solutions, solution_fitnesses, solution_indices
