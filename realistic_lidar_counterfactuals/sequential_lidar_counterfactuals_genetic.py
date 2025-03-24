import time
import numpy as np
import torch
import math
from shapely.geometry import Point
import pygad
from realistic_lidar_counterfactuals import utils, models
from tqdm import tqdm

from realistic_lidar_counterfactuals.pybullet_forward_simulation import Turtlebot3Simulation

# Constants for LiDAR dimensions and distance bounds
LIDAR_DIM = 180
MAX_DISTANCE = 3.5 / 3.5
MIN_DISTANCE = 0.5 / 3.5
GENE_LENGTH = 6

SHAPE_DICT = {}
SHAPE_DICT[1] = "circle"
SHAPE_DICT[2] = "cuboid"


class SequentialLidarCounterfactualsGenetic:
    def __init__(self,
                 ml_model,
                 desired_objective,
                 objective_type,
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
                 cf_base_combination_type='minimum_distance',
                 num_sim_timesteps=1):
        """
        TODO INSERT DESCRIPTION
        :param ml_model:
        :param desired_objective:
        :param objective_type:
        :param num_objects:
        :param gene_space:
        :param gene_add:
        :param gene_multiply:
        :param base_state:
        :param num_cfs:
        :param origin:
        :param combination_type:
        :param loss_weights:
        :param max_tries_per_cf:
        :param y_loss_weight_increase_if_fail:
        :param y_loss_threshold_completion:
        :param coordinate_type:
        :param cf_base_combination_type:
        """
        # Initial configurations
        self.origin = origin or [0.0, 0.0]
        self.base_state = base_state if base_state is not None else self._create_default_base_state()

        # Type validations
        if not isinstance(num_objects, int):
            raise TypeError("num_objects should be an integer.")

        # Check objective type
        if objective_type == 'end_position':
            if not isinstance(desired_objective, np.ndarray):
                raise TypeError(f"for obj_type: {objective_type}, desired_objective should be an array.")
            if not desired_objective.shape == (2,):
                raise ValueError(f"for obj_type: {objective_type}, desired_objective should have shape (2,).")

            self.cost_func = self.c_func_end_pos

        else:
            raise NotImplementedError(f"Objective type {objective_type} is not implemented.")

        self.desired_objective = desired_objective
        self.objective_type = objective_type

        # Model and fitness function parameters
        self.ml_model = ml_model
        # self.lidar_dim = LIDAR_DIM
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


        self.sol_lidar_data_dict = {}

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
        #self.cf_combination_func = self._get_combination_function(cf_base_combination_type)
        self.center_check_no_overlap_circle = Point(0, 0).buffer(MIN_DISTANCE)

        ml_model_func = lambda observation: ml_model(observation)

        self.sim = Turtlebot3Simulation(policy=ml_model_func)
        self.num_sim_timesteps = num_sim_timesteps

    def _create_default_base_state(self):
        """lidar_base_state = [MAX_DISTANCE] * LIDAR_DIM
        goal_angle_base_state = [0.0]
        goal_dist_base_state = [2.0]
        return np.array(lidar_base_state + goal_angle_base_state + goal_dist_base_state)"""

        raise NotImplementedError

    def _validate_coordinate_type(self, coordinate_type):
        if coordinate_type == 'cartesian':
            return 'cartesian'
        elif coordinate_type == 'polar':
            return 'polar'
        else:
            raise ValueError("coordinate_type should be either 'cartesian' or 'polar'.")

    def _decode_chromosome(self, chromosome):

        chromosome = chromosome * self.gene_multiply + self.gene_add

        obstacle_list = []
        for i in range(0, len(chromosome), GENE_LENGTH):
            gene = chromosome[i:i + GENE_LENGTH]
            if len(gene) == GENE_LENGTH:
                shape_type, half_x, half_y, pos_x, pos_y, angle = gene


                shape_type = round(shape_type)
                obstacle_dict = {}
                obstacle_dict['shape'] = SHAPE_DICT[shape_type]
                obstacle_dict['position'] = [pos_x, pos_y, 0.0]
                obstacle_dict['orientation'] = [0, 0, angle]
                obstacle_dict['radius'] = half_x
                obstacle_dict['half_extents'] = [half_x, half_y, 0.5]
                obstacle_dict['height'] = 0.5
                obstacle_list.append(obstacle_dict)


        return obstacle_list

    def compute_loss(self, obs_list, info_list, action_list):
        y_loss = self.cost_func(obs_list, info_list, action_list)
        total_loss = np.array([y_loss])
        return total_loss

    def compute_fitness(self, ga_instance, solution, solution_idx):
        decoded_solution = self._decode_chromosome(solution)

        self.sim.reset_environment()

        self.sim.add_objects(decoded_solution)

        obs_list, info_list, action_list = self.sim.run_simulation(self.num_sim_timesteps)

        return self.compute_loss(obs_list, info_list, action_list)

    def generate_counterfactuals(self):

        solutions, solution_fitnesses, solution_indices = [], [], []
        for curr_cf in range(self.num_cfs):
            cf_generation_terminated = False
            best_solution, best_solution_fitness, best_solution_idx = None, -math.inf, None
            y_loss_best_solution = -math.inf
            attempts = 0
            best_obs_list = None

            while not cf_generation_terminated:
                num_generations = 100
                pbar = tqdm(total=num_generations, desc=f"CF {curr_cf + 1} GA Progress")

                def on_gen_callback(ga_instance):
                    pbar.update(1)

                ga_instance = pygad.GA(
                    num_generations=num_generations,
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
                    stop_criteria=["saturate_10", "reach_0"],
                    on_generation=on_gen_callback
                )

                ga_instance.run()
                pbar.close()

                solution, solution_fitness, solution_idx = ga_instance.best_solution()

                if solution_fitness > best_solution_fitness:
                    best_solution = solution
                    best_solution_fitness = solution_fitness
                    best_solution_idx = solution_idx

                    decoded_solution = self._decode_chromosome(solution)
                    self.sim.reset_environment()
                    self.sim.add_objects(decoded_solution)
                    best_obs_list, best_info_list, best_action_list = self.sim.run_simulation(self.num_sim_timesteps)
                    y_loss_best_solution = self.cost_func(best_obs_list, best_info_list, best_action_list)

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
            sol_lidar_data = best_obs_list[0][:LIDAR_DIM]
            self.cf_lidar_data.append(sol_lidar_data)

            str_best_sol = str(best_solution)
            self.sol_lidar_data_dict[str_best_sol] = sol_lidar_data

        return solutions, solution_fitnesses, solution_indices

    def get_lidar_data_from_sol(self, solution):
        str_sol = str(solution)
        return self.sol_lidar_data_dict[str_sol]


    # Cost functions below

    def c_func_end_pos(self, obs_list, info_list, action_list):

        goal_pos = self.desired_objective
        final_pos = np.array([info_list[-1]['x_pos'], info_list[-1]['y_pos']])

        loss = -np.linalg.norm(goal_pos - final_pos)

        return loss


