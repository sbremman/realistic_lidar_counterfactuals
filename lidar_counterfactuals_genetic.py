import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from shapely.geometry import box, Point
import shapely.affinity
import pygad
import random
import math
from tqdm import tqdm
import utils


import models


class LidarCounterfactualsGenetic:
    def __init__(self,
                 ml_model,
                 output_bounds,
                 num_objects,
                 base_state=None,
                 num_cfs=1,
                 origin=None,
                 combination_type='closest',
                 loss_weights=[1.0, 1.0, 1.0, 1.0],
                 max_tries_per_cf=5,
                 y_loss_weight_increase_if_fail=1.1,
                 y_loss_threshold_completion=-1.0,
                 coordinate_type='cartesian',
                 cf_base_combination_type='minimum_distance'):
        # Combination type is closest or generated_obstacle.
        # Closest takes the closest lidar points between the map resulting from placing and obstacle, and the original map
        # generated_obstacle takes the generated obstacle for all lidar points that is covered by the generated obstacle.
        # In theory generated_obstacle is can make more different explanations
        # Closest is most similar to original data

        if origin is None:
            origin = [0.0, 0.0]
        if base_state is None:
            lidar_base_state = [MAX_DISTANCE]*180
            goal_angle_base_state = [0.0]
            goal_dist_base_state = [2.0]
            base_state = np.array(lidar_base_state+goal_angle_base_state+goal_dist_base_state)

        if not isinstance(output_bounds, np.ndarray):
            raise TypeError("output_bounds should be a numpy array (ndarray).")
        if not isinstance(num_objects, int):
            raise TypeError("num_objects should be an integer.")
        # ml_model should be a function that takes in lidar input and outputs an output
        # TODO Maybe we should somehow include so that if lidar is just part of the input together with goal info
        self.ml_model = ml_model
        self.test_output_bounds = output_bounds
        self.lidar_dim = LIDAR_DIM
        self.min_distance = MIN_DISTANCE
        self.max_distance = MAX_DISTANCE

        self.num_cfs = num_cfs
        self.origin = origin
        self.len_object_params = len(gene_space)
        self.gene_space = gene_space * num_objects
        self.gene_multiply = np.tile(gene_multiply, num_objects)
        self.gene_add = np.tile(gene_add, num_objects)

        self.base_state = base_state
        self.cf_lidar_data = []

        self.y_loss_type = 'hinge_loss'

        self.center_check_no_overlap_circle = Point(0, 0).buffer(MIN_DISTANCE)

        self.loss_weights = np.expand_dims(np.array(loss_weights), axis=1)

        self.max_tries_per_cf = max_tries_per_cf
        self.y_loss_wgt_incr_if_fail = y_loss_weight_increase_if_fail
        self.y_loss_thresh_completion = y_loss_threshold_completion

        if coordinate_type == 'cartesian':
            self.coordinate_type = 'cartesian'
            self.multipolygon_func = utils.genes_to_multipolygon

        elif coordinate_type == 'polar':
            self.coordinate_type = 'polar'
            self.multipolygon_func = utils.genes_to_multipolygon_polar

        else:
            raise ValueError("coordinate_type should be either 'cartesian' or 'polar'.")

        if cf_base_combination_type == 'minimum_distance':
            self.cf_combination_func = self._combine_base_and_cf_by_minimum

        elif cf_base_combination_type == 'cf_priority':
            self.cf_combination_func = self._combine_base_and_cf_by_cf_prio

        else:
            raise ValueError("cf_base_combination_type should be either 'minimum_distance' or 'cf_priority'.")


    def _decode_chromosome(self, chromosome):

        return chromosome * self.gene_multiply + self.gene_add

    def compute_loss(self, model_input, model_output):
        #print("Output: ", output)
        y_loss = self._compute_y_loss(model_output)
        proximity_loss = self._compute_proximity_loss(model_input, model_output)
        sparsity_loss = self._compute_sparsity_loss(model_output)
        diversity_loss = self._compute_diversity_loss(model_input, model_output)
        total_loss = np.array([y_loss, proximity_loss, sparsity_loss, diversity_loss])
        weighted_loss = np.dot(total_loss, self.loss_weights)[0]
        return weighted_loss

    def _compute_y_loss(self, output):
        y_loss = 0.0

        # TODO for now assuming only one counterfactual is generated.
        if self.y_loss_type == 'hinge_loss':
            # Assuming output is 1-dim
            assert output.shape[0] == self.test_output_bounds.shape[0]
            for i in range(len(output)):
                if not self.test_output_bounds[i][0] <= output[i] <= self.test_output_bounds[i][1]:
                    y_loss -= min(abs(output[i] - self.test_output_bounds[i][0]),
                                  abs(output[i] - self.test_output_bounds[i][1]))

        return y_loss

    def _compute_proximity_loss(self, model_input, model_output):

        base_sol_diff = np.abs(model_input - self.base_state)
        proximity_loss = -np.sum(base_sol_diff)

        # Normalize proximity loss..
        proximity_loss /= (self.lidar_dim*(self.max_distance-self.min_distance))

        return proximity_loss

    def _compute_sparsity_loss(self, output):
        sparsity_loss = 0.0

        return sparsity_loss

    def _compute_diversity_loss(self, model_input, model_output):
        diversity_loss = 0.0

        """input_lidar_data = model_input[:LIDAR_DIM]

        for generated_cf_lidar_data in self.cf_lidar_data:
            


        normalized_diversity_loss = diversity_loss / (len(self.cf_lidar_data)*self.lidar_dim*(self.max_distance-self.min_distance))"""
        return diversity_loss

    def _overlap_with_center_radius(self, multipolygon):
        try:
            return multipolygon.intersects(self.center_check_no_overlap_circle)
        except:
            return True

    def _combine_base_and_cf_by_minimum(self, base_lidar, cf_lidar):

        min_arr = np.minimum(base_lidar, cf_lidar)

        return min_arr

    def _combine_base_and_cf_by_cf_prio(self, base_lidar, cf_lidar):
        cf_lidar = np.array(cf_lidar)

        combined_lidar = np.where(cf_lidar < 1.0, cf_lidar, base_lidar).tolist()

        return combined_lidar

    def compute_fitness(self, ga_instance, solution, solution_idx):
        # Decode chromosome
        decoded_solution = self._decode_chromosome(solution)

        # Turn to shapely multipolygon
        multipolygon = self.multipolygon_func(decoded_solution)

        # Assert that multipolygon does not overlap the center within a certain radius
        if self._overlap_with_center_radius(multipolygon):
            return -100.0

        # Generate LiDAR data
        lidar_data = utils.calculate_lidar_readings(multipolygon, self.origin)

        combined_arr = self.cf_combination_func(self.base_state[:LIDAR_DIM], lidar_data)

        # Append the last two elements from arr1
        model_input = np.concatenate((combined_arr, self.base_state[LIDAR_DIM:]))

        # Input data into DRL agent
        model_output = self.ml_model(model_input)

        # Calculate fitness
        fitness = self.compute_loss(model_input, model_output)

        return fitness

    def get_lidar_data_from_sol(self, solution):
        decoded_solution = self._decode_chromosome(solution)

        # Turn to shapely multipolygon
        multipolygon = utils.genes_to_multipolygon(decoded_solution)

        # Generate LiDAR data
        lidar_data = utils.calculate_lidar_readings(multipolygon, self.origin)

        return lidar_data

    def calc_y_loss_from_sol(self, solution):

        lidar_data = self.get_lidar_data_from_sol(solution)

        combined_array = self.cf_combination_func(self.base_state[:LIDAR_DIM], lidar_data)

        # Append the last two elements from arr1
        model_input = np.concatenate((combined_array, self.base_state[LIDAR_DIM:]))
        # model_input = lidar_data

        # Input data into DRL agent
        model_output = self.ml_model(model_input)

        return self._compute_y_loss(model_output)

    def generate_counterfactuals(self):

        solutions, solution_fitnesses, solution_indices = [], [], []
        for curr_cf in range(self.num_cfs):
            cf_generation_terminated = False
            best_solution, best_solution_fitness, best_solution_idx = None, -math.inf, None
            attempts = 0
            while not cf_generation_terminated:
                ga_instance = pygad.GA(
                    num_generations=100,  # Increase number of generations
                    num_parents_mating=10,  # More parents mating
                    fitness_func=self.compute_fitness,  # Fitness function
                    sol_per_pop=100,  # Larger population size
                    num_genes=len(self.gene_space),  # Number of genes
                    gene_space=self.gene_space,  # Gene space
                    parent_selection_type="rws",  # Tournament selection for better diversity
                    keep_parents=10,  # Keep top 10 parents
                    crossover_type="uniform",  # Uniform crossover for better exploration
                    mutation_type="random",  # Keep random mutation
                    mutation_percent_genes=20,  # Lower mutation rate to 8%
                    stop_criteria=["saturate_50"]  # Stop if no improvement in 50 generations
                )

                ga_instance.run()

                solution, solution_fitness, solution_idx = ga_instance.best_solution()

                if solution_fitness > best_solution_fitness:
                    best_solution = solution
                    best_solution_fitness = solution_fitness
                    best_solution_idx = solution_idx

                y_loss_curr_sol = self.calc_y_loss_from_sol(solution)

                if y_loss_curr_sol >= self.y_loss_thresh_completion or attempts > self.max_tries_per_cf:
                    cf_generation_terminated = True
                    print(f"Solution found! y_loss_curr_sol: {y_loss_curr_sol}")

                else:
                    print(f"Solution not found, y_loss_curr_sol: {y_loss_curr_sol}")
                    loss_weights[0] *= self.y_loss_wgt_incr_if_fail
                    attempts += 1

            solutions.append(best_solution)
            solution_fitnesses.append(best_solution_fitness)
            solution_indices.append(best_solution_idx)

            sol_lidar_data = self.get_lidar_data_from_sol(best_solution)

            self.cf_lidar_data.append(sol_lidar_data)



        return solutions, solution_fitnesses, solution_indices

if __name__ == "__main__":
    # Set a specific seed for reproducibility
    """SEED = 69
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    random.seed(SEED)"""

    LIDAR_DIM = 180
    MAX_DISTANCE = 3.5/3.5
    MIN_DISTANCE = 0.75/3.5

    coordinate_type = 'cartesian' #polar or cartesian
    num_objects = 3
    cf_base_combination_type ='cf_priority'

    #model = ExampleModel(input_dim=182, output_dim=2)
    #model = models.get_best_val_model()
    #val_model_func = lambda input: val_model(torch.Tensor(input)).detach().numpy()

    #model = models.load_model('models_and_data/ecc_gazebo_trained/rl_model_50000_steps.zip')
    model, model_real = models.load_model('models_and_data/ecc_gazebo_trained/ECC_2025_models/ECC_2025_hp_from_gz/20241022_144549/rl_model_900000_steps.zip')
    # 20241022_144549/rl_model_900000_steps.zip
    model_func = lambda input: model(torch.Tensor(input)).detach().numpy()
    #test_model_func = lambda input: model_real.predict(torch.Tensor(input), deterministic=True)[0]

    # Get base data sample
    import pandas as pd
    #df = pd.read_csv('models_and_data/icinco_model/cos_sin_data.csv')
    df = pd.read_csv('models_and_data/ecc_gazebo_trained/ECC_2025_models/ECC_2025_hp_from_gz/20241022_144549/df_20241025104346_c576e28d_determinstic.csv')
    #df = pd.read_csv('models_and_data/best_val_trained_gazebo/old_data.csv')

    # Extract states and actions from the dataset
    state_columns = df.filter(regex='^state')
    states = state_columns.values

    action_columns = df.filter(regex='^action')
    actions = action_columns.values

    data_index = 100

    base_state = states[data_index]
    base_action = actions[data_index]
    #val_test_action = val_model_func(base_state)
    test_action = model_func(base_state)
    #test_action_2 = test_model_func(base_state)



    random_numbers = np.random.rand(183)

    test_output = model_func(random_numbers)

    # [[low_1, high_1],[low_2, high_2]]
    testing_output_bounds = np.array([[-1.0, 1.0], [0.5, 1.0]])
    # y_loss, proximity_loss, sparsity_loss, diversity_loss
    loss_weights = [1.0, 0.1, 1.0, 1.0]
    #exit()

    """gene_add = np.array([1.0, 0.1, 0.1, -3.5, -3.5, 0.0])
        gene_multiply = np.array([1.0, 0.2, 0.2, 3.5 * 2, 3.5 * 2, 2 * math.pi])"""

    gene_space = [
        {'low': 0, 'high': 1},
        {'low': 0.0, 'high': 1.0},
        {'low': 0.0, 'high': 1.0},
        {'low': 0.0, 'high': 1.0},
        {'low': 0.0, 'high': 1.0},
        {'low': 0.0, 'high': 1.0}
    ]

    if coordinate_type == 'cartesian':
        max_obj_size = 0.3
        min_obj_size = 0.01
        max_obj_x_pos = 0.2
        min_obj_x_pos = -0.2
        max_obj_y_pos = 0.5
        min_obj_y_pos = -0.0
        min_angle = 0.0
        max_angle = 2*math.pi

        gene_add = np.array([1.0, min_obj_size, min_obj_size, min_obj_x_pos, min_obj_y_pos, min_angle])
        gene_max_values = np.array([2.0, max_obj_size, max_obj_size, max_obj_x_pos, max_obj_y_pos, max_angle])
        gene_multiply = gene_max_values - gene_add

    elif coordinate_type == 'polar':
        max_obj_size = 0.3
        min_obj_size = 0.01
        max_obj_polar_radius = MAX_DISTANCE
        min_obj_polar_radius = MIN_DISTANCE
        max_obj_polar_angle = 2 * math.pi
        min_obj_polar_angle = 0.0
        max_angle = 2 * math.pi
        min_angle = 0.0

        gene_add = np.array([1.0, min_obj_size, min_obj_size, min_obj_polar_radius, min_obj_polar_angle, min_angle])
        gene_max_values = np.array([2.0, max_obj_size, max_obj_size, max_obj_polar_radius, max_obj_polar_angle, max_angle])
        gene_multiply = gene_max_values - gene_add

    else:
        raise ValueError('Coordinate type not recognized')



    start_time = time.time()
    cf_generator = LidarCounterfactualsGenetic(model_func,
                                               testing_output_bounds,
                                               num_objects,
                                               base_state=base_state,
                                               num_cfs=1,
                                               origin=None,
                                               loss_weights=loss_weights,
                                               coordinate_type=coordinate_type,
                                               cf_base_combination_type=cf_base_combination_type)

    solution, solution_fitness, solution_idx = cf_generator.generate_counterfactuals()


    print("Solution: ", solution)
    print("Solution fitness: ", solution_fitness)
    print("Solution idx: ", solution_idx)
    print("time to calculate: ", time.time() - start_time)

    decoded_solution = cf_generator._decode_chromosome(solution)

    multipolygon = cf_generator.multipolygon_func(decoded_solution)

    # Assert that multipolygon does not overlap the center within a certain radius

    # Generate LiDAR data
    lidar_data = utils.calculate_lidar_readings(multipolygon, cf_generator.origin)

    min_arr = cf_generator.cf_combination_func(cf_generator.base_state[:LIDAR_DIM], lidar_data)

    # Append the last two elements from arr1
    test_data = np.concatenate((min_arr, cf_generator.base_state[LIDAR_DIM:]))

    lidar_plus_state = np.concatenate((lidar_data, cf_generator.base_state[LIDAR_DIM:]))
    #test_data = lidar_data

    #plot_lidar_readings(lidar_data)
    output = cf_generator.ml_model(test_data)

    model_output = model_func(base_state)
    utils.plot_lidar_state_cos_sin_unnormalize(base_state, title=f"From dataset, action: {base_action}")
    utils.plot_lidar_state_cos_sin_unnormalize(base_state, title=f"From model, action: {model_output}")
    utils.plot_lidar_state_cos_sin_unnormalize(test_data, title=f"CF, action: {output}, goal: [{testing_output_bounds[0]}, {testing_output_bounds[1]}]")
    utils.plot_lidar_state_cos_sin_unnormalize(lidar_plus_state, title=f"goal: [{testing_output_bounds[0]}, {testing_output_bounds[1]}]")


    print("Sol output: ", output)



