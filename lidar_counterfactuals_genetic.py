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
from utils import ExampleModel, calculate_lidar_readings, genes_to_multipolygon, plot_lidar_readings, plot_lidar_state


from models import get_icinco_model


class LidarCounterfactualsGenetic:
    def __init__(self, ml_model, output_bounds, num_objects, base_state=None, num_cfs=1, origin=None):
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

        self.y_loss_type = 'hinge_loss'

        self.center_check_no_overlap_circle = Point(0, 0).buffer(MIN_DISTANCE)

    def _decode_chromosome(self, chromosome):

        return chromosome * self.gene_multiply + self.gene_add

    def compute_loss(self, output):
        #print("Output: ", output)
        return self._compute_y_loss(output) + self._compute_proximity_loss(output) + self._compute_sparsity_loss(output)

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

    def _compute_proximity_loss(self, output):
        proximity_loss = 0.0

        return proximity_loss

    def _compute_sparsity_loss(self, output):
        sparsity_loss = 0.0

        return sparsity_loss

    def _overlap_with_center_radius(self, multipolygon):
        try:
            return multipolygon.intersects(self.center_check_no_overlap_circle)
        except:
            return True

    def compute_fitness(self, ga_instance, solution, solution_idx):
        # Decode chromosome
        decoded_solution = self._decode_chromosome(solution)

        # Turn to shapely multipolygon
        multipolygon = genes_to_multipolygon(decoded_solution)

        # Assert that multipolygon does not overlap the center within a certain radius
        if self._overlap_with_center_radius(multipolygon):
            return -100.0

        # Generate LiDAR data
        lidar_data = calculate_lidar_readings(multipolygon, self.origin)

        # Take the element-wise minimum of the first 180 lidar elements
        min_arr = np.minimum(self.base_state[:LIDAR_DIM], lidar_data)

        # Append the last two elements from arr1
        model_input = np.concatenate((min_arr, self.base_state[LIDAR_DIM:]))
        #model_input = lidar_data

        # Input data into DRL agent
        output = self.ml_model(model_input)

        # Calculate fitness
        fitness = self.compute_loss(output)

        return fitness

    def generate_counterfactuals(self):

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
            mutation_percent_genes=10,  # Lower mutation rate to 8%
            stop_criteria=["saturate_50"]  # Stop if no improvement in 50 generations
        )

        ga_instance.run()

        solution, solution_fitness, solution_idx = ga_instance.best_solution()

        return solution, solution_fitness, solution_idx

if __name__ == "__main__":
    # Set a specific seed for reproducibility
    SEED = 69
    """np.random.seed(SEED)
    torch.manual_seed(SEED)
    random.seed(SEED)"""

    LIDAR_DIM = 180
    MAX_DISTANCE = 3.5
    MIN_DISTANCE = 0.5

    #model = ExampleModel(input_dim=182, output_dim=2)
    model = get_icinco_model()
    model_func = lambda input: model(torch.Tensor(input)).detach().numpy()

    # Get base data sample
    import pandas as pd
    df = pd.read_csv('models_and_data/data.csv')

    # Extract states and actions from the dataset
    state_columns = df.filter(regex='^state')
    states = state_columns.values

    action_columns = df.filter(regex='^action')
    actions = action_columns.values

    data_index = 0

    base_state = states[data_index]
    base_action = actions[data_index]



    """random_numbers = np.random.rand(180)

    test_output = model_func(random_numbers)"""

    # [[low_1, high_1],[low_2, high_2]]
    testing_output_bounds = np.array([[-1.0, 1.0], [-1.0, -0.2]])

    gene_space = [
        {'low': 0, 'high': 1},  # Shape type: 1 or 2
        {'low': 0.0, 'high': 1.0},  # Half extents x 1 + gene*9
        {'low': 0.0, 'high': 1.0},  # Half extents y 1 + gene*9
        {'low': 0.0, 'high': 1.0},  # Position x -10 + gene*20 # TODO Should use polar coordinates
        {'low': 0.0, 'high': 1.0},  # Position y -10 + gene*20
        {'low': 0.0, 'high': 1.0}  # Angle in radians 0.0 + gene*2*pi
    ]

    gene_add = np.array([1.0, 0.1, 0.1, -3.5, -3.5, 0.0])
    gene_multiply = np.array([1.0, 0.2, 0.2, 3.5 * 2, 3.5 * 2, 2 * math.pi])

    num_objects = 5

    cf_generator = LidarCounterfactualsGenetic(model_func, testing_output_bounds, num_objects, base_state=base_state, num_cfs=1, origin=None)

    solution, solution_fitness, solution_idx = cf_generator.generate_counterfactuals()

    print("Solution: ", solution)
    print("Solution fitness: ", solution_fitness)
    print("Solution idx: ", solution_idx)

    decoded_solution = cf_generator._decode_chromosome(solution)

    multipolygon = genes_to_multipolygon(decoded_solution)

    # Assert that multipolygon does not overlap the center within a certain radius

    # Generate LiDAR data
    lidar_data = calculate_lidar_readings(multipolygon, cf_generator.origin)

    # Take the element-wise minimum of the first 180 lidar elements
    min_arr = np.minimum(cf_generator.base_state[:LIDAR_DIM], lidar_data)

    # Append the last two elements from arr1
    test_data = np.concatenate((min_arr, cf_generator.base_state[LIDAR_DIM:]))

    lidar_plus_state = np.concatenate((lidar_data, cf_generator.base_state[LIDAR_DIM:]))
    #test_data = lidar_data

    #plot_lidar_readings(lidar_data)
    output = cf_generator.ml_model(test_data)

    plot_lidar_state(base_state, title=f"Original, action: {base_action}")
    plot_lidar_state(test_data, title=f"CF, action: {output}")
    plot_lidar_state(lidar_plus_state, title=f"CF, action: {output}")



    print("Sol output: ", output)



