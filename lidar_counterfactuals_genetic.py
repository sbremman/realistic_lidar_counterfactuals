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
from utils import ExampleModel, calculate_lidar_readings

# Set a specific seed for reproducibility
SEED = 69
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)

LIDAR_DIM = 180
MAX_DISTANCE = 3.5
MIN_DISTANCE = 0.5

model = ExampleModel()

testing_output_bounds = [[-0.2, 0.7], [0.1, 0.9]]

class LidarCounterfactualsGenetic:
    def __init__(self, ml_model, output_bounds, num_cfs=1, origin=None):
        if origin is None:
            origin = [0.0, 0.0]
        self.ml_model = ml_model
        self.test_output_bounds = output_bounds
        self.lidar_dim = LIDAR_DIM
        self.min_distance = MIN_DISTANCE
        self.max_distance = MAX_DISTANCE

        self.num_cfs = num_cfs
        self.origin = origin

    def _decode_chromosome(self, chromosome):
        raise NotImplemented

    def compute_loss(self, output):
        raise NotImplemented

    def compute_fitness(self, ga_instance, solution, solution_idx):
        # Decode chromosome
        shapes_list = self._decode_chromosome(solution)

        # Make shape_list into shapely multipolygon
        multipolygon = shapely.geometry.MultiPolygon(shapes_list)

        # Generate LiDAR data
        lidar_data = calculate_lidar_readings(multipolygon, self.origin)

        # Input data into DRL agent
        output = self.ml_model(lidar_data)

        # Calculate fitness
        fitness = self._fitness_function(output)

        return fitness

    def generate_counterfactuals(self, solution, solution_idx):






