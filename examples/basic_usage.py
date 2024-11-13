import numpy as np
import pandas as pd
import torch
import time
import math
from realistic_lidar_counterfactuals import models, utils
from realistic_lidar_counterfactuals.lidar_counterfactuals_genetic import LidarCounterfactualsGenetic
import os

# Constants and Parameters
SEED = 42  # Random seed for reproducibility
COORDINATE_TYPE = 'cartesian'  # Options: 'polar' or 'cartesian'
NUM_OBJECTS = 3  # Number of objects to generate in counterfactuals
CF_BASE_COMBINATION_TYPE = 'minimum_distance'  # Combination method for counterfactual base
NUM_CFS = 1  # Number of counterfactuals to generate
LIDAR_DIM = 180  # Number of LiDAR readings
LOSS_WEIGHTS = [1.0, 0.0, 0.0, 0.0]  # Loss weights for genetic algorithm
OUTPUT_BOUNDS = np.array([[0.0, 1.0], [0.5, 1.0]])  # Bounds for counterfactual output values

# Cartesian Parameters
MAX_OBJ_SIZE_CARTESIAN = 0.3
MIN_OBJ_SIZE_CARTESIAN = 0.01
MAX_DISTANCE_CARTESIAN = 1.0
MIN_DISTANCE_CARTESIAN = -1.0
MIN_ANGLE = 0.0
MAX_ANGLE = 2 * math.pi

# Polar Parameters
MAX_OBJ_SIZE_POLAR = 0.3
MIN_OBJ_SIZE_POLAR = 0.01
MAX_DISTANCE_POLAR = 1.0
MIN_DISTANCE_POLAR = 0.1
MAX_ANGLE_POLAR = 2 * math.pi

# Paths for loading model and data
BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # Project root directory
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'drl_sac_model_sb3.zip')
DATA_PATH = os.path.join(BASE_DIR, 'data', 'real_world_data_small_obstacle.csv')

# Set seeds for reproducibility
np.random.seed(SEED)
torch.manual_seed(SEED)

# Load model and define a callable model function
model, model_real = models.load_model(MODEL_PATH)
model_func = lambda input: model(torch.Tensor(input)).detach().numpy()

# Load and prepare data from CSV file
df = pd.read_csv(DATA_PATH)
states = df.filter(regex='^state').values
actions = df.filter(regex='^action').values
data_index = 0

base_state = states[data_index]  # Select a base state from dataset
base_action = actions[data_index]  # Corresponding action

# Define genetic algorithm parameters based on the coordinate type
gene_space = [{'low': 0, 'high': 1} for _ in range(6)]

# Set parameter bounds based on chosen coordinate type
if COORDINATE_TYPE == 'cartesian':
    gene_add = np.array([
        1.0,
        MIN_OBJ_SIZE_CARTESIAN,
        MIN_OBJ_SIZE_CARTESIAN,
        MIN_DISTANCE_CARTESIAN,
        MIN_DISTANCE_CARTESIAN,
        MIN_ANGLE
    ])
    gene_max_values = np.array([
        2.0,
        MAX_OBJ_SIZE_CARTESIAN,
        MAX_OBJ_SIZE_CARTESIAN,
        MAX_DISTANCE_CARTESIAN,
        MAX_DISTANCE_CARTESIAN,
        MAX_ANGLE
    ])
elif COORDINATE_TYPE == 'polar':
    gene_add = np.array([
        1.0,
        MIN_OBJ_SIZE_POLAR,
        MIN_OBJ_SIZE_POLAR,
        MIN_DISTANCE_POLAR,
        MIN_ANGLE,
        MIN_ANGLE
    ])
    gene_max_values = np.array([
        2.0,
        MAX_OBJ_SIZE_POLAR,
        MAX_OBJ_SIZE_POLAR,
        MAX_DISTANCE_POLAR,
        MAX_ANGLE_POLAR,
        MAX_ANGLE_POLAR
    ])
else:
    raise ValueError("Coordinate type not recognized. Use 'cartesian' or 'polar'.")

gene_multiply = gene_max_values - gene_add

# Initialize the counterfactual generator
cf_generator = LidarCounterfactualsGenetic(
    ml_model=model_func,  # Model function for prediction
    output_bounds=OUTPUT_BOUNDS,
    num_objects=NUM_OBJECTS,
    gene_space=gene_space,
    gene_add=gene_add,
    gene_multiply=gene_multiply,
    base_state=base_state,
    num_cfs=NUM_CFS,
    loss_weights=LOSS_WEIGHTS,
    coordinate_type=COORDINATE_TYPE,
    cf_base_combination_type=CF_BASE_COMBINATION_TYPE
)

# Generate counterfactuals using the genetic algorithm
start_time = time.time()
solution_list, solution_fitness, solution_idx = cf_generator.generate_counterfactuals()
print(f"Solution: {solution_list}")
print(f"Solution fitness: {solution_fitness}")
print(f"Solution idx: {solution_idx}")
print(f"Time to calculate: {time.time() - start_time:.2f} seconds")

# Plot the original state and corresponding model output
model_output = model_func(base_state)
utils.plot_lidar_state_cos_sin_unnormalize(base_state, title=f"Model action: {model_output}")

# Plot generated counterfactuals and their corresponding outputs
for i, solution in enumerate(solution_list):
    lidar_data = cf_generator.get_lidar_data_from_sol(solution)
    test_data = np.concatenate((cf_generator.cf_combination_func(base_state[:LIDAR_DIM], lidar_data), base_state[LIDAR_DIM:]))
    output = model_func(test_data)
    utils.plot_lidar_state_cos_sin_unnormalize(test_data, title=f"Counterfactual {i+1}, action: {output}")
