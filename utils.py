import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point, LineString, MultiPolygon
from shapely.ops import unary_union
import torch.nn as nn
import torch
from joblib import Parallel, delayed
from numba import jit
from rtree import index
import time
from shapely.geometry import Point, box
from shapely.affinity import rotate, translate
from shapely.geometry import MultiPolygon

GENE_LENGTH = 6

# Example neural network model (for future reference)
class ExampleModel(nn.Module):
    def __init__(self, input_dim=180, output_dim=1):
        super(ExampleModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 10)
        self.fc4 = nn.Linear(10, output_dim)

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.xavier_uniform_(self.fc4.weight)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.tanh(self.fc4(x))  # Use tanh to allow wider range of output values
        return x

# Original LiDAR calculation function
def calculate_lidar_readings(multipolygon: MultiPolygon, origin: tuple, num_rays: int = 180, max_distance: float = 1.0, debug: bool = False):
    multipolygon = unary_union(multipolygon)
    multipolygon = multipolygon.simplify(0.001, preserve_topology=True)

    origin_point = Point(origin)
    angles = np.linspace(np.pi / 2, 2 * np.pi + np.pi / 2, num_rays, endpoint=False)
    lidar_readings = []

    for angle in angles:
        dx = np.cos(angle)
        dy = np.sin(angle)
        ray = LineString([origin_point, (origin_point.x + max_distance * dx, origin_point.y + max_distance * dy)])
        intersection = multipolygon.intersection(ray)

        if intersection.is_empty:
            lidar_readings.append(max_distance)
            if debug:
                print(f"Ray at angle {np.degrees(angle):.2f}°: No intersection, distance = {max_distance}")
        else:
            if intersection.geom_type == 'Point':
                distance = origin_point.distance(intersection)
            elif intersection.geom_type in ['MultiPoint', 'GeometryCollection']:
                distance = min(origin_point.distance(pt) for pt in intersection.geoms if pt.geom_type == 'Point')
            else:
                distance = origin_point.distance(intersection)

            lidar_readings.append(distance)
            if debug:
                print(f"Ray at angle {np.degrees(angle):.2f}°: Intersection, distance = {distance:.2f}")

    return lidar_readings



# Plotting function
def plot_lidar_readings(lidar_readings, origin=(0, 0), lidar_dim=180):
    angles = np.linspace(np.pi / 2, 2 * np.pi + np.pi / 2, len(lidar_readings), endpoint=False)

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(1)
    ax.scatter(angles, lidar_readings, marker='o', label='LIDAR Readings')
    ax.set_title("LiDAR Readings from Origin ({}, {})".format(origin[0], origin[1]))
    plt.show()

def plot_lidar_state(state, origin=(0, 0), lidar_dim=180, title='None'):
    lidar_readings = state[:lidar_dim]
    angle_to_goal = state[-2]
    distance_to_goal = state[-1]
    angles = np.linspace(np.pi / 2, 2 * np.pi + np.pi / 2, len(lidar_readings), endpoint=False)

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(1)
    ax.scatter(angles, lidar_readings, marker='o', label='LIDAR Readings')
    ax.plot(angle_to_goal, distance_to_goal, 'ro', label='Goal')
    ax.set_title(title)
    plt.show()

def plot_lidar_state_cos_sin(state, origin=(0, 0), lidar_dim=180, title='None', max_range=3.5):
    lidar_readings = state[:lidar_dim]
    cos_angle_to_goal = state[-3]
    sin_angle_to_goal = state[-2]

    assert cos_angle_to_goal <= 1.0
    assert cos_angle_to_goal >= 0.0

    # Reverse the scaling
    sin_angle = 2 * sin_angle_to_goal - 1
    cos_angle = 2 * cos_angle_to_goal - 1

    # Calculate the angle in radians using atan2
    angle_to_goal = np.arctan2(sin_angle, cos_angle)

    distance_to_goal = state[-1]
    angles = np.linspace(np.pi / 2, 2 * np.pi + np.pi / 2, len(lidar_readings), endpoint=False)

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(1)
    ax.scatter(angles, lidar_readings, marker='o', label='LIDAR Readings')
    ax.plot(angle_to_goal, distance_to_goal, 'ro', label='Goal')
    ax.set_title(title)
    #ax.set_ylim([0.0, max_range+0.1])
    plt.show()


def plot_lidar_state_cos_sin_unnormalize(state, origin=(0, 0), lidar_dim=180, title='None', max_range=3.5):
    state = unnormalize_state(state, num_lidar=180)

    lidar_readings = state[:lidar_dim]
    cos_angle = state[-3]
    sin_angle = state[-2]

    assert cos_angle <= 1.0
    assert cos_angle >= 0.0

    # Reverse the scaling
    #sin_angle = 2 * sin_angle_to_goal - 1
    #cos_angle = 2 * cos_angle_to_goal - 1

    # Calculate the angle in radians using atan2
    angle_to_goal = np.arctan2(sin_angle, cos_angle)# + np.pi/2

    distance_to_goal = state[-1]
    angles = np.linspace(0, 2 * np.pi, len(lidar_readings), endpoint=False)

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(1)
    ax.scatter(angles, lidar_readings, marker='o', label='LIDAR Readings')
    ax.plot(angle_to_goal, distance_to_goal, 'ro', label='Goal')
    ax.set_title(title)
    #ax.set_ylim([0.0, max_range+0.1])
    plt.show()

# Shape Conversion Functions
def gene_to_rectangle(gene):
    """
    Converts a gene to a Shapely Rectangle (Polygon) object.
    """
    _, half_x, half_y, pos_x, pos_y, angle = gene
    rect = box(-half_x, -half_y, half_x, half_y)
    rect = rotate(rect, np.degrees(angle), origin=(0, 0), use_radians=False)
    rect = translate(rect, xoff=pos_x, yoff=pos_y)
    return rect


def gene_to_circle(gene):
    """
    Converts a gene to a Shapely Circle (Polygon) object.
    """
    _, radius, _, pos_x, pos_y, _ = gene
    circle = Point(pos_x, pos_y).buffer(radius)
    return circle


def gene_to_shape(gene):
    """
    Converts a gene to the corresponding Shapely shape.
    """
    shape_type = 1 if gene[0] < 1.5 else 2
    gene[0] = shape_type
    if shape_type == 1:
        gene[2] = gene[2]  # half_y for rectangle
        gene[5] = gene[5]  # angle
    else:
        gene[2] = 0.0  # half_y ignored for circle
        gene[5] = 0.0  # angle ignored for circle
    return gene_to_rectangle(gene) if shape_type == 1 else gene_to_circle(gene)


def genes_to_multipolygon(genes):
    shape_list = []

    # Loop through genes, ensuring we take gene_length chunks
    for i in range(0, len(genes), GENE_LENGTH):
        gene = genes[i:i + GENE_LENGTH]  # Segment the gene list
        if len(gene) == GENE_LENGTH:  # Ensure it's a full gene
            gene_shape = gene_to_shape(gene)
            shape_list.append(gene_shape)

    return MultiPolygon(shape_list)

def genes_to_multipolygon_polar(genes):
    shape_list = []

    # Loop through genes, ensuring we take gene_length chunks
    for i in range(0, len(genes), GENE_LENGTH):
        gene = genes[i:i + GENE_LENGTH]  # Segment the gene list
        if len(gene) == GENE_LENGTH:  # Ensure it's a full gene
            gene_shape = gene_to_shape(gene)
            shape_list.append(gene_shape)

    return MultiPolygon(shape_list)

def plot_shapes(shapes):

    fig, ax = plt.subplots(figsize=(8, 8))

    def plot_shape(shape, color, label):
        """Plot a shape, which could be a Polygon or MultiPolygon."""
        if shape.geom_type == 'Polygon':
            x, y = shape.exterior.xy
            ax.fill(x, y, alpha=0.5, fc=color, ec='black', label=label)
        elif shape.geom_type == 'MultiPolygon':
            for poly in shape.geoms:  # Use shape.geoms to iterate over MultiPolygon
                x, y = poly.exterior.xy
                ax.fill(x, y, alpha=0.5, fc=color, ec='black', label=label)

    # Plot target shape
    plot_shape(shapes, 'blue', 'Shape')


    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.legend()
    ax.set_aspect('equal', 'box')
    ax.grid(True)
    plt.show()


def add_column_sin_cos(df):
    # Get the 181st 'state' column ('state_181' which is at index 186 in your list)
    column_181 = df['state_181']

    # Calculate sine and cosine
    sin_values = (np.sin(column_181)+1.0)/2.0
    cos_values = (np.cos(column_181)+1.0)/2.0

    # Add the sin and cos values as new columns, before reordering
    df['state_181_cos'] = cos_values
    df['state_181_sin'] = sin_values

    # Create a new column order based on your instructions
    new_column_order = ['Episode', 'TimeStep', 'Reward', 'Terminated', 'action_1', 'action_2'] + \
                       [f'state_{i}' for i in range(1, 181)] + \
                       ['state_181_cos', 'state_181_sin', 'state_182']

    # Ensure all expected columns exist before reordering
    missing_cols = set(new_column_order) - set(df.columns)
    if missing_cols:
        raise KeyError(f"The following columns are missing and cannot be reordered: {missing_cols}")

    # Reorder the dataframe according to the new order
    df = df[new_column_order]

    # Rename all state columns to follow the pattern state_1 -> state_183
    for i in range(1, 184):
        current_col_name = df.columns[6 + i - 1]  # Start at the 7th column (after Episode, TimeStep, etc.)
        df.rename(columns={current_col_name: f'state_{i}'}, inplace=True)

    return df

def add_column_sin_cos(df):
    # Get the 181st 'state' column ('state_181' which is at index 186 in your list)
    column_181 = df['state_181']

    # Calculate sine and cosine
    sin_values = (np.sin(column_181)+1.0)/2.0
    cos_values = (np.cos(column_181)+1.0)/2.0

    # Add the sin and cos values as new columns, before reordering
    df['state_181_cos'] = cos_values
    df['state_181_sin'] = sin_values

    # Create a new column order based on your instructions
    new_column_order = ['Episode', 'TimeStep', 'Reward', 'Terminated', 'action_1', 'action_2'] + \
                       [f'state_{i}' for i in range(1, 181)] + \
                       ['state_181_cos', 'state_181_sin', 'state_182']

    # Ensure all expected columns exist before reordering
    missing_cols = set(new_column_order) - set(df.columns)
    if missing_cols:
        raise KeyError(f"The following columns are missing and cannot be reordered: {missing_cols}")

    # Reorder the dataframe according to the new order
    df = df[new_column_order]

    # Rename all state columns to follow the pattern state_1 -> state_183
    for i in range(1, 184):
        current_col_name = df.columns[6 + i - 1]  # Start at the 7th column (after Episode, TimeStep, etc.)
        df.rename(columns={current_col_name: f'state_{i}'}, inplace=True)

    return df

def unnormalize_state(state, num_lidar=180):
    lidar_states = state[:num_lidar]
    goal_states = state[num_lidar:]
    cos_state = goal_states[0]
    sin_state = goal_states[1]
    distance_state = goal_states[2]

    unnormal_cos = cos_state * 2.0 - 1.0
    unnormal_sin = sin_state * 2.0 - 1.0
    unnormal_distance = distance_state * 12.0
    unnormal_lidar = lidar_states*3.5

    # Convert scalars to arrays
    unnormal_cos = np.array([unnormal_cos])
    unnormal_sin = np.array([unnormal_sin])
    unnormal_distance = np.array([unnormal_distance])

    return_value = np.concatenate([unnormal_lidar, unnormal_cos, unnormal_sin, unnormal_distance])

    return return_value





if __name__ == "__main__":
    import pandas as pd
    test_df = pd.read_csv('models_and_data/best_val_trained_gazebo/old_data.csv')
    df = pd.read_csv('models_and_data/icinco_model/data.csv')

    df = add_column_sin_cos(df)

    #df.to_csv('models_and_data/icinco_model/cos_sin_data.csv')

    exit()




    # Create a MultiPolygon with 10 obstacles in a polar manner
    obstacles = []
    for _ in range(100):
        angle = np.random.uniform(0, 2 * np.pi)
        distance = np.random.uniform(1.0, 3.5)
        x = distance * np.cos(angle)
        y = distance * np.sin(angle)
        obstacles.append(Point(x, y).buffer(np.random.uniform(0.1, 0.3)))

    multipolygon = MultiPolygon(obstacles)
    origin = (0, 0)
    num_rays = 180

    # Measure performance for each implementation
    lidar_time = time.time()
    lidar_readings = calculate_lidar_readings(multipolygon, origin, debug=False, num_rays=num_rays)
    end_lidar_time = time.time()

    # Plot the LiDAR readings
    plot_lidar_readings(lidar_readings, origin)

    print("Lidar time: ", end_lidar_time - lidar_time)
