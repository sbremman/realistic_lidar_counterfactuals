import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point, LineString, MultiPolygon
from shapely.ops import unary_union
import torch.nn as nn
import torch

class ExampleModel(nn.Module):
    def __init__(self):
        super(ExampleModel, self).__init__()
        self.fc1 = nn.Linear(180, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 10)
        self.fc4 = nn.Linear(10, 2)

        # Initialize weights with a wider distribution
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

def calculate_lidar_readings(multipolygon: MultiPolygon, origin: tuple, num_rays: int = 180, max_distance: float = 3.5, debug: bool = False):
    """
    Calculate LiDAR-like readings from a given origin point and a Shapely MultiPolygon.

    Parameters:
    - multipolygon (MultiPolygon): A Shapely MultiPolygon object representing obstacles.
    - origin (tuple): The origin point (x, y) from which the LiDAR rays are cast.
    - num_rays (int): Number of rays to cast in different directions (default is 360).
    - max_distance (float): Maximum distance for each ray (default is 1000 units).
    - debug (bool): If True, print debug information (default is False).

    Returns:
    - list: A list of distances representing the LiDAR readings for each ray direction.
    """
    # Ensure the MultiPolygon is valid by taking the unary union to merge geometries
    multipolygon = unary_union(multipolygon)
    # Simplify the geometry slightly to avoid complex intersections
    multipolygon = multipolygon.simplify(0.001, preserve_topology=True)

    # Create a Point object for the origin
    origin_point = Point(origin)
    # Generate evenly spaced angles between 0 and 2*pi for the number of rays specified
    angles = np.linspace(np.pi / 2, 2 * np.pi + np.pi / 2, num_rays, endpoint=False)
    # Initialize an empty list to store LiDAR readings
    lidar_readings = []

    # Iterate over each angle to cast a ray
    for angle in angles:
        # Calculate the direction vector for the ray using cosine and sine of the angle
        dx = np.cos(angle)
        dy = np.sin(angle)
        # Create a ray (a line) extending from the origin in the given direction
        # The ray extends 'max_distance' units from the origin
        ray = LineString([origin_point, (origin_point.x + max_distance * dx, origin_point.y + max_distance * dy)])

        # Calculate the intersection points between the ray and the MultiPolygon
        intersection = multipolygon.intersection(ray)

        if intersection.is_empty:
            # If there's no intersection, append the max distance
            lidar_readings.append(max_distance)
            if debug:
                print(f"Ray at angle {np.degrees(angle):.2f}°: No intersection, distance = {max_distance}")
        else:
            # If there is an intersection, find the nearest intersection point
            if intersection.geom_type == 'Point':
                # If the intersection is a single point, calculate the distance from the origin
                distance = origin_point.distance(intersection)
            elif intersection.geom_type in ['MultiPoint', 'GeometryCollection']:
                # If there are multiple intersection points, take the minimum distance from the origin
                distance = min(origin_point.distance(pt) for pt in intersection.geoms if pt.geom_type == 'Point')
            else:
                # For other geometries (e.g., LineString or Polygon), calculate the distance to the closest point
                distance = origin_point.distance(intersection)

            # Append the calculated distance to the list of LiDAR readings
            lidar_readings.append(distance)
            if debug:
                print(f"Ray at angle {np.degrees(angle):.2f}°: Intersection, distance = {distance:.2f}")

    # Return the list of distances representing the LiDAR readings
    return lidar_readings


def plot_lidar_readings(lidar_readings, origin=(0, 0)):
    """
    Plot the LiDAR readings on a polar plot.

    Parameters:
    - lidar_readings (list): A list of distances representing the LiDAR readings for each ray direction.
    - origin (tuple): The origin point (x, y) from which the LiDAR rays are cast.
    """
    angles = np.linspace(np.pi / 2, 2 * np.pi + np.pi / 2, len(lidar_readings), endpoint=False)

    # Create a polar plot
    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)
    ax.plot(angles, lidar_readings, linestyle='-', marker='o')
    ax.set_title("LiDAR Readings from Origin ({}, {})".format(origin[0], origin[1]))
    plt.show()


# Example usage:
# Create a MultiPolygon object with two buffered points representing obstacles
multipolygon = MultiPolygon([Point(1.0, 1.0).buffer(0.5), Point(-1, -1).buffer(0.75), Point(-1, 1).buffer(0.25).envelope])
# Define the origin point from which the LiDAR rays will be cast
origin = (0, 0)
# Calculate LiDAR readings from the origin
lidar_readings = calculate_lidar_readings(multipolygon, origin, debug=True)
# Plot the LiDAR readings
plot_lidar_readings(lidar_readings, origin)