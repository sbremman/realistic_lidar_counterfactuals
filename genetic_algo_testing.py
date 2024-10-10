import time

import pygad
import numpy as np
from shapely.geometry import Point, box
from shapely.affinity import rotate, translate
import matplotlib.pyplot as plt
from shapely.geometry import MultiPolygon

from utils import genes_to_multipolygon



# Fitness Function
def calculate_overlapping_area(shape1, shape2):
    """
    Calculates the overlapping area between two Shapely shapes.
    """
    intersection = shape1.intersection(shape2)
    return intersection.area

def decode_gene(genes):

    new_genes = gene_A+gene_B*genes

    """for i in range(0, len(genes), gene_length):
        genes[i:i + gene_length] = gene_A + gene_B*genes[i:i + gene_length]  # Segment the gene list"""

    return new_genes


def fitness_func(ga_instance, solution, solution_idx):
    """
    Fitness function for PyGAD. Calculates the IoU between the target shape and the evolved shape.
    """
    gene = solution.copy()
    gene = decode_gene(gene)

    # Convert genes to MultiPolygon
    try:
        evolved_shape = genes_to_multipolygon(gene)
    except Exception:
        return 0  # Penalize invalid shapes

    # Calculate intersection and union areas
    try:
        intersection_area = calculate_overlapping_area(target_shape, evolved_shape)
        union_area = target_shape.union(evolved_shape).area
    except Exception:
        return 0

    # Avoid division by zero
    if union_area == 0:
        return 0

    # Calculate IoU
    iou = intersection_area / union_area


    return iou


def plot_shapes(target, evolved):

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
    plot_shape(target, 'blue', 'Target Shape')

    # Plot evolved shape
    plot_shape(evolved, 'red', 'Evolved Shape')

    # Plot overlapping area
    intersection = target.intersection(evolved)
    if not intersection.is_empty:
        plot_shape(intersection, 'purple', 'Overlap Area')

    ax.set_title('Target vs Evolved Shape with Overlap')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.legend()
    ax.set_aspect('equal', 'box')
    ax.grid(True)
    plt.show()

 if __name__ == "__main__":

    # Define the target shape gene and shape
    # Example: Rectangle with half-width=3, half-height=2, positioned at (5,5), rotated by 30 degrees

    target_gene_1 = [1, 3.0, 2.0, 5.0, 5.0, np.radians(30)]
    gene_length = len(target_gene_1)
    target_gene_2 = [2, 2.3, 2.0, 1.0, -2.0, np.radians(30)]
    target_gene_3 = [2, 3.3, 3.0, -3.0, 3.0, np.radians(30)]
    target_gene_4 = [1, 1.0, 2.5, -6.0, -6.0, np.radians(1.0)]

    genes = [target_gene_1, target_gene_2, target_gene_3, target_gene_4]

    target_genes = []
    for gene in genes:
        target_genes.extend(gene)
    target_shape = genes_to_multipolygon(target_genes)

    """test_gene = [1, 2.0, 2.0, 5.0, 5.0, np.radians(60)]
    test_shape = gene_to_shape(test_gene)
    
    print(fitness_func(None, test_gene, 0))
    
    plot_shapes(target_shape, test_shape)
    exit()"""

    # Define gene space
    """gene_space = [
        {'low': 1, 'high': 2},  # Shape type: 1 or 2
        {'low': 1.0, 'high': 10.0},  # Half extents x
        {'low': 1.0, 'high': 10.0},  # Half extents y
        {'low': -10.0, 'high': 10.0},  # Position x
        {'low': -10.0, 'high': 10.0},  # Position y
        {'low': 0.0, 'high': 2 * np.pi}  # Angle in radians
    ]*(len(genes))
    
    import math
    gene_A = np.array([0, 0.0, 0.0, 0.0, 0.0, 0.0])
    gene_B = np.array([1, 1.0, 1.0, 1.0, 1.0, 1.0])"""

    gene_space = [
        {'low': 0, 'high': 1},  # Shape type: 1 or 2
        {'low': 0.0, 'high': 1.0},  # Half extents x 1 + gene*9
        {'low': 0.0, 'high': 1.0},  # Half extents y 1 + gene*9
        {'low': 0.0, 'high': 1.0},  # Position x -10 + gene*20
        {'low': 0.0, 'high': 1.0},  # Position y -10 + gene*20
        {'low': 0.0, 'high': 1.0}  # Angle in radians 0.0 + gene*2*pi
    ]*(len(genes))

    import math
    gene_A = np.array([1.0, 1.0, 1.0, -10.0, -10.0, 0.0]*len(genes))
    gene_B = np.array([1.0, 9.0, 9.0, 20.0, 20.0, 2*math.pi]*len(genes))



    # Initialize GA instance
    ga_instance = pygad.GA(
        num_generations=100,  # Increase number of generations
        num_parents_mating=10,  # More parents mating
        fitness_func=fitness_func,  # Fitness function
        sol_per_pop=100,  # Larger population size
        num_genes=len(gene_space),  # Number of genes
        gene_space=gene_space,  # Gene space
        parent_selection_type="rws",  # Tournament selection for better diversity
        keep_parents=10,  # Keep top 10 parents
        crossover_type="uniform",  # Uniform crossover for better exploration
        mutation_type="random",  # Keep random mutation
        mutation_percent_genes=8,  # Lower mutation rate to 8%
        stop_criteria=["saturate_50"]  # Stop if no improvement in 50 generations
    )

    # Run GA
    ga_instance.run()

    # Fetch the best solution
    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    print("Best Solution Gene:", solution)
    print("Best Fitness (IoU):", solution_fitness)

    # Decode and convert to shape
    best_gene = solution.copy()
    best_gene = decode_gene(best_gene)

    shape_type = 1 if best_gene[0] < 1.5 else 2
    best_gene[0] = shape_type
    if shape_type == 1:
        best_gene[2] = best_gene[2]  # half_y for rectangle
        best_gene[5] = best_gene[5]  # angle
    else:
        best_gene[2] = 0.0  # half_y ignored for circle
        best_gene[5] = 0.0  # angle ignored for circle
    best_shape = genes_to_multipolygon(best_gene)


    # Visualization

    ga_instance.plot_fitness()


    plot_shapes(target_shape, best_shape)
