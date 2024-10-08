import numpy as np
import random
import yaml
import time
import matplotlib.pyplot as plt
import os

# Ensure the 'Plots' directory exists
os.makedirs('Plots', exist_ok=True)

# Design constraints
import numpy as np
import random
import yaml
import time
import matplotlib.pyplot as plt
import os
import pandas as pd


# Design constraints
def is_balanced_laminate(laminate):
    """
    Checks if the given laminate is balanced, which means it has an equal number of +theta and -theta plies.
    
    Args:
        laminate (list of tuples): A list representing the laminate, where each tuple contains material, angle, and thickness.
    
    Returns:
        bool: True if the laminate is balanced, False otherwise.
    """
    angles = [ply[1] for ply in laminate]
    balanced = all(angles.count(angle) == angles.count(-angle) for angle in set(angles))
    return balanced


# Strength/safety criteria functions
# Function to calculate the margin of safety for Maximum Stress Criterion (MSC)
def maximum_stress_criterion(laminate):
    """
    Evaluates the margin of safety for each lamina based on the Maximum Stress Criterion.
    
    Args:
        laminate (list of tuples): A list representing the laminate, where each tuple contains material, angle, and thickness.
    
    Returns:
    - margin_safety: float, the minimum margin of safety for the given stresses and strengths
    """
    # Load parameters from parameters.yml
    with open('parameters.yml', 'r') as file:
        params = yaml.safe_load(file)

    applied_stresses = params['applied_stresses']
    margin_safeties = []

    for ply in laminate:
        material = ply[0]
        strengths = params['materials'][material]
        
        sigma_1, sigma_2, tau_12 = applied_stresses['sigma_1'], applied_stresses['sigma_2'], applied_stresses['tau_12']
        Xt, Xc, Yt, Yc, S = strengths['Xt'], strengths['Xc'], strengths['Yt'], strengths['Yc'], strengths['S']

        # Calculate margin of safety for tensile and compressive stresses
        margin_sigma_1 = Xt / sigma_1 if sigma_1 > 0 else abs(Xc / sigma_1)
        margin_sigma_2 = Yt / sigma_2 if sigma_2 > 0 else abs(Yc / sigma_2)
        margin_tau_12 = S / abs(tau_12)

        # Minimum margin of safety for MSC
        margin_safeties.append(min(margin_sigma_1, margin_sigma_2, margin_tau_12))
    
    min_margin_safety = min(margin_safeties)
    return min_margin_safety

# Function to calculate the margin of safety for Tsai-Wu Failure Criterion (TWC)
def tsai_wu_criterion(laminate):
    """
    Evaluates the margin of safety for each lamina based on the Tsai-Wu Failure Criterion.
    
    Args:
        laminate (list of tuples): A list representing the laminate, where each tuple contains material, angle, and thickness.
    
    Returns:
    - margin_safety: float, the margin of safety for the given stresses and strengths
    """
    # Load parameters from parameters.yml
    with open('parameters.yml', 'r') as file:
        params = yaml.safe_load(file)

    applied_stresses = params['applied_stresses']
    margin_safeties = []

    for ply in laminate:
        material = ply[0]
        strengths = params['materials'][material]
        
        sigma_1, sigma_2, tau_12 = applied_stresses['sigma_1'], applied_stresses['sigma_2'], applied_stresses['tau_12']
        Xt, Xc, Yt, Yc, S = strengths['Xt'], strengths['Xc'], strengths['Yt'], strengths['Yc'], strengths['S']

        F1 = 1/Xt - 1/Xc
        F2 = 1/Yt - 1/Yc
        F11 = 1/(Xt * Xc)
        F22 = 1/(Yt * Yc)
        F12 = -0.5 * np.sqrt(F11 * F22)
        F66 = 1/S**2

        # Tsai-Wu failure index
        tsai_wu_index = F11 * sigma_1**2 + F22 * sigma_2**2 + 2 * F12 * sigma_1 * sigma_2 + F66 * tau_12**2
        margin_safeties.append(1 / tsai_wu_index)
    
    min_margin_safety = min(margin_safeties)
    return min_margin_safety

# Function to calculate the margin of safety for Distortional Energy Failure Criterion (DEC)
def distortional_energy_criterion(laminate):
    """
    Evaluates the margin of safety for each lamina based on the Distortional Energy Failure Criterion.
    
    Args:
        laminate (list of tuples): A list representing the laminate, where each tuple contains material, angle, and thickness.
    
    Returns:
    - margin_safety: float, the margin of safety for the given stresses and strengths
    """
    # Load parameters from parameters.yml
    with open('parameters.yml', 'r') as file:
        params = yaml.safe_load(file)

    applied_stresses = params['applied_stresses']
    margin_safeties = []

    for ply in laminate:
        material = ply[0]
        strengths = params['materials'][material]
        
        sigma_1, sigma_2, tau_12 = applied_stresses['sigma_1'], applied_stresses['sigma_2'], applied_stresses['tau_12']
        F1, F2, F12 = strengths['F1'], strengths['F2'], strengths['F12']

        # Distortional Energy failure index
        distortional_energy_index = (sigma_1 / F1)**2 + (sigma_2 / F2)**2 + (tau_12 / F12)**2
        margin_safeties.append(1 / distortional_energy_index)
    
    min_margin_safety = min(margin_safeties)
    return min_margin_safety



# Wrapper function to check all safety constraints for a given laminate
def check_safety_criteria(laminate):
    """
    Checks all safety criteria for a given laminate and returns the minimum margin of safety.
    
    Args:
        laminate (list of tuples): A list representing the laminate, where each tuple contains material, angle, and thickness.
    
    Returns:
    - min_margin_safety: float, the minimum margin of safety across all criteria
    """
    msc_margin = maximum_stress_criterion(laminate)
    twc_margin = tsai_wu_criterion(laminate)
    dec_margin = distortional_energy_criterion(laminate)

    # Ensure all safety constraints are satisfied
    min_margin_safety = min(msc_margin, twc_margin, dec_margin)
    return min_margin_safety


# Objective function
def calculate_weight(laminate):
    """
    Calculates the weight of the given laminate.
    
    Args:
        laminate (list of tuples): A list representing the laminate, where each tuple contains material, angle, and thickness.
    
    Returns:
        float: The total weight of the laminate, considering material density and thickness.
    """
    weight = 0

    # Load parameters from YAML file
    with open('parameters.yml', 'r') as file:
        params = yaml.safe_load(file)
    MATERIALS = params['materials']
    G = params['gravity']

    for ply in laminate:
        material, _, thickness = ply
        weight += MATERIALS[material]["density"] * thickness * G
    return weight

def calculate_cost(laminate):
    """
    Calculates the cost of the given laminate.
    
    Args:
        laminate (list of tuples): A list representing the laminate, where each tuple contains material, angle, and thickness.
    
    Returns:
        float: The total cost of the laminate based on material cost factor and thickness.
    """
    cost = 0

    # Load parameters from YAML file
    with open('parameters.yml', 'r') as file:
        params = yaml.safe_load(file)
    MATERIALS = params['materials']

    for ply in laminate:
        material, _, thickness = ply
        fm = MATERIALS[material]["cost_factor"]
        cost += fm * thickness
    return cost

# Fitness function
def fitness_function(laminate, alpha):
    """
    Calculates the fitness score for the given laminate design.
    The fitness is calculated as a weighted combination of cost and weight, penalizing unbalanced laminates and safety violations.
    
    Args:
        laminate (list of tuples): A list representing the laminate, where each tuple contains material, angle, and thickness.
        alpha (float): Weighting factor for cost versus weight in the fitness function.
    
    Returns:
        float: The fitness value, where lower values are better.
    """
    weight = calculate_weight(laminate)
    cost = calculate_cost(laminate)
    
    # Evaluate safety criteria
    min_margin_safety = check_safety_criteria(laminate)

    # Penalize if the laminate does not meet safety constraints
    penalty = 0
    if min_margin_safety < 1.0:  # Minimum acceptable margin of safety
        penalty += 1e6  # Heavy penalty for failing safety criteria
    
    if not is_balanced_laminate(laminate):
        penalty += 1e6  # Penalty for unbalanced laminates

    return alpha * cost + (1 - alpha) * weight + penalty



# Genetic Algorithm Operators
def crossover(parent1, parent2, method):
    """
    Performs crossover between two parent laminates based on the selected crossover method.
    
    Args:
        parent1 (list of tuples): The first parent laminate.
        parent2 (list of tuples): The second parent laminate.
        method (str): The crossover method to use ('single-point', 'multi-point', 'uniform').
    
    Returns:
        tuple: Two new child laminates resulting from the crossover.
    """
    if method == "single-point":
        crossover_point = random.randint(1, min(len(parent1), len(parent2)) - 1)
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]
    elif method == "multi-point":
        points = sorted(random.sample(range(1, min(len(parent1), len(parent2)) - 1), 2))
        child1 = parent1[:points[0]] + parent2[points[0]:points[1]] + parent1[points[1]:]
        child2 = parent2[:points[0]] + parent1[points[0]:points[1]] + parent2[points[1]:]
    elif method == "uniform":
        child1, child2 = [], []
        for p1, p2 in zip(parent1, parent2):
            if random.random() > 0.5:
                child1.append(p1)
                child2.append(p2)
            else:
                child1.append(p2)
                child2.append(p1)
    return child1, child2

def mutate(laminate, mutation_rate):
    """
    Mutates the given laminate by changing the properties of random plies based on the mutation rate.
    
    Args:
        laminate (list of tuples): The laminate to be mutated.
        mutation_rate (float): The probability of mutating each ply.
    
    Returns:
        list of tuples: The mutated laminate.
    """

    # Load parameters from YAML file
    with open('parameters.yml', 'r') as file:
        params = yaml.safe_load(file)
    MATERIALS = params['materials']


    for index in range(len(laminate)):
        if random.random() < mutation_rate:
            material = random.choice(list(MATERIALS.keys()))
            angle = random.choice([0, 15, 30, 45, 60, 75, 90])
            thickness = random.uniform(0.1, 0.5)  # Assuming thickness in meters
            laminate[index] = (material, angle, thickness)
    return laminate

def plyswap(laminate):
    """
    Swaps two random plies in the given laminate.
    
    Args:
        laminate (list of tuples): The laminate in which two plies are to be swapped.
    
    Returns:
        list of tuples: The laminate after swapping two random plies.
    """
    i, j = random.sample(range(len(laminate)), 2)
    laminate[i], laminate[j] = laminate[j], laminate[i]
    return laminate

def plyadd(laminate):
    """
    Adds a new ply to the laminate with random material, angle, and thickness.
    
    Args:
        laminate (list of tuples): The original laminate.
    
    Returns:
        list of tuples: The laminate with the new ply added.
    """
    material = random.choice(list(MATERIALS.keys()))
    angle = random.choice([0, 15, 30, 45, 60, 75, 90])
    thickness = random.uniform(0.1, 0.5)
    laminate.append((material, angle, thickness))
    return laminate

def plydel(laminate):
    """
    Deletes the outermost ply from the given laminate.
    
    Args:
        laminate (list of tuples): The original laminate.
    
    Returns:
        list of tuples: The laminate with the outermost ply removed, if it has more than two plies.
    """
    if len(laminate) > 2:
        laminate.pop()
    return laminate

# Selection Methods
def select_parents(population, fitness_values, method, num_parents):
    """
    Selects parents for the next generation based on the specified selection method.
    
    Args:
        population (list of lists): The current population of laminates.
        fitness_values (list of floats): The fitness values corresponding to each laminate in the population.
        method (str): The selection method to use ('tournament', 'roulette', 'rank').
        num_parents (int): The number of parents to select.
    
    Returns:
        list of lists: The selected parent laminates for the next generation.
    """
    if method == "tournament":
        parents = []
        for _ in range(num_parents):
            tournament = random.sample(list(zip(population, fitness_values)), 3)
            parents.append(min(tournament, key=lambda x: x[1])[0])
        return parents
    elif method == "roulette":
        total_fitness = sum(fitness_values)
        selection_probs = [f / total_fitness for f in fitness_values]
        return random.choices(population, weights=selection_probs, k=num_parents)
    elif method == "rank":
        sorted_population = sorted(zip(population, fitness_values), key=lambda x: x[1])
        ranks = [i for i in range(len(sorted_population))]
        selection_probs = [((2 * len(ranks) - rank) / (2 * len(ranks))) for rank in ranks]
        sorted_population = [x[0] for x in sorted_population]
        return random.choices(sorted_population, weights=selection_probs, k=num_parents)

# Main Genetic Algorithm


def genetic_algorithm():
    """
    Runs the genetic algorithm for optimizing laminate design.
    
    Args:
        population_size (int): The number of laminates in the population.
        generations (int): The number of generations to evolve the population.
        cost_param (float): Weighting factor for cost versus weight in the fitness function.
    
    Returns:
        list of tuples: The best laminate design found after all generations.
    """
    start_time = time.time()

    # Load parameters from YAML file
    with open('parameters.yml', 'r') as file:
        params = yaml.safe_load(file)
    
    # Extract material properties and genetic algorithm parameters
    MATERIALS = params['materials']
    ga_params = params['ga_params']
    min_ply = params['min_ply']
    max_ply = params['max_ply']
    population_size = params['run_time_params']['population_size']
    generations = params['run_time_params']['num_generations']
    cost_param = params['run_time_params']['cost_param']


    # Gravitational acceleration constant
    G = params['gravity']

    # Allowable angles for the plies
    allowable_angles = params['allowable_angles']

    # Allowable ply thicknesses
    ply_thickness_lower_limit = params['ply_thickness_lower_limit']
    ply_thickness_upper_limit = params['ply_thickness_upper_limit']

    # Initialize random population of laminates
    population = [
        [(random.choice(list(MATERIALS.keys())), random.choice(allowable_angles), random.uniform(ply_thickness_lower_limit, ply_thickness_upper_limit))
         for _ in range(random.randint(min_ply, max_ply))]
        for _ in range(population_size)
    ]

    # Mutation probability
    mutation_rate = ga_params['mutation_prob']

    # Lists to track the evolution of weight and cost over generations
    weight_evolution = []
    cost_evolution = []

    for generation in range(generations):

        # Evaluate fitness of the population without multiprocessing
        fitness_values = [fitness_function(laminate, cost_param) for laminate in population]
        
        # Sort population based on fitness values (lower is better)
        sorted_population = [x for _, x in sorted(zip(fitness_values, population))]

        # Selection of elite individuals to carry over to the next generation
        elite_count = int(ga_params["elite_fraction"] * population_size)
        new_population = sorted_population[:elite_count]

        # Select parents for crossover
        parents = select_parents(sorted_population, fitness_values, ga_params["selection_method"], population_size - elite_count)

        # Perform crossover and mutation to generate new population
        while len(new_population) < population_size:
            parent1, parent2 = random.sample(parents, 2)
            if random.random() < ga_params["crossover_prob"]:
                child1, child2 = crossover(parent1, parent2, ga_params["crossover_method"])
                new_population.extend([child1, child2])

            # Adaptive mutation rate adjustment based on population diversity
            if ga_params["adaptive_mutation"] and (generation > 1):
                diversity = len(set(tuple(ind) for ind in population)) / len(population)
                if diversity < ga_params["diversity_threshold"]:
                    mutation_rate += ga_params["mutation_rate_adaptive_factor"]

            # Perform mutation on a random laminate
            if random.random() < mutation_rate:
                laminate = random.choice(new_population)
                new_population.append(mutate(laminate, mutation_rate))

        # Update population for the next generation
        population = new_population[:population_size]

        # Track the best laminate's weight and cost for this generation
        best_laminate = min(population, key=lambda x: fitness_function(x, cost_param))
        weight_evolution.append(calculate_weight(best_laminate))
        cost_evolution.append(calculate_cost(best_laminate))

    # Return the best solution found after all generations
    best_laminate = min(population, key=lambda x: fitness_function(x, cost_param))

    # Print final results
    print("Final Best Laminate Design:", best_laminate)
    print("Final Weight:", calculate_weight(best_laminate))
    print("Final Cost:", calculate_cost(best_laminate))

    # Generate evolution chart for weight and cost evolution on two y-axes
    fig, ax1 = plt.subplots(figsize=(10, 5))

    color = 'tab:blue'
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Weight', color=color)
    ax1.plot(range(generations), weight_evolution, label='Weight', color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:red'
    ax2.set_ylabel('Cost', color=color)  # we already handled the x-label with ax1
    ax2.plot(range(generations), cost_evolution, label='Cost', color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.title('Evolution of Weight and Cost over Generations')
    plt.savefig('Plots/evolution_weight_cost.png')
    plt.close()

    # Plot to show pareto optimal solutions
    final_generation_weights = [calculate_weight(laminate) for laminate in population]
    final_generation_costs = [calculate_cost(laminate) for laminate in population]

    # Identify pareto optimal solutions
    pareto_optimal_indices = []
    for i in range(len(final_generation_weights)):
        dominated = False
        for j in range(len(final_generation_weights)):
            if (final_generation_weights[j] <= final_generation_weights[i] and
                final_generation_costs[j] < final_generation_costs[i]) or \
               (final_generation_weights[j] < final_generation_weights[i] and
                final_generation_costs[j] <= final_generation_costs[i]):
                dominated = True
                break
        if not dominated:
            pareto_optimal_indices.append(i)

    pareto_optimal_weights = [final_generation_weights[i] for i in pareto_optimal_indices]
    pareto_optimal_costs = [final_generation_costs[i] for i in pareto_optimal_indices]

    plt.figure(figsize=(10, 5))
    plt.scatter(final_generation_weights, final_generation_costs, label='Laminates', alpha=0.5)
    plt.scatter(pareto_optimal_weights, pareto_optimal_costs, color='red', label='Pareto Optimal', alpha=0.8)
    plt.plot(pareto_optimal_weights, pareto_optimal_costs, color='red', linestyle='--', label='Pareto Front')
    plt.xlabel('Weight')
    plt.ylabel('Cost')
    plt.title('Pareto Optimal Solutions in Final Generation')
    plt.legend()
    plt.savefig('Plots/pareto_optimal_solutions.png')
    plt.close()

    # Additional Visualizations

    # Plot the distribution of laminate angles in the final population
    def plot_angle_distribution(population):
        angles = [ply[1] for laminate in population for ply in laminate]
        plt.figure(figsize=(10, 5))
        plt.hist(angles, bins=range(-90, 105, 15), edgecolor='black')
        plt.xlabel('Angle (degrees)')
        plt.ylabel('Frequency')
        plt.title('Distribution of Ply Angles in Final Population')
        plt.savefig('Plots/angle_distribution.png')
        plt.close()

    # Plot the distribution of laminate thicknesses in the final population
    def plot_thickness_distribution(population):
        thicknesses = [ply[2] for laminate in population for ply in laminate]
        plt.figure(figsize=(10, 5))
        plt.hist(thicknesses, bins=20, edgecolor='black')
        plt.xlabel('Thickness (meters)')
        plt.ylabel('Frequency')
        plt.title('Distribution of Ply Thicknesses in Final Population')
        plt.savefig('Plots/thickness_distribution.png')
        plt.close()

    # Plot the distribution of laminate materials in the final population
    def plot_material_distribution(population):
        materials = [ply[0] for laminate in population for ply in laminate]
        plt.figure(figsize=(10, 5))
        plt.hist(materials, bins=len(set(materials)), edgecolor='black')
        plt.xlabel('Material')
        plt.ylabel('Frequency')
        plt.title('Distribution of Ply Materials in Final Population')
        plt.savefig('Plots/material_distribution.png')
        plt.close()

    # Call the visualization functions
    plot_angle_distribution(population)
    plot_thickness_distribution(population)
    plot_material_distribution(population)

    end_time = time.time()
    total_time = end_time - start_time
    minutes, seconds = divmod(total_time, 60)
    print(f"Total run time: {int(minutes)} minutes and {seconds:.2f} seconds")

    # Exporting results to Excel
    # Exporting results to Excel
    def export_results_to_excel(best_laminate, weight_evolution, cost_evolution, pareto_optimal_weights, pareto_optimal_costs):
        # Create a Pandas Excel writer using XlsxWriter as the engine
        writer = pd.ExcelWriter('optimization_results.xlsx', engine='xlsxwriter')

        # Convert the best laminate to a DataFrame and write to Excel
        best_laminate_df = pd.DataFrame(best_laminate, columns=['Material', 'Angle', 'Thickness'])
        best_laminate_df.to_excel(writer, sheet_name='Best Laminate', index=False)

        # Convert the weight and cost evolution to a DataFrame and write to Excel
        evolution_df = pd.DataFrame({
            'Generation': range(len(weight_evolution)),
            'Weight': weight_evolution,
            'Cost': cost_evolution
        })
        evolution_df.to_excel(writer, sheet_name='Evolution', index=False)

        # Convert the Pareto optimal solutions to a DataFrame and write to Excel
        pareto_df = pd.DataFrame({
            'Weight': pareto_optimal_weights,
            'Cost': pareto_optimal_costs
        })
        pareto_df.to_excel(writer, sheet_name='Pareto Optimal Solutions', index=False)

        # Save the Excel file
        writer.save()

    # Call the export function
    export_results_to_excel(best_laminate, weight_evolution, cost_evolution, pareto_optimal_weights, pareto_optimal_costs)

# Run the genetic algorithm to optimize laminate design and visualize results
genetic_algorithm()

