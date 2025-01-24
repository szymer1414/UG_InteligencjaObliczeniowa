import pygad
import numpy as np

items = [
    (100, 7),   # Zegar
    (300, 7),   # Obraz-pejzaż
    (200, 6),   # Obraz-portret
    (40, 2),    # Radio
    (500, 5),   # Laptop
    (70, 6),    # Lampka nocna
    (100, 1),   # Srebrne sztućce
    (250, 3),   # Porcelana
    (300, 10),  # Figura z brązu
    (280, 3),   # Skórzana torebka
    (300, 15),  # Odkurzacz
]

capacity = 25

def fitness_func(ga_instance, solution, solution_idx):
    total_value = np.sum(np.array([item[0] for item in items]) * solution) 
    total_weight = np.sum(np.array([item[1] for item in items]) * solution) 
    if total_weight > capacity: 
        return -1
    return total_value

sol_per_pop = 20
num_genes = len(items)
num_parents_mating = 10
num_generations = 50
keep_parents = 2
parent_selection_type = "sss"
crossover_type = "single_point"
mutation_type = "random"
mutation_percent_genes = 50  

gene_space = [0, 1]


ga_instance = pygad.GA(
    gene_space=gene_space,
    num_generations=num_generations,
    num_parents_mating=num_parents_mating,
    fitness_func=fitness_func,
    sol_per_pop=sol_per_pop,
    num_genes=num_genes,
    parent_selection_type=parent_selection_type,
    keep_parents=keep_parents,
    crossover_type=crossover_type,
    mutation_type=mutation_type,
    mutation_percent_genes=mutation_percent_genes,
)


ga_instance.run()


solution, solution_fitness, solution_idx = ga_instance.best_solution()
print(f"Selected items: {solution}")
print(f"Fitness value (total value): {solution_fitness}")


values = np.array([item[0] for item in items])  
weights = np.array([item[1] for item in items])  
total_weight = np.sum(solution * weights)
print(f"Total weight of selected items: {total_weight} kg")


selected_items = [i + 1 for i in range(len(solution)) if solution[i] == 1]
print(f"Items selected: {selected_items}")


import time

optimal_value = 1630
count = 0
runs = 10
time_cnt = 0
time_waste = 0
for _ in range(runs):
    start = time.time()
    
    ga_instance = pygad.GA(
        gene_space=[0, 1],
        num_generations=num_generations,
        num_parents_mating=num_parents_mating,
        fitness_func=fitness_func,
        sol_per_pop=sol_per_pop,
        num_genes=num_genes,
        parent_selection_type=parent_selection_type,
        keep_parents=keep_parents,
        crossover_type=crossover_type,
        mutation_type=mutation_type,
        mutation_percent_genes=mutation_percent_genes,
        stop_criteria=["reach_{}".format(optimal_value)]  # Stop when optimal is reached
    )
    
    ga_instance.run()
    solution, solution_fitness, _ = ga_instance.best_solution()
    end = time.time()
    if solution_fitness == optimal_value:
        count += 1
        print("Optimal solution found.")
        time_cnt += end - start
    else:
        time_waste += end - start
    print(f"Run completed in {end - start} seconds.")

time_cnt = time_cnt/count
time = time_waste/(runs - count)
rate = (count / runs) * 100
print(f"Success rate: {rate}%, average time: {time_cnt} seconds., average time wasted: {time} seconds.")