"""
𝑒𝑛𝑑𝑢𝑟𝑎𝑛𝑐𝑒(𝑥, 𝑦, 𝑧, 𝑣, 𝑢, 𝑤)
= 𝑒−2∙(𝑦−sin (𝑥))2 + sin(𝑧 ∙ 𝑢) + cos (𝑣 ∙ 𝑤)
"""
import pygad
import math
import numpy as np
def endurance(x, y, z, u, v, w):
 return math.exp(-2*(y-math.sin(x))**2)+math.sin(z*u)+math.cos(v*w)

#𝑨 = [𝟎. 𝟎𝟗, 𝟎. 𝟎𝟔, 𝟎. 𝟗𝟗, 𝟎. 𝟗𝟖, 𝟎. 𝟏, 𝟎. 𝟏𝟓]
def fitness_func(ga_instance, solution, solution_idx):
    x, y, z, u, v, w = solution
    return endurance(x, y, z, u, v, w)

sol_per_pop = 20
num_genes = 6
num_parents_mating = 10
num_generations = 50
mutation_percent_genes = 50
parent_selection_type = "sss"
crossover_type = "single_point"
mutation_type = "random"
gene_space = [0.09, 0.06, 0.99, 0.98, 0.1, 0.15]

ga_instance = pygad.GA(
    gene_space=gene_space,
    num_generations=num_generations,
    num_parents_mating=num_parents_mating,
    fitness_func=fitness_func,
    sol_per_pop=sol_per_pop,
    num_genes=num_genes,
    parent_selection_type=parent_selection_type,
    crossover_type=crossover_type,
    mutation_type=mutation_type,
    mutation_percent_genes=mutation_percent_genes,
)

ga_instance.run()

solution, solution_fitness, solution_idx = ga_instance.best_solution()
print(f"Best endurance: {solution_fitness}")
print(f"Best combination of metals: {solution}")


ga_instance.plot_fitness(title="Generations vs Fit")