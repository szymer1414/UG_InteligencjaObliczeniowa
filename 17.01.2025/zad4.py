import pyswarms as ps
import numpy as np
import math
from pyswarms.utils.plotters import plot_cost_history
import matplotlib.pyplot as plt


def endurance(x, y, z, u, v, w):
    return math.exp(-2 * (y - math.sin(x))**2) + math.sin(z * u) + math.cos(v * w)
def swarm_fitness(X):
    fitness_values = []
    for particle in X:
        x, y, z, u, v, w = particle
        fitness_values.append(endurance(x, y, z, u, v, w))
    return -np.array(fitness_values)  


lower_bounds = np.zeros(6)  
upper_bounds = np.ones(6)   
bounds = (lower_bounds, upper_bounds)


options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}  


optimizer = ps.single.GlobalBestPSO(
    n_particles=10, 
    dimensions=6,    
    options=options,
    bounds=bounds
)


best_cost, best_pos = optimizer.optimize(swarm_fitness, iters=100)

print(f"Best cost (maximum endurance): {-best_cost}") 
print(f"Best position (combination of metals): {best_pos}")



# Plot cost history
plot_cost_history(cost_history=optimizer.cost_history)
plt.title("Cost History")
plt.show()
