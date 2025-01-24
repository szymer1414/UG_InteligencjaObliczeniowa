import matplotlib.pyplot as plt
import random
import time
from aco import AntColony

plt.style.use("dark_background")
#
def generate_random_coords(num_nodes):
    return [(random.randint(0, 100), random.randint(0, 100)) for _ in range(num_nodes)]


def plot_nodes(coords, w=12, h=8):
    for x, y in coords:
        plt.plot(x, y, "g.", markersize=40)
    plt.axis("off")
    fig = plt.gcf()
    fig.set_size_inches([w, h])


def plot_optimal_path(coords, optimal_nodes):
    for i in range(len(optimal_nodes) - 1):
        plt.plot(
            (optimal_nodes[i][0], optimal_nodes[i + 1][0]),
            (optimal_nodes[i][1], optimal_nodes[i + 1][1]),
            "r-",
        )

def run_aco(coords, ant_count, alpha, beta, pheromone_evaporation_rate, pheromone_constant, iterations):
    colony = AntColony(
        coords,
        ant_count=ant_count,
        alpha=alpha,
        beta=beta,
        pheromone_evaporation_rate=pheromone_evaporation_rate,
        pheromone_constant=pheromone_constant,
        iterations=iterations,
    )

    start_time = time.time()
    optimal_nodes = colony.get_path()
    end_time = time.time()

    runtime = end_time - start_time
    return optimal_nodes, runtime

num_nodes = 40  # Increase the number of nodes for TSP
coords = generate_random_coords(num_nodes)
plot_nodes(coords)

optimal_nodes, runtime = run_aco(
    coords,
    ant_count=300,
    alpha=0.5,
    beta=1.2,
    pheromone_evaporation_rate=0.4,
    pheromone_constant=1000.0,
    iterations=300,
)

# Plot the optimal path
plot_optimal_path(coords, optimal_nodes)

# Show the results
plt.title(f"ACO TSP Solution (Runtime: {runtime:.2f} seconds)")
plt.show()

print("Optimal Path:", optimal_nodes)
print("Runtime:", runtime)

