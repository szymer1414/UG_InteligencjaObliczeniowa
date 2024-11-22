import math
import random
import matplotlib.pyplot as plt

def calculate_distance(v0, alpha_deg, height):
    """Calculate the horizontal distance of the projectile."""
    alpha = math.radians(alpha_deg)  # Convert angle from degrees to radians
    g = 9.81  # Gravitational acceleration in m/s^2

    # Horizontal and vertical components of the velocity
    v0x = v0 * math.cos(alpha)
    v0y = v0 * math.sin(alpha)

    # Time of flight (solving the quadratic equation)
    discriminant = v0y**2 + 2 * g * height
    t = (v0y + math.sqrt(discriminant)) / g  # Time until the projectile hits the ground

    distance = v0x * t  # Horizontal distance
    return distance

def generate_target():
    """Generate a random target distance between 50 and 340 meters."""
    return random.randint(50, 340)

def calculate_trajectory(v0, alpha_deg, height):
    """Calculate the x and y coordinates of the projectile over time."""
    alpha = math.radians(alpha_deg)
    g = 9.81  # Gravitational acceleration in m/s^2

    # Horizontal and vertical components of the velocity
    v0x = v0 * math.cos(alpha)
    v0y = v0 * math.sin(alpha)

    # Time of flight
    discriminant = v0y**2 + 2 * g * height
    t_total = (v0y + math.sqrt(discriminant)) / g

    # Calculate positions at different times
    x_vals = []
    y_vals = []
    t = 0  # Initial time
    dt = 0.01  # Time step

    while t <= t_total:
        x = v0x * t  # x position
        y = height + v0y * t - 0.5 * g * t**2  # y position

        if y < 0:  # Stop if the projectile hits the ground
            break

        x_vals.append(x)
        y_vals.append(y)
        t += dt  # Increment time

    return x_vals, y_vals

def plot_trajectory(x_vals, y_vals, target, count):
    """Plot the trajectory of the projectile and save it to a file."""
    plt.figure(figsize=(10, 5))
    plt.plot(x_vals, y_vals, label="Trajectory")
    plt.axvline(x=target, color='r', linestyle='--', label=f'Target ({target} m)')
    plt.title("Trebuchet Projectile Trajectory")
    plt.xlabel("Horizontal Distance (m)")
    plt.ylabel("Vertical Distance (m)")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{target}_{count}.png")  # Save the plot
    plt.close()  # Close the plot to free memory

def main():
    """Main function to run the trebuchet simulation."""
    v0 = 50  # Initial velocity in m/s
    height = 100  # Height of the trebuchet in meters

    target = generate_target()  # Generate random target
    distance = 0
    count = 0

    print(f"Musisz trafić w cel, który jest na odległości: {target} metrów")

    while abs(distance - target) > 5:  # Continue until close to target
        try:
            alpha_deg = int(input("Podaj kąt trebuszeta (w stopniach): "))
            distance = calculate_distance(v0, alpha_deg, height)
            print(f"The projectile will travel {distance:.2f} meters.")

            # Check how close the user is to the target
            difference = abs(distance - target)
            print(f"Od celu: {difference:.2f} metrów")

            # Increment shot count and calculate trajectory
            count += 1
            x_vals, y_vals = calculate_trajectory(v0, alpha_deg, height)
            plot_trajectory(x_vals, y_vals, target, count)

        except ValueError:
            print("Błąd: Wprowadź poprawny kąt (liczba całkowita)!")

    print(f"Mamy go szefie! I to tylko w {count} próbach.")

if __name__ == "__main__":
    main()
