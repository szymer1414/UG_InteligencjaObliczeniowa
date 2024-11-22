
import math
import random
import matplotlib.pyplot as plt
def calculate_distance(v0, alpha_deg, height):
    # Convert angle from degrees to radians
    alpha = math.radians(alpha_deg)
    
    # Constants
    g = 9.81  # gravitational acceleration in m/s^2
    
    # Horizontal and vertical components of the velocity
    v0x = v0 * math.cos(alpha)
    v0y = v0 * math.sin(alpha)
    
    # Time of flight (solving the quadratic equation)
    discriminant = v0y**2 + 2 * g * height
    t = (v0y + math.sqrt(discriminant)) / g
    
    # Horizontal distance
    distance = v0x * t
    
    return distance

def generate_target():
    x = random.randint(50, 340)
    return x
def calculate_trajectory(v0, alpha_deg, height):
    """
    Calculate the x and y coordinates of the projectile over time.
    Args:
    v0: initial velocity in m/s
    alpha_deg: launch angle in degrees
    height: initial height in meters

    Returns:
    x_vals: list of x positions (horizontal distance)
    y_vals: list of y positions (vertical distance)
    """
    # Convert angle from degrees to radians
    alpha = math.radians(alpha_deg)
    
    # Constants
    g = 9.81  # gravitational acceleration in m/s^2
    
    # Horizontal and vertical components of the velocity
    v0x = v0 * math.cos(alpha)
    v0y = v0 * math.sin(alpha)
    
    # Time of flight (solving the quadratic equation for positive root)
    discriminant = v0y**2 + 2 * g * height
    t_total = (v0y + math.sqrt(discriminant)) / g  # Total flight time
    
    # Calculate positions at different times
    x_vals = []
    y_vals = []
    t = 0  # Initial time
    dt = 0.01  # Time step (smaller = more points on trajectory)

    while t <= t_total:
        # x(t) = v0x * t
        x = v0x * t
        
        # y(t) = h + v0y * t - (1/2) * g * t^2
        y = height + v0y * t - 0.5 * g * t**2
        
        # Stop if the projectile hits the ground (y <= 0)
        if y < 0:
            break
        
        x_vals.append(x)
        y_vals.append(y)
        
        t += dt  # Increment time by dt
    
    return x_vals, y_vals
def plot_trajectory(x_vals, y_vals, target):
    plt.figure(figsize=(10, 5))
    plt.plot(x_vals, y_vals, label="Trajectory")
    plt.axvline(x=target, color='r', linestyle='--', label=f'Target ({target} m)')
    plt.title("Trebuchet Projectile Trajectory")
    plt.xlabel("Horizontal Distance (m)")
    plt.ylabel("Vertical Distance (m)")
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.savefig(f"{target}_{count}.png")
# Example usage
v0 = 50  # initial velocity in m/s
height = 100  # height of the trebuchet in meters
alpha_deg = 45  # launch angle in degrees


target = generate_target()
distance = 0
count =0
print("musisz trafic w cel, jest na oko, z jakieś ", target, "metrów")

while (abs(distance - target)>5):
    alpha_deg = int(input("kąt trebuszeta:"))
    distance = calculate_distance(v0, alpha_deg, height)
    print(f"The projectile will travel {distance:.2f} meters.")
    print(abs(distance - target), "od celu")
    count=count+1
    
    x_vals, y_vals = calculate_trajectory(v0, alpha_deg, height)
    plot_trajectory(x_vals, y_vals, target)
    
print("mamy go szefie! I to tylko w",count,"próbach")
