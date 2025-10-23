import numpy as np
import matplotlib.pyplot as plt

# ==============================
# USER INPUTS
# ==============================
shape = input("Enter boundary shape ('circle' or 'rectangle'): ").strip().lower()

# Boundary setup
if shape == 'circle':
    radius = float(input("Enter circle radius (e.g., 5): "))
    center = np.array([0.0, 0.0])
    bounds = {'shape': 'circle', 'radius': radius, 'center': center}
elif shape == 'rectangle':
    x_min = float(input("Enter rectangle min X (e.g., -5): "))
    x_max = float(input("Enter rectangle max X (e.g., 5): "))
    y_min = float(input("Enter rectangle min Y (e.g., -5): "))
    y_max = float(input("Enter rectangle max Y (e.g., 5): "))
    bounds = {'shape': 'rectangle', 'x': (x_min, x_max), 'y': (y_min, y_max)}
else:
    raise ValueError("Invalid shape. Choose 'circle' or 'rectangle'.")

# Simulation parameters
num_points = int(input("Enter number of points (e.g., 3): "))
duration = float(input("Enter simulation duration (s, e.g., 50): "))
dt = 0.01 # Decreased dt for better collision detection
speed = float(input("Enter base speed (e.g., 1.0): "))
num_steps = int(duration / dt) # Recalculate num_steps based on new dt
arrow_interval = int(1 / dt)  # every 1s

# Momentum/mass for each point
masses = []
for i in range(num_points):
    m = float(input(f"Enter momentum (mass) for point {i+1} (e.g., 1.0): "))
    masses.append(m)
masses = np.array(masses)

# Same starting position?
same_start = input("Should all points start at the same position? (yes/no): ").strip().lower()

np.random.seed(0)

if same_start == "yes":
    x0 = float(input("Enter X coordinate of start position (e.g., 0): "))
    y0 = float(input("Enter Y coordinate of start position (e.g., 0): "))
    start_position = np.array([x0, y0])
    positions = np.repeat(start_position[np.newaxis, :], num_points, axis=0)
else:
    if shape == 'circle':
        r = bounds['radius'] * np.sqrt(np.random.rand(num_points))
        theta = 2 * np.pi * np.random.rand(num_points)
        positions = np.column_stack((r * np.cos(theta), r * np.sin(theta))) + bounds['center']
    else:
        positions = np.random.uniform([bounds['x'][0], bounds['y'][0]],
                                      [bounds['x'][1], bounds['y'][1]], (num_points, 2))

# Random directions for velocities
angles = np.random.uniform(0, 2 * np.pi, num_points)
velocities = np.stack((np.cos(angles), np.sin(angles)), axis=1) * speed

# Data storage
trajectories = np.zeros((num_points, num_steps, 2))
vel_history = np.zeros((num_points, num_steps, 2))
collision_events = []  # store (x, y, time) for plotting

# ==============================
# SIMULATION LOOP
# ==============================
collision_threshold = 0.5 # Increased collision threshold

for t in range(num_steps):
    positions += velocities * dt

    # --- Wall reflections ---
    for i in range(num_points):
        if shape == 'rectangle':
            for dim, axis in enumerate(['x', 'y']):
                if positions[i, dim] < bounds[axis][0]:
                    positions[i, dim] = bounds[axis][0]
                    velocities[i, dim] *= -1
                elif positions[i, dim] > bounds[axis][1]:
                    positions[i, dim] = bounds[axis][1]
                    velocities[i, dim] *= -1
        else:
            vec_from_center = positions[i] - bounds['center']
            dist = np.linalg.norm(vec_from_center)
            if dist > bounds['radius']:
                normal = vec_from_center / dist
                velocities[i] = velocities[i] - 2 * np.dot(velocities[i], normal) * normal
                positions[i] = bounds['center'] + normal * bounds['radius']

    # --- Inter-particle collisions ---
    for i in range(num_points):
        for j in range(i + 1, num_points):
            delta_pos = positions[i] - positions[j]
            dist = np.linalg.norm(delta_pos)
            if dist < collision_threshold:  # collision threshold
                normal = delta_pos / (dist + 1e-8)
                rel_vel = velocities[i] - velocities[j]
                vel_along_normal = np.dot(rel_vel, normal)

                if vel_along_normal < 0:  # moving toward each other
                    m1, m2 = masses[i], masses[j]
                    impulse = (2 * vel_along_normal) / (1/m1 + 1/m2)
                    velocities[i] -= (impulse / m1) * normal
                    velocities[j] += (impulse / m2) * normal
                    # store collision location and time
                    mid_point = (positions[i] + positions[j]) / 2
                    collision_events.append((mid_point[0], mid_point[1], t * dt))

    trajectories[:, t, :] = positions
    vel_history[:, t, :] = velocities

# ==============================
# HEATMAP CREATION
# ==============================
sampled_points = trajectories[:, ::arrow_interval, :].reshape(-1, 2)
bins = 100

if shape == 'circle':
    limit = bounds['radius']
    range_limits = [[-limit, limit], [-limit, limit]]
else:
    range_limits = [[bounds['x'][0], bounds['x'][1]], [bounds['y'][0], bounds['y'][1]]]

heatmap, xedges, yedges = np.histogram2d(
    sampled_points[:, 0], sampled_points[:, 1],
    bins=bins, range=range_limits
)
heatmap = heatmap.T / np.max(heatmap)

# ==============================
# PLOTTING
# ==============================
plt.figure(figsize=(8, 8))
extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
plt.imshow(heatmap, extent=extent, origin='lower', cmap='plasma', alpha=0.4)

# Boundary visualization
if shape == 'circle':
    circle = plt.Circle(bounds['center'], bounds['radius'], color='k', fill=False, linestyle='--', label='Boundary')
    plt.gca().add_patch(circle)
else:
    x_min, x_max = bounds['x']
    y_min, y_max = bounds['y']
    plt.plot([x_min, x_max, x_max, x_min, x_min],
             [y_min, y_min, y_max, y_max, y_min],
             'k--', label='Boundary')

# Trajectories + arrows
colors = plt.cm.tab10(np.linspace(0, 1, num_points))
for i in range(num_points):
    plt.plot(trajectories[i, :, 0], trajectories[i, :, 1], color=colors[i], label=f'Point {i+1} (m={masses[i]})')
    for t in range(0, num_steps, arrow_interval):
        x, y = trajectories[i, t, :]
        u, v = vel_history[i, t, :]
        plt.arrow(x, y, u * 0.5, v * 0.5,
                  head_width=0.15, head_length=0.25, fc=colors[i], ec=colors[i])

# --- Highlight collisions ---
if len(collision_events) > 0:
    collision_events = np.array(collision_events)
    plt.scatter(collision_events[:, 0], collision_events[:, 1],
                c='red', s=100, alpha=0.8, label='Collisions', edgecolors='black', zorder=5) # Increased size and zorder for visibility

plt.title(f"{shape.capitalize()} Boundary with Interacting Points\nElastic Collisions (Red Dots) & Heatmap of Positions")
plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.axis("equal")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
