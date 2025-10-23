import numpy as np
import matplotlib.pyplot as plt

# =====================================
# Take user inputs
# =====================================
shape = input("Enter boundary shape ('circle' or 'rectangle'): ").strip().lower()

# Boundary input
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

# Simulation settings
num_points = int(input("Enter number of points (e.g., 3): "))
duration = float(input("Enter simulation duration in seconds (e.g., 50): "))
dt = float(input("Enter time step (e.g., 0.01): "))
speed = float(input("Enter constant speed for points (e.g., 1.0): "))

num_steps = int(duration / dt)
arrow_interval = int(1 / dt)  # every 1 second
np.random.seed(0)

# Ask about starting position
same_start = input("Should all points start at the same position? (yes/no): ").strip().lower()

if same_start == "yes":
    x0 = float(input("Enter X coordinate of starting position (e.g., 0): "))
    y0 = float(input("Enter Y coordinate of starting position (e.g., 0): "))
    start_position = np.array([x0, y0])

    # Check if valid within boundary
    if shape == 'circle':
        if np.linalg.norm(start_position - bounds['center']) > bounds['radius']:
            raise ValueError("Starting point is outside the circular boundary.")
    else:
        if not (bounds['x'][0] <= x0 <= bounds['x'][1] and bounds['y'][0] <= y0 <= bounds['y'][1]):
            raise ValueError("Starting point is outside the rectangular boundary.")

    positions = np.repeat(start_position[np.newaxis, :], num_points, axis=0)
else:
    if shape == 'circle':
        r = bounds['radius'] * np.sqrt(np.random.rand(num_points))
        theta = 2 * np.pi * np.random.rand(num_points)
        positions = np.column_stack((r * np.cos(theta), r * np.sin(theta))) + bounds['center']
    else:
        positions = np.random.uniform([bounds['x'][0], bounds['y'][0]],
                                      [bounds['x'][1], bounds['y'][1]], (num_points, 2))

# Initialize velocities
angles = np.random.uniform(0, 2 * np.pi, num_points)
velocities = np.stack((np.cos(angles), np.sin(angles)), axis=1) * speed

# Arrays to store data
trajectories = np.zeros((num_points, num_steps, 2))
vel_history = np.zeros((num_points, num_steps, 2))

# =====================================
# Simulation loop
# =====================================
for t in range(num_steps):
    positions += velocities * dt

    for i in range(num_points):
        if bounds['shape'] == 'rectangle':
            for dim, axis in enumerate(['x', 'y']):
                if positions[i, dim] < bounds[axis][0]:
                    positions[i, dim] = bounds[axis][0]
                    velocities[i, dim] *= -1
                elif positions[i, dim] > bounds[axis][1]:
                    positions[i, dim] = bounds[axis][1]
                    velocities[i, dim] *= -1

        elif bounds['shape'] == 'circle':
            vec_from_center = positions[i] - bounds['center']
            dist = np.linalg.norm(vec_from_center)
            if dist > bounds['radius']:
                normal = vec_from_center / dist
                velocities[i] = velocities[i] - 2 * np.dot(velocities[i], normal) * normal
                positions[i] = bounds['center'] + normal * bounds['radius']

    trajectories[:, t, :] = positions
    vel_history[:, t, :] = velocities

# =====================================
# Create heatmap (sample every second)
# =====================================
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

# =====================================
# Plot
# =====================================
plt.figure(figsize=(8, 8))
extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
plt.imshow(heatmap, extent=extent, origin='lower', cmap='plasma', alpha=0.4)

# Plot boundary
if shape == 'circle':
    circle = plt.Circle(bounds['center'], bounds['radius'], color='k', fill=False, linestyle='--', label='Boundary')
    plt.gca().add_patch(circle)
else:
    x_min, x_max = bounds['x']
    y_min, y_max = bounds['y']
    plt.plot([x_min, x_max, x_max, x_min, x_min],
             [y_min, y_min, y_max, y_max, y_min],
             'k--', label='Boundary')

# Plot trajectories + direction arrows
colors = plt.cm.tab10(np.linspace(0, 1, num_points))
for i in range(num_points):
    plt.plot(trajectories[i, :, 0], trajectories[i, :, 1], color=colors[i], label=f'Point {i+1}')
    plt.scatter(trajectories[i, 0, 0], trajectories[i, 0, 1], color=colors[i], marker='o', s=40)
    for t in range(0, num_steps, arrow_interval):
        x, y = trajectories[i, t, :]
        u, v = vel_history[i, t, :]
        plt.arrow(x, y, u * 0.5, v * 0.5,
                  head_width=0.15, head_length=0.25, fc=colors[i], ec=colors[i])

plt.title(f"{shape.capitalize()} Boundary Motion with {num_points} Point(s)\nHeatmap of Point Occurrence (1s Interval)")
plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.axis("equal")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
