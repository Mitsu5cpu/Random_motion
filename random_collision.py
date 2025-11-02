import numpy as np
import matplotlib.pyplot as plt
# USER INPUTS
shape = input("Enter boundary shape ('circle', 'rectangle', or 'triangle'): ").strip().lower()
# --- Define boundary ---
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
elif shape == 'triangle':
    print("Enter triangle vertices (x1 y1), (x2 y2), (x3 y3):")
    v1 = np.array(list(map(float, input("Vertex 1 (e.g., 0 0): ").split())))
    v2 = np.array(list(map(float, input("Vertex 2 (e.g., 10 0): ").split())))
    v3 = np.array(list(map(float, input("Vertex 3 (e.g., 5 8): ").split())))
    verts = np.array([v1, v2, v3])
    bounds = {'shape': 'triangle', 'vertices': verts}
else:
    raise ValueError("Invalid shape. Choose 'circle', 'rectangle', or 'triangle'.")
# --- Simulation parameters ---
num_points = int(input("Enter number of points (e.g., 3): "))
duration = float(input("Enter simulation duration in seconds (e.g., 50): "))
dt = 0.01        # time step
speed = float(input("Enter base speed (e.g., 1.0): "))
num_steps = int(duration / dt)
arrow_interval = max(1, int(1 / dt))  # every 1s (at least 1)
# --- Particle properties ---
masses = np.array([float(input(f"Enter mass for point {i+1} (e.g., 1.0): ")) for i in range(num_points)])
# --- Start positions ---
same_start = input("Should all points start at the same position? (yes/no): ").strip().lower()
np.random.seed(0)
def point_in_triangle(pt, v1, v2, v3):
    # barycentric coordinates
    denom = (v2[1]-v3[1])*(v1[0]-v3[0]) + (v3[0]-v2[0])*(v1[1]-v3[1])
    a = ((v2[1]-v3[1])*(pt[0]-v3[0]) + (v3[0]-v2[0])*(pt[1]-v3[1])) / denom
    b = ((v3[1]-v1[1])*(pt[0]-v3[0]) + (v1[0]-v3[0])*(pt[1]-v3[1])) / denom
    c = 1 - a - b
    return (a >= 0) and (b >= 0) and (c >= 0)
def nearest_point_on_segment(p, a, b):
    # returns nearest point on segment ab to p
    ab = b - a
    t = np.dot(p - a, ab) / (np.dot(ab, ab) + 1e-12)
    t_clamped = max(0.0, min(1.0, t))
    return a + ab * t_clamped
# Initialize positions
if same_start == "yes":
    x0 = float(input("Enter X coordinate of start position (e.g., 0): "))
    y0 = float(input("Enter Y coordinate of start position (e.g., 0): "))
    start_position = np.array([x0, y0])
    # Validate inside boundary
    if bounds['shape'] == 'circle':
        if np.linalg.norm(start_position - bounds['center']) > bounds['radius']:
            raise ValueError("Starting point is outside the circular boundary.")
    elif bounds['shape'] == 'rectangle':
        if not (bounds['x'][0] <= x0 <= bounds['x'][1] and bounds['y'][0] <= y0 <= bounds['y'][1]):
            raise ValueError("Starting point is outside the rectangular boundary.")
    else:  # triangle
        v1, v2, v3 = bounds['vertices']
        if not point_in_triangle(start_position, v1, v2, v3):
            raise ValueError("Starting point is outside the triangular boundary.")
    positions = np.repeat(start_position[np.newaxis, :], num_points, axis=0)
else:
    if bounds['shape'] == 'circle':
        r = bounds['radius'] * np.sqrt(np.random.rand(num_points))
        theta = 2 * np.pi * np.random.rand(num_points)
        positions = np.column_stack((r * np.cos(theta), r * np.sin(theta))) + bounds['center']
    elif bounds['shape'] == 'rectangle':
        positions = np.random.uniform([bounds['x'][0], bounds['y'][0]],
                                      [bounds['x'][1], bounds['y'][1]], (num_points, 2))
    else:  # triangle
        v1, v2, v3 = bounds['vertices']
        positions = []
        bbox_min = np.min(bounds['vertices'], axis=0)
        bbox_max = np.max(bounds['vertices'], axis=0)
        while len(positions) < num_points:
            p = np.random.uniform(bbox_min, bbox_max)
            if point_in_triangle(p, v1, v2, v3):
                positions.append(p)
        positions = np.array(positions)
# --- Initialize velocities ---
angles = np.random.uniform(0, 2 * np.pi, num_points)
velocities = np.stack((np.cos(angles), np.sin(angles)), axis=1) * speed
# --- Storage ---
trajectories = np.zeros((num_points, num_steps, 2))
vel_history = np.zeros((num_points, num_steps, 2))
collision_events = []
# Precompute triangle edge normals (pointing outward) if triangle
if bounds['shape'] == 'triangle':
    v1, v2, v3 = bounds['vertices']
    edges = [(v1, v2), (v2, v3), (v3, v1)]
    # compute centroid
    centroid = np.mean(bounds['vertices'], axis=0)
    edge_normals = []
    for a, b in edges:
        e = b - a
        # candidate normal (perp)
        n = np.array([-e[1], e[0]])
        n = n / (np.linalg.norm(n) + 1e-12)
        # ensure n points outward: check vector from midpoint toward centroid
        midpoint = (a + b) / 2
        to_centroid = centroid - midpoint
        # if normal points toward centroid, flip (we want outward)
        if np.dot(n, to_centroid) > 0:
            n = -n
        edge_normals.append((a, b, n))  # store edge endpoints and outward normal
else:
    edge_normals = None
# SIMULATION LOOP
collision_threshold = 0.5  # Increased collision threshold
for t in range(num_steps):
    positions += velocities * dt
    # --- Boundary reflections ---
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
            vec = positions[i] - bounds['center']
            dist = np.linalg.norm(vec)
            if dist > bounds['radius']:
                normal = vec / (dist + 1e-12)
                velocities[i] = velocities[i] - 2 * np.dot(velocities[i], normal) * normal
                positions[i] = bounds['center'] + normal * bounds['radius']
        elif bounds['shape'] == 'triangle':
            # check each edge; if outside (positive signed distance), reflect and push back in
            pushed = False
            for a, b, n in edge_normals:
                signed_dist = np.dot(positions[i] - a, n)
                if signed_dist > 1e-6:  # outside this edge
                    # reflect velocity about outward normal
                    velocities[i] = velocities[i] - 2 * np.dot(velocities[i], n) * n
                    # push point back inside by amount signed_dist + small epsilon
                    positions[i] = positions[i] - (signed_dist + 1e-6) * n
                    pushed = True
            # After edge handling, if still outside (numerical/vertex case), snap to nearest point on triangle
            if not point_in_triangle(positions[i], v1, v2, v3):
                # find nearest point on any edge segment
                nearest = None
                nearest_dist = np.inf
                for a, b in [(v1, v2), (v2, v3), (v3, v1)]:
                    q = nearest_point_on_segment(positions[i], a, b)
                    d = np.linalg.norm(positions[i] - q)
                    if d < nearest_dist:
                        nearest_dist = d
                        nearest = q
                # reflect about vector from nearest->pos
                normal_vec = positions[i] - nearest
                norm_len = np.linalg.norm(normal_vec)
                if norm_len < 1e-9:
                    # extremely close; just nudge inside toward centroid and reverse velocity component
                    to_centroid = centroid - positions[i]
                    normal_unit = to_centroid / (np.linalg.norm(to_centroid) + 1e-12)
                else:
                    normal_unit = normal_vec / norm_len
                velocities[i] = velocities[i] - 2 * np.dot(velocities[i], normal_unit) * normal_unit
                positions[i] = nearest + 1e-6 * normal_unit  # place just inside
    # --- Inter-particle collisions (mass-dependent elastic) ---
    for i in range(num_points):
        for j in range(i + 1, num_points):
            delta_pos = positions[i] - positions[j]
            dist = np.linalg.norm(delta_pos)
            if dist < collision_threshold:
                normal = delta_pos / (dist + 1e-12)
                rel_vel = velocities[i] - velocities[j]
                vel_along_normal = np.dot(rel_vel, normal)
                if vel_along_normal < 0:
                    m1, m2 = masses[i], masses[j]
                    # impulse scalar using reduced mass formulation
                    impulse = (2 * vel_along_normal) / (1/m1 + 1/m2)
                    velocities[i] -= (impulse / m1) * normal
                    velocities[j] += (impulse / m2) * normal
                    # store collision time step and point index for both particles
                    collision_events.append((t, i))
                    collision_events.append((t, j))
    trajectories[:, t, :] = positions
    vel_history[:, t, :] = velocities
# HEATMAP
sampled_points = trajectories[:, ::arrow_interval, :].reshape(-1, 2)
bins = 120
# define plotting range: adapt to bounds
if bounds['shape'] == 'circle':
    limit = bounds['radius'] * 1.1
    range_limits = [[-limit, limit], [-limit, limit]]
elif bounds['shape'] == 'rectangle':
    range_limits = [[bounds['x'][0], bounds['x'][1]], [bounds['y'][0], bounds['y'][1]]]
else: # triangle
    bbox_min = np.min(bounds['vertices'], axis=0) - 1.0
    bbox_max = np.max(bounds['vertices'], axis=0) + 1.0
    range_limits = [[bbox_min[0], bbox_max[0]], [bbox_min[1], bbox_max[1]]]
heatmap, xedges, yedges = np.histogram2d(sampled_points[:, 0], sampled_points[:, 1],
                                         bins=bins, range=range_limits)
heatmap = heatmap.T / (np.max(heatmap) + 1e-12)
# PLOTTING
plt.figure(figsize=(9, 9))
extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
plt.imshow(heatmap, extent=extent, origin='lower', cmap='plasma', alpha=0.45)
# --- Boundary ---
if bounds['shape'] == 'circle':
    circle = plt.Circle(bounds['center'], bounds['radius'], color='k', fill=False, linestyle='--', label='Boundary')
    plt.gca().add_patch(circle)
elif bounds['shape'] == 'rectangle':
    x_min, x_max = bounds['x']
    y_min, y_max = bounds['y']
    plt.plot([x_min, x_max, x_max, x_min, x_min],
             [y_min, y_min, y_max, y_max, y_min], 'k--', label='Boundary')
else:  # triangle
    tri = np.vstack([bounds['vertices'], bounds['vertices'][0]])
    plt.plot(tri[:, 0], tri[:, 1], 'k--', label='Boundary')
# --- Trajectories & Arrows ---
colors = plt.cm.tab10(np.linspace(0, 1, num_points))
for i in range(num_points):
    plt.plot(trajectories[i, :, 0], trajectories[i, :, 1], color=colors[i], label=f'Point {i+1} (m={masses[i]:.2f})')
    plt.scatter(trajectories[i, 0, 0], trajectories[i, 0, 1], color=colors[i], marker='o', s=30)
    for t in range(0, num_steps, arrow_interval):
        x, y = trajectories[i, t, :]
        u, v = vel_history[i, t, :]
        plt.arrow(x, y, u * 0.45, v * 0.45, head_width=0.12, head_length=0.18, fc=colors[i], ec=colors[i], alpha=0.9)
# --- Highlight collisions on trajectories ---
if len(collision_events) > 0:
    # Filter out duplicate collision events for a cleaner plot
    unique_collision_events = list(set(collision_events))
    # Plot markers on trajectories at collision points
    for t_step, p_index in unique_collision_events:
        plt.plot(trajectories[p_index, t_step, 0], trajectories[p_index, t_step, 1],
                 marker='o', color='red', markersize=8, markeredgecolor='black', zorder=5)
plt.title(f"{bounds['shape'].capitalize()} Boundary Simulation with Elastic Collisions\nHeatmap + Trajectories + Collision Highlights")
plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.axis("equal")
plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
plt.grid(True)
plt.tight_layout()
plt.show()
