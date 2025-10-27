import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange

# ==============================================================
#               FAST SIMULATION (Numba Accelerated)
# ==============================================================

@njit(parallel=True)
def simulate_numba(positions, velocities, masses, bounds, dt, steps, shape_code, radius):
    n = positions.shape[0]
    traj = np.zeros((steps, n, 2))
    collisions = np.zeros((steps, n, 2))
    flags = np.zeros((steps, n))

    for t in prange(steps):
        # Move particles
        positions += velocities * dt

        # Boundary reflections
        for i in range(n):
            if shape_code == 0:  # rectangle
                if positions[i,0] <= bounds[0,0] or positions[i,0] >= bounds[0,1]:
                    velocities[i,0] *= -1
                    flags[t,i] = 1
                if positions[i,1] <= bounds[1,0] or positions[i,1] >= bounds[1,1]:
                    velocities[i,1] *= -1
                    flags[t,i] = 1

            elif shape_code == 1:  # circle
                r = np.sqrt(positions[i,0]**2 + positions[i,1]**2)
                if r >= radius:
                    nx, ny = positions[i,0]/r, positions[i,1]/r
                    vn = velocities[i,0]*nx + velocities[i,1]*ny
                    velocities[i,0] -= 2*vn*nx
                    velocities[i,1] -= 2*vn*ny
                    positions[i,:] = positions[i,:] / r * (radius - 1e-6)
                    flags[t,i] = 1

            elif shape_code == 2:  # equilateral triangle
                x, y = positions[i]
                h = np.sqrt(3) * radius
                if y < 0:
                    velocities[i,1] *= -1
                    positions[i,1] = 0
                    flags[t,i] = 1
                elif y > h:
                    velocities[i,1] *= -1
                    positions[i,1] = h
                    flags[t,i] = 1
                else:
                    left_x = (h - y) / np.sqrt(3)
                    right_x = -left_x
                    if x < right_x:
                        velocities[i,0] *= -1
                        positions[i,0] = right_x
                        flags[t,i] = 1
                    elif x > left_x:
                        velocities[i,0] *= -1
                        positions[i,0] = left_x
                        flags[t,i] = 1

        # Inter-particle collisions (elastic)
        for i in range(n):
            for j in range(i+1, n):
                dx = positions[i,0] - positions[j,0]
                dy = positions[i,1] - positions[j,1]
                dist_sq = dx*dx + dy*dy
                if dist_sq < 0.04**2:  # collision radius
                    nx, ny = dx / np.sqrt(dist_sq), dy / np.sqrt(dist_sq)
                    p = 2 * (velocities[i,0]*nx + velocities[i,1]*ny
                             - velocities[j,0]*nx - velocities[j,1]*ny) / (masses[i] + masses[j])
                    velocities[i,0] -= p * masses[j] * nx
                    velocities[i,1] -= p * masses[j] * ny
                    velocities[j,0] += p * masses[i] * nx
                    velocities[j,1] += p * masses[i] * ny
                    flags[t,i] = 1
                    flags[t,j] = 1

        traj[t,:,:] = positions
        for i in range(n):
            if flags[t,i] == 1:
                collisions[t,i,:] = positions[i,:]

    return traj, collisions, flags


# ==============================================================
#                   MAIN INTERACTIVE SCRIPT
# ==============================================================

def run_simulation():
    print("\n=== PARTICLE COLLISION SIMULATION ===")

    # ----- Shape Selection -----
    shape = input("Choose boundary shape (rectangle / circle / triangle): ").strip().lower()
    if shape == "rectangle":
        shape_code = 0
        x_min = float(input("Enter x_min: "))
        x_max = float(input("Enter x_max: "))
        y_min = float(input("Enter y_min: "))
        y_max = float(input("Enter y_max: "))
        bounds = np.array([[x_min, x_max], [y_min, y_max]])
        radius = 0.0
    elif shape == "circle":
        shape_code = 1
        radius = float(input("Enter circle radius: "))
        bounds = np.array([[-radius, radius], [-radius, radius]])
    elif shape == "triangle":
        shape_code = 2
        radius = float(input("Enter triangle half-side length: "))
        h = np.sqrt(3)*radius
        bounds = np.array([[-radius, radius], [0, h]])
    else:
        raise ValueError("Invalid shape entered.")

    # ----- Simulation Parameters -----
    n = int(input("Enter number of particles: "))
    duration = float(input("Enter simulation duration (seconds): "))
    dt = float(input("Enter time step (e.g., 0.01): "))
    steps = int(duration / dt)

    # ----- Starting Positions -----
    same_start = input("Do all points start from the same position? (yes/no): ").strip().lower()
    if same_start == "yes":
        x0 = float(input("Enter starting X: "))
        y0 = float(input("Enter starting Y: "))
        positions = np.tile(np.array([x0, y0]), (n, 1))
    else:
        if shape_code == 0:  # rectangle
            positions = np.random.uniform([bounds[0,0], bounds[1,0]],
                                          [bounds[0,1], bounds[1,1]], (n, 2))
        else:
            # uniformly distributed inside circle/triangle area
            theta = np.random.uniform(0, 2*np.pi, n)
            r = radius * np.sqrt(np.random.uniform(0, 1, n))
            positions = np.stack((r*np.cos(theta), r*np.sin(theta)), axis=1)

    # ----- Velocities -----
    angles = np.random.uniform(0, 2*np.pi, n)
    speeds = np.ones(n)
    velocities = np.stack((np.cos(angles), np.sin(angles)), axis=1) * speeds[:, None]


    # ----- Masses -----
    mode = input("Mass assignment (same/random/custom): ").strip().lower()
    if mode == "same":
        m = float(input("Enter common mass: "))
        masses = np.full(n, m)
    elif mode == "random":
        masses = np.random.uniform(0.5, 2.0, n)
    else:
        masses = np.zeros(n)
        print("Enter masses for each particle:")
        for i in range(n):
            masses[i] = float(input(f"Mass {i+1}: "))

    # ----- Collision Visualization -----
    show_col = input("Show collision points? (yes/no): ").strip().lower() == "yes"
    if show_col:
        col_color = input("Enter color for collision points (e.g. red, lime, magenta): ").strip()
        col_size = float(input("Enter marker size for collision points (e.g. 20): "))
    else:
        col_color = "none"
        col_size = 0

    # ==============================================================
    #               Run the Simulation (Numba Accelerated)
    # ==============================================================
    traj, col, flags = simulate_numba(positions.copy(), velocities.copy(),
                                      masses, bounds, dt, steps, shape_code, radius)

    # ==============================================================
    #                     Visualization
    # ==============================================================
    plt.figure(figsize=(8, 8))
    if shape_code == 0:
        plt.xlim(bounds[0])
        plt.ylim(bounds[1])
    else:
        plt.xlim(-radius*1.1, radius*1.1)
        plt.ylim(-radius*1.1, radius*1.1)

    for i in range(n):
        plt.plot(traj[:, i, 0], traj[:, i, 1], lw=0.6)
        if show_col:
            idx = np.where(flags[:, i] == 1)[0]
            plt.scatter(col[idx, i, 0], col[idx, i, 1],
                        s=col_size, color=col_color, alpha=0.7)

    plt.title(f"Numba Accelerated Simulation ({shape.capitalize()} boundary)")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.gca().set_aspect('equal')
    plt.tight_layout()
    plt.show()

    # Heatmap of occupancy
    all_xy = traj.reshape(-1, 2)
    heat, xedges, yedges = np.histogram2d(all_xy[:, 0], all_xy[:, 1], bins=100)
    plt.figure(figsize=(8, 6))
    plt.imshow(heat.T, origin='lower', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], cmap='hot')
    plt.title("Occupancy Heatmap")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.colorbar(label="Frequency")
    plt.show()


# ==============================================================
#                     Run Everything
# ==============================================================

if __name__ == "__main__":
    run_simulation()
