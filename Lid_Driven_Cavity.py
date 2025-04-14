import numpy as np
import matplotlib.pyplot as plt
from numba import njit
import csv
import os
import imageio

# Apply boundary conditions
def apply_boundary_conditions(u, v):
    # === u velocity boundaries ===
    u[0, :] = 0            # Left wall (x = 0)
    u[-1, :] = 0           # Right wall (x = Lx)
    u[:, 0] = 0            # Bottom wall (y = 0)
    u[:, -1] = 1           # Top lid (y = Ly)

    # === v velocity boundaries ===
    v[0, :] = 0            # Left wall (x = 0)
    v[-1, :] = 0           # Right wall (x = Lx)
    v[:, 0] = 0            # Bottom wall (y = 0)
    v[:, -1] = 0           # Top wall (y = Ly)

    return u, v

@njit
def compute_rhs(u, v, dx, dy, nu):
    # Allocate arrays for du/dt and dv/dt
    dudt = np.zeros_like(u)
    dvdt = np.zeros_like(v)

    # Internal domain (excluding boundaries)
    for i in range(1, u.shape[0] - 1):
        for j in range(1, u.shape[1] - 1):
            # First-order convection terms (u ∂u/∂x + v ∂u/∂y)
            u_conv = u[i, j] * (u[i+1, j] - u[i-1, j]) / (2 * dx) + v[i, j] * (u[i, j+1] - u[i, j-1]) / (2 * dy)
            v_conv = u[i, j] * (v[i+1, j] - v[i-1, j]) / (2 * dx) + v[i, j] * (v[i, j+1] - v[i, j-1]) / (2 * dy)

            # Diffusion terms (ν ∇²u and ∇²v)
            u_diff = nu * ((u[i+1, j] - 2*u[i, j] + u[i-1, j]) / dx**2 +
                           (u[i, j+1] - 2*u[i, j] + u[i, j-1]) / dy**2)
            v_diff = nu * ((v[i+1, j] - 2*v[i, j] + v[i-1, j]) / dx**2 +
                           (v[i, j+1] - 2*v[i, j] + v[i, j-1]) / dy**2)

            dudt[i, j] = -u_conv + u_diff
            dvdt[i, j] = -v_conv + v_diff

    return dudt, dvdt

def rk4_step(u, v, dt, dx, dy, nu):
    dudt1, dvdt1 = compute_rhs(u, v, dx, dy, nu)
    u1 = u + 0.5 * dt * dudt1
    v1 = v + 0.5 * dt * dvdt1

    dudt2, dvdt2 = compute_rhs(u1, v1, dx, dy, nu)
    u2 = u + 0.5 * dt * dudt2
    v2 = v + 0.5 * dt * dvdt2

    dudt3, dvdt3 = compute_rhs(u2, v2, dx, dy, nu)
    u3 = u + dt * dudt3
    v3 = v + dt * dvdt3

    dudt4, dvdt4 = compute_rhs(u3, v3, dx, dy, nu)

    u_new = u + (dt / 6.0) * (dudt1 + 2*dudt2 + 2*dudt3 + dudt4)
    v_new = v + (dt / 6.0) * (dvdt1 + 2*dvdt2 + 2*dvdt3 + dvdt4)

    return u_new, v_new

@njit
def compute_divergence(u, v, dx, dy):
    div = np.zeros_like(u)
    for i in range(1, u.shape[0] - 1):
        for j in range(1, u.shape[1] - 1):
            div[i, j] = ((u[i+1, j] - u[i-1, j]) / (2 * dx) +
                         (v[i, j+1] - v[i, j-1]) / (2 * dy))
    return div

@njit
def solve_pressure_poisson(p, div, dx, dy, rho, dt, max_iter=1000, tol=1e-4):
    for it in range(max_iter):
        p_old = p.copy()
        for i in range(1, p.shape[0] - 1):
            for j in range(1, p.shape[1] - 1):
                p[i, j] = ((dy**2 * (p[i+1, j] + p[i-1, j]) +
                            dx**2 * (p[i, j+1] + p[i, j-1]) -
                            rho * dx**2 * dy**2 * div[i, j] / dt) /
                           (2 * (dx**2 + dy**2)))

        # Boundary conditions: Neumann (∂p/∂n = 0)
        p[0, :] = p[1, :]           # Left
        p[-1, :] = p[-2, :]         # Right
        p[:, 0] = p[:, 1]           # Bottom
        p[:, -1] = p[:, -2]         # Top

        # Convergence check
        res = np.linalg.norm(p - p_old, ord=2)
        if res < tol:
            break
    return p

@njit
def correct_velocity(u_star, v_star, p, dx, dy, rho, dt):
    u = u_star.copy()
    v = v_star.copy()

    for i in range(1, u.shape[0] - 1):
        for j in range(1, u.shape[1] - 1):
            u[i, j] -= dt / rho * (p[i+1, j] - p[i-1, j]) / (2 * dx)
            v[i, j] -= dt / rho * (p[i, j+1] - p[i, j-1]) / (2 * dy)
    return u, v

def simulate(u, v, p, dx, dy, dt, t_end, nu, rho, save_interval=100, csv_file_prefix="simulation_data"):
    t = 0.0
    step = 0

    # Open CSV files for writing u, v, p data
    u_file = open(f'{csv_file_prefix}_u.csv', 'w', newline='')
    v_file = open(f'{csv_file_prefix}_v.csv', 'w', newline='')
    p_file = open(f'{csv_file_prefix}_p.csv', 'w', newline='')

    # Create CSV writers
    u_writer = csv.writer(u_file)
    v_writer = csv.writer(v_file)
    p_writer = csv.writer(p_file)

    while t < t_end:
        # Step 1: RK4 intermediate velocity (u*, v*)
        u_star, v_star = rk4_step(u, v, dt, dx, dy, nu)

        # Step 2: Apply boundary conditions to intermediate velocity
        u_star, v_star = apply_boundary_conditions(u_star, v_star)

        # Step 3: Compute divergence of intermediate velocity
        div = compute_divergence(u_star, v_star, dx, dy)

        # Step 4: Solve pressure Poisson equation
        p = solve_pressure_poisson(p, div, dx, dy, rho, dt)

        # Step 5: Correct velocity to make it divergence-free
        u, v = correct_velocity(u_star, v_star, p, dx, dy, rho, dt)

        # Step 6: Apply boundary conditions again (important after correction)
        u, v = apply_boundary_conditions(u, v)

        # Time update
        t += dt
        step += 1

        # Save data for animation and write to CSV
        if step % save_interval == 0:
            # Write u, v, p to their respective CSV files
            u_writer.writerow(u.flatten())  # Flatten the array to a single row
            v_writer.writerow(v.flatten())  # Flatten the array to a single row
            p_writer.writerow(p.flatten())  # Flatten the array to a single row

    # Close the files after simulation
    u_file.close()
    v_file.close()
    p_file.close()

    return u, v, p

def postprocess(csv_prefix="simulation_data", Nx=64, Ny=64, Lx=1.0, Ly=1.0):
    dx, dy = Lx / Nx, Ly / Ny
    x = np.linspace(0, Lx, Nx+1)
    y = np.linspace(0, Ly, Ny+1)
    X, Y = np.meshgrid(x, y, indexing='ij')

    # Load CSV data
    u_data = np.loadtxt(f"{csv_prefix}_u.csv", delimiter=",")
    v_data = np.loadtxt(f"{csv_prefix}_v.csv", delimiter=",")
    p_data = np.loadtxt(f"{csv_prefix}_p.csv", delimiter=",")

    # Ensure data is 2D: (frames, flattened field)
    if u_data.ndim == 1:
        u_data = np.expand_dims(u_data, axis=0)
        v_data = np.expand_dims(v_data, axis=0)
        p_data = np.expand_dims(p_data, axis=0)

    os.makedirs("frames", exist_ok=True)

    for idx in range(u_data.shape[0]):
        u = u_data[idx].reshape((Nx+1, Ny+1))
        v = v_data[idx].reshape((Nx+1, Ny+1))
        speed = np.sqrt(u**2 + v**2)

        plt.figure(figsize=(14, 12))

        # Plot contour
        contour = plt.contourf(X, Y, speed, levels=50, cmap='viridis')
        cbar = plt.colorbar(contour, label='Velocity Magnitude', ticks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0], fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=14)
        cbar.set_label('Velocity Magnitude', fontsize=16)

        # Quiver plot
        plt.quiver(X[::2, ::2], Y[::2, ::2], u[::2, ::2], v[::2, ::2], color='white', scale=5)

        # Titles and labels
        plt.title(f"Velocity field (t = {(idx + 1)*dt*100})", fontsize=20)
        plt.xlabel("x", fontsize=16)
        plt.ylabel("y", fontsize=16)

        # Tick labels
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)

        plt.axis('equal')
        plt.tight_layout()

        # Save each frame
        plt.savefig(f"frames/frame_{idx:04d}.png")
        plt.close()

    print("Postprocessing complete. Frames saved in ./frames/")

def make_gif(frame_folder="frames", output_gif="simulation.gif", fps=5):
    # Collect all frame filenames and sort them
    frame_files = sorted([f for f in os.listdir(frame_folder) if f.endswith(".png")])
    images = []

    for filename in frame_files:
        filepath = os.path.join(frame_folder, filename)
        images.append(imageio.v2.imread(filepath))  # Use v2 to avoid deprecation warning

    # Save as GIF
    imageio.mimsave(output_gif, images, fps=fps)
    print(f"GIF saved as {output_gif}")


############################################################################################################

# Domain parameters
Lx, Ly = 1.0, 1.0
Nx, Ny = 64, 64
dx, dy = Lx / Nx, Ly / Ny

# Time parameters
dt = 0.001
t_end = 1.0
nu = 0.01  # kinematic viscosity
rho = 1.0  # density
'''
# Grid
x = np.linspace(0, Lx, Nx+1)
y = np.linspace(0, Ly, Ny+1)
X, Y = np.meshgrid(x, y, indexing='ij')

# Initialize fields
u = np.zeros((Nx+1, Ny+1))  # x-velocity
v = np.zeros((Nx+1, Ny+1))  # y-velocity
p = np.zeros((Nx+1, Ny+1))  # pressure

# SIMULATION ###############################################################################################
u, v = apply_boundary_conditions(u, v)
u_final, v_final, p_final = simulate(u, v, p, dx, dy, dt, t_end, nu, rho)
'''
# POSTPROCESSING ###########################################################################################
postprocess(csv_prefix="simulation_data", Nx=64, Ny=64, Lx=1.0, Ly=1.0)
make_gif(frame_folder="frames", output_gif="simulation.gif", fps=10)