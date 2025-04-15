import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange
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

# Right-Hand Side
@njit(parallel=True)
def RHS(u, v, dx, dy, nu):
    # Allocate arrays for du/dt and dv/dt
    dudt = np.zeros_like(u)
    dvdt = np.zeros_like(v)

    # Internal domain (excluding boundaries)
    for i in prange(1, u.shape[0] - 1):
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

# Runge-Kutta 3 (Shu-Osher form)
def RK3(u, v, dt, dx, dy, nu):
    # Stage 1
    dudt1, dvdt1 = RHS(u, v, dx, dy, nu)
    u1 = u + dt * dudt1
    v1 = v + dt * dvdt1

    # Stage 2
    dudt2, dvdt2 = RHS(u1, v1, dx, dy, nu)
    u2 = 0.75 * u + 0.25 * (u1 + dt * dudt2)
    v2 = 0.75 * v + 0.25 * (v1 + dt * dvdt2)

    # Stage 3
    dudt3, dvdt3 = RHS(u2, v2, dx, dy, nu)
    u_new = (1.0 / 3.0) * u + (2.0 / 3.0) * (u2 + dt * dudt3)
    v_new = (1.0 / 3.0) * v + (2.0 / 3.0) * (v2 + dt * dvdt3)

    return u_new, v_new

# Runge-Kutta 4
def RK4(u, v, dt, dx, dy, nu):
    dudt1, dvdt1 = RHS(u, v, dx, dy, nu)
    u1 = u + 0.5 * dt * dudt1
    v1 = v + 0.5 * dt * dvdt1

    dudt2, dvdt2 = RHS(u1, v1, dx, dy, nu)
    u2 = u + 0.5 * dt * dudt2
    v2 = v + 0.5 * dt * dvdt2

    dudt3, dvdt3 = RHS(u2, v2, dx, dy, nu)
    u3 = u + dt * dudt3
    v3 = v + dt * dvdt3

    dudt4, dvdt4 = RHS(u3, v3, dx, dy, nu)

    u_new = u + (dt / 6.0) * (dudt1 + 2*dudt2 + 2*dudt3 + dudt4)
    v_new = v + (dt / 6.0) * (dvdt1 + 2*dvdt2 + 2*dvdt3 + dvdt4)

    return u_new, v_new

@njit(parallel=True)
def compute_divergence(u, v, dx, dy):
    div = np.zeros_like(u)
    for i in prange(1, u.shape[0] - 1):
        for j in range(1, u.shape[1] - 1):
            div[i, j] = ((u[i+1, j] - u[i-1, j]) / (2 * dx) +
                         (v[i, j+1] - v[i, j-1]) / (2 * dy))
    return div

@njit(parallel=True)
def solve_pressure_poisson(p, div, dx, dy, rho, dt, max_iter=1000, tol=1e-4, omega=1.7):
    dx2 = dx * dx
    dy2 = dy * dy
    denom = 2.0 * (dx2 + dy2)
    
    for it in range(max_iter):
        res = 0.0
        for i in prange(1, p.shape[0] - 1):
            for j in range(1, p.shape[1] - 1):
                rhs = ((dy2 * (p[i+1, j] + p[i-1, j]) +
                        dx2 * (p[i, j+1] + p[i, j-1]) -
                        rho * dx2 * dy2 * div[i, j] / dt) / denom)
                diff = rhs - p[i, j]
                p[i, j] += omega * diff
                res += diff * diff

        # Apply Neumann BCs (∂p/∂n = 0)
        p[0, :] = p[1, :]
        p[-1, :] = p[-2, :]
        p[:, 0] = p[:, 1]
        p[:, -1] = p[:, -2]

        # Set pressure reference to zero at top-left corner (to avoid drift)
        p[0, -1] = 0.0

        # Check for convergence
        res = np.sqrt(res) / (p.shape[0] * p.shape[1])
        if res < tol:
            break
    
    return p

@njit(parallel=True)
def correct_velocity(u_star, v_star, p, dx, dy, rho, dt):
    u = u_star.copy()
    v = v_star.copy()

    for i in prange(1, u.shape[0] - 1):
        for j in range(1, u.shape[1] - 1):
            u[i, j] -= dt / rho * (p[i+1, j] - p[i-1, j]) / (2 * dx)
            v[i, j] -= dt / rho * (p[i, j+1] - p[i, j-1]) / (2 * dy)
    
    return u, v

def simulate(u, v, p, dx, dy, dt, t_end, nu, rho, method, save_interval=100, save_dir="sim_data_npz"):
    t = 0.0
    step = 0
    frame = 0

    os.makedirs(save_dir, exist_ok=True)

    if method == 'RK3':
        time_stepper = RK3
    else:
        time_stepper = RK4
    
    while t < t_end:
        # Step 1: RK intermediate velocity (u*, v*)
        u_star, v_star = time_stepper(u, v, dt, dx, dy, nu)

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

        if step % save_interval == 0:
              np.savez_compressed(f"{save_dir}/frame_{frame:04d}.npz", u=u, v=v, p=p)
              frame += 1
    
    return u, v, p

def plot_mesh(Lx, Ly, Nx, Ny, save_path="mesh.png"):
    dx = Lx / Nx
    dy = Ly / Ny

    x = np.linspace(0, Lx, Nx+1)
    y = np.linspace(0, Ly, Ny+1)
    X, Y = np.meshgrid(x, y, indexing='ij')

    plt.figure(figsize=(12, 12))
    ax = plt.gca()
    ax.set_facecolor('lightgray')
    
    # Draw vertical grid lines
    for xi in x:
        plt.plot([xi]*len(y), y, color='black', linewidth=0.5)
    
    # Draw horizontal grid lines
    for yi in y:
        plt.plot(x, [yi]*len(x), color='black', linewidth=0.5)

    # Show node points
    plt.plot(X, Y, 'k.', markersize=2)

    plt.title(f"Structured mesh {Nx}x{Ny}", fontsize=20)
    plt.xlabel("x", fontsize=16)
    plt.ylabel("y", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.gca().set_aspect('equal')
    plt.xlim(0, Lx)
    plt.ylim(0, Ly)
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def velocity_magnitude(Nx, Ny, Lx, Ly, save_dir="sim_data_npz"):
    dx, dy = Lx / Nx, Ly / Ny
    x = np.linspace(0, Lx, Nx+1)
    y = np.linspace(0, Ly, Ny+1)
    X, Y = np.meshgrid(x, y, indexing='ij')

    frame_files = sorted([f for f in os.listdir(save_dir) if f.endswith(".npz")])
    os.makedirs("velocity_magnitude", exist_ok=True)

    for idx, file in enumerate(frame_files):
        data = np.load(os.path.join(save_dir, file))
        u = data["u"]
        v = data["v"]
        speed = np.sqrt(u**2 + v**2)

        plt.figure(figsize=(14, 12))

        # Plot contour
        contour = plt.contourf(X, Y, speed, levels=50, cmap='viridis')  # cmap='plasma'
        cbar = plt.colorbar(contour, label='Velocity Magnitude', ticks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0], fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=14)
        cbar.set_label('Velocity Magnitude', fontsize=16)

        # Quiver plot
        #plt.quiver(X[::2, ::2], Y[::2, ::2], u[::2, ::2], v[::2, ::2], color='white', scale=5)

        # Streamplot
        plt.streamplot(X.T[::2, ::2], Y.T[::2, ::2], u.T[::2, ::2], v.T[::2, ::2],
            color='white',       # Color of streamlines
            linewidth=1,         # Thickness of lines
            density=1.5,         # Controls how many lines are drawn
            arrowsize=1,         # Size of arrows on streamlines
            arrowstyle='->'      # Arrow style (can be '-' for no arrowheads)
        )
        
        # Titles and labels
        plt.title(f"Velocity field (t = {((idx + 1) * dt * 100):.1f})", fontsize=20)
        plt.xlabel("x", fontsize=16)
        plt.ylabel("y", fontsize=16)

        # Tick labels
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)

        # Set the aspect ratio to be equal
        plt.axis('equal')
        
        # Set consistent axis limits
        plt.xlim(0, Lx)
        plt.ylim(0, Ly)

        plt.tight_layout()

        # Save each frame
        plt.savefig(f"velocity_magnitude/vel_mag_{idx:04d}.png")
        plt.close()
    
    print("    > Velocity field complete. Frames saved in ./velocity_magnitude/") 

def pressure_isolines(Nx, Ny, Lx, Ly, save_dir="sim_data_npz"):
    dx, dy = Lx / Nx, Ly / Ny
    x = np.linspace(0, Lx, Nx+1)
    y = np.linspace(0, Ly, Ny+1)
    X, Y = np.meshgrid(x, y, indexing='ij')

    frame_files = sorted([f for f in os.listdir(save_dir) if f.endswith(".npz")])
    os.makedirs("pressure_isolines", exist_ok=True)

    for idx, file in enumerate(frame_files):
        data = np.load(os.path.join(save_dir, file))
        p = data["p"]

        plt.figure(figsize=(14, 12))

        # Filled background pressure field
        filled = plt.contourf(X, Y, p, levels=50, cmap='coolwarm')
        cbar = plt.colorbar(filled, label='Pressure', fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=14)
        cbar.set_label('Pressure', fontsize=16)

        # Pressure isolines
        lines = plt.contour(X, Y, p, levels=20, colors='black', linewidths=0.5)
        plt.clabel(lines, inline=True, fontsize=10, fmt="%.2f")

        # Titles and labels
        plt.title(f"Pressure Field & Isolines (t = {((idx + 1) * dt * 100):.1f})", fontsize=20)
        plt.xlabel("x", fontsize=16)
        plt.ylabel("y", fontsize=16)

        # Fix plot scaling
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.axis('equal')
        plt.xlim(0, Lx)
        plt.ylim(0, Ly)
        plt.tight_layout()

        # Save frame
        plt.savefig(f"pressure_isolines/p_isolines_{idx:04d}.png")
        plt.close()
    
    print("    > Pressure field complete. Frames saved in ./pressure_isolines/")

def make_gif(fps=10):
    for folder in os.listdir("."):
        if os.path.isdir(folder):
            frame_files = sorted([f for f in os.listdir(folder) if f.endswith(".png")])
            if frame_files:
                images = []
                for filename in frame_files:
                    filepath = os.path.join(folder, filename)
                    images.append(imageio.v2.imread(filepath))  # Avoid deprecation warning

                output_gif = f"{folder}.gif"
                imageio.mimsave(output_gif, images, fps=fps)
                
                print(f"      > GIF saved as {output_gif}")

############################################################################################################

# Domain parameters
Lx, Ly = 1.0, 1.0
Nx, Ny = 64, 64
dx, dy = Lx / Nx, Ly / Ny

# Time parameters
dt = 0.001
t_end = 5.0
nu = 0.01  # kinematic viscosity
rho = 1.0  # density

# Grid
x = np.linspace(0, Lx, Nx+1)
y = np.linspace(0, Ly, Ny+1)
X, Y = np.meshgrid(x, y, indexing='ij')

# Initialize fields
u = np.zeros((Nx+1, Ny+1))  # x-velocity
v = np.zeros((Nx+1, Ny+1))  # y-velocity
p = np.zeros((Nx+1, Ny+1))  # pressure

# Check for existing .npz simulation data
data_files_exist = os.path.exists("sim_data_npz") and any(f.endswith(".npz") for f in os.listdir("sim_data_npz"))

if data_files_exist:
    print("  > Binary data found, skipping asimulation and proceeding to postprocessing...")
else:
    print("  > No data found, running simulation...")
    u, v = apply_boundary_conditions(u, v)
    u_final, v_final, p_final = simulate(u, v, p, dx, dy, dt, t_end, nu, rho, method='RK4')

# POSTPROCESSING ###########################################################################################
plot_mesh(Lx, Ly, Nx, Ny)
velocity_magnitude(Nx, Ny, Lx, Ly)
pressure_isolines(Nx, Ny, Lx, Ly)
make_gif(fps=10)