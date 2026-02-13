import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
from scipy.integrate import odeint


# --- Constants ---
G = 6.67430e-11      # Gravitational constant (m^3 kg^-1 s^-2)
c = 2.998e8          # Speed of light (m/s)
M_sun = 1.989e30     # Solar mass (kg)
M = 10 * M_sun       # Black hole mass (10 solar masses for stronger effects)

# Schwarzschild radius
r_s = 2 * G * M / c**2

# Innermost Stable Circular Orbit (ISCO) for Schwarzschild black hole
r_isco = 3 * r_s


# --- Paczyński-Wiita Pseudo-Newtonian Potential ---
# This approximates general relativistic effects near black holes
# Potential: Φ = -GM / (r - r_s)
# This gives the correct ISCO at 3*r_s and captures perihelion precession

def equations_of_motion_pw_3d(y, _t, G, M, r_s):
    """
    3D Equations of motion using Paczyński-Wiita potential.
    Approximates GR effects including:
    - Correct ISCO location
    - Perihelion precession
    - Unstable orbits inside ISCO
    """
    x, y_pos, z, vx, vy, vz = y
    r_cyl = np.sqrt(x**2 + y_pos**2)  # Cylindrical radius
    r = np.sqrt(x**2 + y_pos**2 + z**2)  # Spherical radius

    # Prevent singularity at r = r_s
    if r <= r_s * 1.01:
        return [0, 0, 0, 0, 0, 0]  # Particle has fallen in

    # Paczyński-Wiita acceleration: a = -GM / (r - r_s)^2 * (r_hat)
    # Using spherical r for the potential
    factor = G * M / (r - r_s)**2
    ax = -factor * x / r
    ay = -factor * y_pos / r
    az = -factor * z / r

    return [vx, vy, vz, ax, ay, az]


# --- Disk Initialization with Keplerian Orbits ---
num_particles = 800
np.random.seed(42)  # For reproducibility

def initialize_disk_particles_3d(num_particles, r_inner, r_outer, G, M, r_s):
    """
    Initialize particles in a 3D accretion disk configuration.
    - Radial distribution follows r^(-1/2) (surface density profile)
    - Velocities are Keplerian (circular orbits)
    - Disk has vertical thickness that increases with radius
    - Small random perturbations for realism
    """
    particles = []

    # Disk scale height parameter (H/R ratio, typically 0.01-0.1 for thin disks)
    h_over_r = 0.05

    for _ in range(num_particles):
        # Radial distribution: more particles at smaller radii
        u = np.random.uniform(0, 1)
        r = r_inner + (r_outer - r_inner) * u**0.5

        # Random angle
        theta = np.random.uniform(0, 2 * np.pi)

        # Disk thickness increases with radius (scale height H = h_over_r * r)
        scale_height = h_over_r * r
        z = np.random.normal(0, scale_height * 0.5)

        # Position
        x = r * np.cos(theta)
        y_pos = r * np.sin(theta)

        # Keplerian velocity for Paczyński-Wiita potential
        v_kepler = np.sqrt(G * M * r / (r - r_s)**2)

        # Add small random perturbation (5% variation)
        v = v_kepler * np.random.uniform(0.95, 1.05)

        # Tangential velocity (perpendicular to radius, prograde)
        vx = -v * np.sin(theta)
        vy = v * np.cos(theta)
        vz = np.random.normal(0, v_kepler * 0.01)  # Small vertical velocity

        # Add small radial velocity perturbation
        v_radial = v_kepler * np.random.uniform(-0.02, 0.02)
        vx += v_radial * np.cos(theta)
        vy += v_radial * np.sin(theta)

        particles.append([x, y_pos, z, vx, vy, vz])

    return np.array(particles)


# Initialize disk between ISCO and 50 * r_s
r_inner = r_isco * 1.5  # Start just outside ISCO for stability
r_outer = r_s * 80
initial_conditions = initialize_disk_particles_3d(num_particles, r_inner, r_outer, G, M, r_s)


# --- Time Configuration ---
# Orbital period at r_inner for reference
T_inner = 2 * np.pi * np.sqrt(r_inner**3 / (G * M))
t_max = T_inner * 5  # Simulate for 5 orbital periods of inner disk
num_frames = 300
t = np.linspace(0, t_max, num_frames)


# --- Run Simulation ---
print(f"Black hole mass: {M/M_sun:.1f} solar masses")
print(f"Schwarzschild radius: {r_s:.2e} m ({r_s/1000:.1f} km)")
print(f"ISCO radius: {r_isco:.2e} m ({r_isco/r_s:.1f} r_s)")
print(f"Simulating {num_particles} particles...")
print(f"Simulation time: {t_max:.2e} s ({t_max/T_inner:.1f} inner orbital periods)")

trajectories = []
for i in range(num_particles):
    sol = odeint(equations_of_motion_pw_3d, initial_conditions[i], t, args=(G, M, r_s))
    trajectories.append(sol)
    if (i + 1) % 50 == 0:
        print(f"  Computed {i + 1}/{num_particles} particle trajectories")

trajectories = np.array(trajectories)
print("Simulation complete. Generating animation...")


# --- Calculate velocities for coloring ---
def calculate_velocities_3d(trajectories):
    """Calculate velocity magnitude at each timestep for coloring."""
    velocities = np.sqrt(trajectories[:, :, 3]**2 + trajectories[:, :, 4]**2 + trajectories[:, :, 5]**2)
    return velocities

velocities = calculate_velocities_3d(trajectories)
v_min, v_max = np.percentile(velocities, [5, 95])


# --- 3D Animation Setup ---
plt.style.use('dark_background')
fig = plt.figure(figsize=(12, 10), dpi=120)
ax = fig.add_subplot(111, projection='3d')

# Custom accretion disk colormap: dark red → orange → yellow → white → blue-white
disk_colors = [
    (0.0, (0.15, 0.0, 0.0)),     # deep dark red
    (0.2, (0.7, 0.1, 0.0)),      # dark red-orange
    (0.4, (1.0, 0.4, 0.0)),      # orange
    (0.6, (1.0, 0.8, 0.2)),      # yellow
    (0.8, (1.0, 1.0, 0.9)),      # white-hot
    (1.0, (0.7, 0.85, 1.0)),     # blue-white
]
cmap = LinearSegmentedColormap.from_list('accretion', disk_colors, N=256)
norm = Normalize(vmin=v_min, vmax=v_max)
sm = ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

# Plot limits (in r_s units)
plot_limit = r_outer * 1.2 / r_s
z_limit = r_outer * 0.15 / r_s  # Smaller z range to emphasize disk thinness

# Viewing angle (elevation and azimuth)
view_elev = 25  # Degrees above the disk plane
view_azim = -60  # Rotation around z-axis


def draw_solid_sphere(ax, radius_rs, color, alpha=0.95):
    """Draw a solid sphere for the black hole shadow (radius in r_s units)."""
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 20)
    x = radius_rs * np.outer(np.cos(u), np.sin(v))
    y = radius_rs * np.outer(np.sin(u), np.sin(v))
    z = radius_rs * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, color=color, alpha=alpha, shade=False)


def draw_circle_3d(ax, radius_rs, color, linestyle='-', alpha=0.5, z_offset=0, linewidth=1):
    """Draw a circle in the xy-plane (radius in r_s units)."""
    theta = np.linspace(0, 2 * np.pi, 100)
    x = radius_rs * np.cos(theta)
    y = radius_rs * np.sin(theta)
    z = np.full_like(theta, z_offset)
    ax.plot(x, y, z, color=color, linestyle=linestyle, alpha=alpha, linewidth=linewidth)


def init():
    ax.set_xlim(-plot_limit, plot_limit)
    ax.set_ylim(-plot_limit, plot_limit)
    ax.set_zlim(-z_limit, z_limit)
    ax.set_xlabel('x / $r_s$', fontsize=10, labelpad=8)
    ax.set_ylabel('y / $r_s$', fontsize=10, labelpad=8)
    ax.set_zlabel('z / $r_s$', fontsize=10, labelpad=8)
    ax.view_init(elev=view_elev, azim=view_azim)
    return []


def update(frame):
    ax.clear()
    ax.set_xlim(-plot_limit, plot_limit)
    ax.set_ylim(-plot_limit, plot_limit)
    ax.set_zlim(-z_limit, z_limit)

    current_azim = view_azim + frame * 0.3  # Slow rotation
    ax.view_init(elev=view_elev, azim=current_azim)

    # --- Black hole shadow (solid dark sphere at ~2.6 r_s) ---
    draw_solid_sphere(ax, 2.6, 'black', alpha=0.95)

    # --- Photon ring (bright thin ring at shadow edge) ---
    draw_circle_3d(ax, 2.6, '#FFD080', linestyle='-', alpha=0.8, linewidth=1.5)

    # --- ISCO circle ---
    draw_circle_3d(ax, r_isco / r_s, 'cyan', linestyle='--', alpha=0.4)

    # --- Get particle positions (in meters) ---
    positions_x = trajectories[:, frame, 0]
    positions_y = trajectories[:, frame, 1]
    positions_z = trajectories[:, frame, 2]
    vel_x = trajectories[:, frame, 3]
    vel_y = trajectories[:, frame, 4]
    vels = velocities[:, frame]

    # Filter out particles that have fallen in
    r = np.sqrt(positions_x**2 + positions_y**2 + positions_z**2)
    mask = r > r_s * 1.1

    # Convert positions to r_s units for plotting
    px = positions_x[mask] / r_s
    py = positions_y[mask] / r_s
    pz = positions_z[mask] / r_s
    r_masked = r[mask] / r_s

    # --- Doppler beaming ---
    # Camera direction in the xy-plane based on rotating azimuth
    azim_rad = np.radians(current_azim)
    cam_dir_x = -np.cos(azim_rad)
    cam_dir_y = -np.sin(azim_rad)

    # Line-of-sight velocity component (positive = approaching camera)
    v_los = vel_x[mask] * cam_dir_x + vel_y[mask] * cam_dir_y
    v_los_norm = v_los / (np.max(np.abs(v_los)) + 1e-30)
    # Boost: approaching side brighter (alpha 0.5→1.0), receding dimmer (0.3→0.5)
    doppler_alpha = 0.5 + 0.5 * v_los_norm
    doppler_alpha = np.clip(doppler_alpha, 0.2, 1.0)

    # --- Variable particle size (inner = larger/brighter) ---
    r_min_plot = r_isco / r_s
    r_max_plot = r_outer / r_s
    size_scale = 1.0 - np.clip((r_masked - r_min_plot) / (r_max_plot - r_min_plot), 0, 1)
    particle_sizes = 6 + 30 * size_scale  # Range: 6 (outer) to 36 (inner)

    # --- Plot particles colored by velocity, sized by radius, alpha by Doppler ---
    base_colors = cmap(norm(vels[mask]))
    base_colors[:, 3] = doppler_alpha  # Modulate alpha channel
    ax.scatter(px, py, pz, c=base_colors, s=particle_sizes, depthshade=True,
               edgecolors='none')

    # --- Draw trailing paths with fade gradient ---
    trail_length = min(15, frame)
    if trail_length > 1:
        for i in range(num_particles):
            if mask[i]:
                t_start = max(0, frame - trail_length)
                trail_x = trajectories[i, t_start:frame + 1, 0] / r_s
                trail_y = trajectories[i, t_start:frame + 1, 1] / r_s
                trail_z = trajectories[i, t_start:frame + 1, 2] / r_s
                trail_color = cmap(norm(vels[i]))
                n_seg = len(trail_x) - 1
                for seg in range(n_seg):
                    seg_alpha = 0.05 + 0.35 * (seg / max(n_seg - 1, 1))
                    ax.plot(trail_x[seg:seg + 2], trail_y[seg:seg + 2],
                            trail_z[seg:seg + 2], color=trail_color,
                            alpha=seg_alpha, linewidth=0.8)

    # --- Labels and title ---
    ax.set_xlabel('x / $r_s$', fontsize=10, labelpad=8)
    ax.set_ylabel('y / $r_s$', fontsize=10, labelpad=8)
    ax.set_zlabel('z / $r_s$', fontsize=10, labelpad=8)
    time_orbits = t[frame] / T_inner
    particles_remaining = np.sum(mask)
    ax.set_title(f'Relativistic Accretion Disk  (Paczy\u0144ski\u2013Wiita)\n'
                 f't = {time_orbits:.2f} orbits   |   '
                 f'{particles_remaining}/{num_particles} particles',
                 fontsize=11, color='#cccccc')

    # --- Clean dark background ---
    ax.set_facecolor('black')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor((1, 1, 1, 0.05))
    ax.yaxis.pane.set_edgecolor((1, 1, 1, 0.05))
    ax.zaxis.pane.set_edgecolor((1, 1, 1, 0.05))
    ax.grid(False)

    return []


# --- Create Animation ---
anim = FuncAnimation(fig, update, frames=num_frames, init_func=init,
                     blit=False, interval=50)

# Add colorbar
cbar = fig.colorbar(sm, ax=ax, label='Velocity (m/s)', shrink=0.6, pad=0.1)

plt.tight_layout()
print("Animation ready. Displaying...")
plt.show()


# --- Optional: Save animation ---
# Uncomment to save as MP4 (requires ffmpeg) or GIF
# anim.save('accretion_disk_3d.mp4', writer='ffmpeg', fps=20, dpi=150)
# anim.save('accretion_disk_3d.gif', writer='pillow', fps=20, dpi=100)
