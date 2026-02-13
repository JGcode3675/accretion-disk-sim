import numpy as np
from scipy.integrate import odeint
from vispy import app, scene
from vispy.geometry import create_sphere


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


# --- Custom Colormap ---
def accretion_colormap(values):
    """Map normalized [0,1] values to RGBA. Dark red → orange → yellow → white → blue-white."""
    values = np.clip(np.atleast_1d(values), 0, 1)
    cpoints = np.array([
        [0.0, 0.15, 0.0, 0.0],
        [0.2, 0.70, 0.1, 0.0],
        [0.4, 1.00, 0.4, 0.0],
        [0.6, 1.00, 0.8, 0.2],
        [0.8, 1.00, 1.0, 0.9],
        [1.0, 0.70, 0.85, 1.0],
    ])
    colors = np.ones((len(values), 4))
    for ch in range(3):
        colors[:, ch] = np.interp(values, cpoints[:, 0], cpoints[:, ch + 1])
    return colors.astype(np.float32)


def norm_velocity(v):
    """Normalize velocity values to [0,1] range."""
    return np.clip((v - v_min) / (v_max - v_min + 1e-30), 0, 1)


# --- Vispy Scene Setup ---
canvas = scene.SceneCanvas(
    keys='interactive', size=(1400, 1000), bgcolor='black',
    title='Relativistic Accretion Disk'
)
view = canvas.central_widget.add_view()

plot_limit = r_outer * 1.2 / r_s
view.camera = scene.TurntableCamera(
    elevation=25, azimuth=-60,
    distance=plot_limit * 2.5, fov=45
)

# --- Black hole shadow sphere (2.6 r_s) ---
sphere_data = create_sphere(rows=30, cols=30, radius=2.6)
black_hole = scene.visuals.Mesh(
    vertices=sphere_data.get_vertices(),
    faces=sphere_data.get_faces(),
    color=(0.02, 0.02, 0.02, 1.0),
    parent=view.scene
)

# --- Photon ring at shadow edge ---
theta_ring = np.linspace(0, 2 * np.pi, 200)
ring_pos = np.column_stack([
    2.6 * np.cos(theta_ring),
    2.6 * np.sin(theta_ring),
    np.zeros(200)
])
photon_ring = scene.visuals.Line(
    pos=ring_pos, color=(1.0, 0.82, 0.5, 0.9),
    width=2.5, parent=view.scene, method='gl'
)

# --- ISCO ring ---
isco_r = r_isco / r_s
isco_pos = np.column_stack([
    isco_r * np.cos(theta_ring),
    isco_r * np.sin(theta_ring),
    np.zeros(200)
])
isco_ring = scene.visuals.Line(
    pos=isco_pos, color=(0.0, 1.0, 1.0, 0.3),
    width=1.5, parent=view.scene, method='gl'
)

# --- Particle markers ---
scatter = scene.visuals.Markers(parent=view.scene)

# --- Trail lines (pre-allocated fixed-size buffers to prevent flickering) ---
trail_frames = 15
_max_trail_verts = num_particles * trail_frames * 2  # 2 vertices per segment
trail_pos_buf = np.zeros((_max_trail_verts, 3), dtype=np.float32)
trail_clr_buf = np.zeros((_max_trail_verts, 4), dtype=np.float32)
trails_visual = scene.visuals.Line(
    pos=trail_pos_buf, color=trail_clr_buf, connect='segments',
    parent=view.scene, method='gl', width=1.0, antialias=True
)

# --- Animation state ---
current_frame = [0]


def update(event):
    frame = current_frame[0]

    # Auto-rotate camera
    view.camera.azimuth = -60 + frame * 0.3

    # Particle data for this frame
    positions = trajectories[:, frame, :3]
    vel_xy = trajectories[:, frame, 3:5]
    vels = velocities[:, frame]

    # Filter fallen particles
    r = np.linalg.norm(positions, axis=1)
    mask = r > r_s * 1.1
    active = np.where(mask)[0]
    n_active = len(active)

    if n_active == 0:
        current_frame[0] = (frame + 1) % num_frames
        return

    # Positions in r_s units
    pos_rs = positions[active] / r_s
    r_rs = r[active] / r_s

    # --- Doppler beaming ---
    azim_rad = np.radians(view.camera.azimuth)
    cam_dir = np.array([-np.cos(azim_rad), -np.sin(azim_rad)])
    v_los = vel_xy[active, 0] * cam_dir[0] + vel_xy[active, 1] * cam_dir[1]
    v_los_norm = v_los / (np.max(np.abs(v_los)) + 1e-30)
    doppler_alpha = np.clip(0.5 + 0.5 * v_los_norm, 0.15, 1.0)

    # --- Variable particle size (inner = larger) ---
    r_min_p = r_isco / r_s
    r_max_p = r_outer / r_s
    size_scale = 1.0 - np.clip((r_rs - r_min_p) / (r_max_p - r_min_p), 0, 1)
    sizes = 4 + 18 * size_scale

    # --- Colors from custom colormap + Doppler alpha ---
    v_norm = norm_velocity(vels[active])
    colors = accretion_colormap(v_norm)
    colors[:, 3] = doppler_alpha

    scatter.set_data(
        pos_rs.astype(np.float32),
        face_color=colors, size=sizes,
        edge_width=0, edge_color=None
    )

    # --- Trails (fill pre-allocated buffers, no GPU reallocation) ---
    trail_pos_buf[:] = 0
    trail_clr_buf[:] = 0

    t_start = max(0, frame - trail_frames)
    n_pts = frame - t_start + 1

    if n_pts >= 2:
        n_segs = n_pts - 1
        all_trails = trajectories[active, t_start:frame + 1, :3] / r_s

        starts = all_trails[:, :-1, :].reshape(-1, 3)
        ends = all_trails[:, 1:, :].reshape(-1, 3)
        total = n_active * n_segs
        used = total * 2

        trail_pos_buf[:used:2] = starts
        trail_pos_buf[1:used:2] = ends

        base_colors = accretion_colormap(norm_velocity(vels[active]))
        alphas = (0.03 + 0.25 * np.arange(n_segs) / max(n_segs - 1, 1)).astype(np.float32)

        seg_colors = np.repeat(base_colors, n_segs * 2, axis=0)
        alpha_pattern = np.repeat(alphas, 2)
        alpha_tiled = np.tile(alpha_pattern, n_active)
        seg_colors[:, 3] = alpha_tiled
        trail_clr_buf[:used] = seg_colors

    trails_visual.set_data(pos=trail_pos_buf, color=trail_clr_buf, connect='segments')

    # --- Update window title ---
    time_orbits = t[frame] / T_inner
    canvas.title = (
        f'Accretion Disk (Paczy\u0144ski\u2013Wiita)  |  '
        f't = {time_orbits:.2f} orbits  |  '
        f'{n_active}/{num_particles} particles'
    )

    canvas.update()
    current_frame[0] = (frame + 1) % num_frames


timer = app.Timer(interval=1/30, connect=update, start=True)

print("Animation ready. Displaying...")
print("Controls: drag to rotate, scroll to zoom, middle-click to pan")
canvas.show()
app.run()
