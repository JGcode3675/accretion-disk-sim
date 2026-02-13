# Accretion Disk Simulation

3D relativistic accretion disk simulation around a 10 solar mass black hole, built with Python and Vispy (GPU-accelerated OpenGL).

## Physics

- **Paczyński-Wiita pseudo-Newtonian potential** — approximates general relativistic effects (correct ISCO, perihelion precession, unstable orbits)
- **Keplerian orbital initialization** with radial density profile and vertical scale height
- **800 particles** integrated with `scipy.integrate.odeint`

## Visualization

- Custom colormap (dark red → orange → yellow → white → blue-white)
- **Doppler beaming** — approaching side of the disk appears brighter
- **Variable particle size** — inner particles are larger/more intense
- **Fading trails** — transparent at the tail, semi-opaque at the head
- Solid black hole shadow with photon ring at ~2.6 r_s
- Positions normalized to Schwarzschild radii (r_s)
- Interactive camera — drag to rotate, scroll to zoom, middle-click to pan

## Requirements

```
numpy
scipy
vispy
PyOpenGL
PyQt6
```

## Usage

```bash
python3 accretion_disk.py
```

The animation window will open with a slowly rotating camera. Drag to rotate, scroll to zoom, middle-click to pan. Close the window to exit.

## Parameters

| Parameter       | Default  | Description              |
| --------------- | -------- | ------------------------ |
| `M`             | 10 M_sun | Black hole mass          |
| `num_particles` | 800      | Number of disk particles |
| `r_inner`       | 4.5 r_s  | Inner disk edge          |
| `r_outer`       | 80 r_s   | Outer disk edge          |
| `num_frames`    | 300      | Animation frames         |
| `view_elev`     | 25°      | Camera elevation         |
