import pyvista as pv
import numpy as np
import time
import math

###############################################################################
#                           Grid and Wave Setup
###############################################################################
n = 50  # grid resolution
x = np.linspace(-5, 5, n)
y = np.linspace(-5, 5, n)
X, Y = np.meshgrid(x, y)
Z = np.zeros_like(X)

# Convert (X, Y, Z) into a point cloud for PyVista
points = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))

# Create a StructuredGrid in PyVista
grid = pv.StructuredGrid()
grid.points = points
# Set dimensions: n x n grid with 1 layer in Z
grid.dimensions = (n, n, 1)

# Wave parameters: (amplitude, initial_phase, frequency, phase_speed)
waves_params = [
    (1.0, 0.0,         1.0, 0.3),
    (0.8, math.pi / 4, 1.0, 0.3),
    (1.2, -math.pi / 6,1.0, 0.3)
]
exaggeration = 2.0  # For extra vertical 'bounce'

def generate_wave(amp, phase, freq, exaggeration):
    """Generate a 2D wave over the global X, Y grids."""
    return exaggeration * amp * np.sin(freq * X + phase) * np.cos(freq * Y + phase)

def compute_superposition(t):
    """Compute the superposition of the three waves at time t."""
    wave_total = np.zeros_like(X)
    for (amp, init_phase, freq, phase_speed) in waves_params:
        phase = init_phase + phase_speed * t
        wave_total += generate_wave(amp, phase, freq, exaggeration)
    return wave_total

###############################################################################
#                           PyVista Visualization Setup
###############################################################################
# Create three copies of the grid for different subplots.
grid_wave = grid.copy()        # For individual wave guess (we'll show the first wave)
grid_super = grid.copy()       # For the superposition
grid_collapsed = grid.copy()   # For the collapsed measurement

# Create a Plotter with 1 row and 3 columns.
plotter = pv.Plotter(shape=(1, 3), window_size=(1800, 600))

# Subplot 1: Individual wave guess using 'viridis' with 60% opacity.
plotter.subplot(0, 0)
mesh_wave = plotter.add_mesh(grid_wave, show_edges=True, cmap='viridis', opacity=0.6)
plotter.add_text("Wave Guess (Wave 1)", position='upper_edge', font_size=12)

# Subplot 2: Superposition using 'cividis'.
plotter.subplot(0, 1)
mesh_super = plotter.add_mesh(grid_super, show_edges=True, cmap='cividis')
plotter.add_text("Superposition", position='upper_edge', font_size=12)

# Subplot 3: Collapsed measurement using 'magma'.
plotter.subplot(0, 2)
mesh_collapsed = plotter.add_mesh(grid_collapsed, show_edges=True, cmap='magma')
plotter.add_text("Collapsed (Magnitude Squared)", position='upper_edge', font_size=12)

# Set z-axis ranges for each subplot via the renderers.
# In a plotter with shape (1,3), plotter.renderers is a list of 3 renderer objects.
plotter.renderers[0].scale.z_axis.range = [-10, 10]
plotter.renderers[1].scale.z_axis.range = [-10, 10]
plotter.renderers[2].scale.z_axis.range = [0, 100]

start_time = time.time()

def update_scene():
    """Update the grid's z-values each frame to animate the process."""
    t = time.time() - start_time
    # Generate dynamic waves: use the first wave for the individual wave subplot.
    wave1 = generate_wave(waves_params[0][0],
                          waves_params[0][1] + waves_params[0][3] * t,
                          waves_params[0][2], exaggeration)
    # Compute the superposition (sum of all waves).
    superposition = compute_superposition(t)
    # Compute collapsed measurement: squared magnitude of the superposition.
    collapsed = superposition ** 2
    
    # Update the z-coordinates of each grid.
    grid_wave.points[:, 2] = wave1.ravel()
    grid_super.points[:, 2] = superposition.ravel()
    grid_collapsed.points[:, 2] = collapsed.ravel()
    
    # Trigger a redraw.
    plotter.update()

# Use a timer callback to update the scene at 20 ms intervals (~50 FPS).
plotter.add_timer_callback(update_scene, 20)

plotter.show()