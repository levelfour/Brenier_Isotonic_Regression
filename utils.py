import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import matplotlib.tri as tri
from matplotlib.colors import Normalize
import matplotlib.cm as cm


def barycentric_to_cartesian(barycentric_coords):
    """Convert barycentric coordinates to 2D cartesian coordinates."""
    vertices = np.array([[0, 0],           # vertex for p3
                        [1, 0],           # vertex for p2  
                        [0.5, np.sqrt(3)/2]])  # vertex for p1
    
    cartesian = np.dot(barycentric_coords, vertices)
    return cartesian

def create_simplex_grid(resolution=50):
    """Create a grid of points on the probability simplex."""
    coords = []
    for i in range(resolution + 1):
        for j in range(resolution + 1 - i):
            k = resolution - i - j
            p1, p2, p3 = i/resolution, j/resolution, k/resolution
            coords.append([p1, p2, p3])
    
    simplex_points = np.array(coords)
    cart_points = barycentric_to_cartesian(simplex_points)
    triangulation = tri.Triangulation(cart_points[:, 0], cart_points[:, 1])
    
    return simplex_points, triangulation, cart_points

def add_simplex_boundary_and_labels(ax):
    """Add triangle boundary and vertex labels to simplex plot."""
    # Plot the triangle boundary
    triangle = Polygon([[0, 0], [1, 0], [0.5, np.sqrt(3)/2]], 
                      fill=False, edgecolor='black', linewidth=2)
    ax.add_patch(triangle)
    
    # Set equal aspect ratio and clean up
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, np.sqrt(3)/2 + 0.1)

def plot_calibration_map(mapping_func, name, resolution=50, reduced=False):
    """
    Create a four-panel visualization:
    1-3: Contour plots for each output coordinate
    4: Calibration map
    
    Args:
        mapping_func: function that takes (n, 3) array and returns (n, 3) array
        name: calibrator name
        resolution: resolution of the simplex grid for contour plots
        reduced: show counter plot of 1st coordinate only
    """
    if reduced:
        fig, axes = plt.subplots(1, 2, figsize=(6, 3),
                                 gridspec_kw={
                                     'width_ratios': [1, 1.25],
                                     'height_ratios': [1]
                                 })
    else:
        fig, axes = plt.subplots(1, 4, figsize=(13, 3),
                                 gridspec_kw={
                                     'width_ratios': [1, 1.25, 1.25, 1.25],
                                     'height_ratios': [1]
                                 })
    
    # Create simplex grid for contour plots
    simplex_points, triangulation, cart_points = create_simplex_grid(resolution)
    mapped_points = mapping_func(simplex_points)
    
    # Subplot 1: Calibration map as vector field (bottom-right)
    ax = axes[0]
    
    # Create a coarser grid for vector field arrows
    arrow_resolution = 15  # Adjust this for arrow density
    arrow_simplex_points, _, arrow_cart_points = create_simplex_grid(arrow_resolution)
    arrow_mapped_points = mapping_func(arrow_simplex_points)
    arrow_mapped_cart = barycentric_to_cartesian(arrow_mapped_points)
    
    # Calculate displacement vectors in cartesian space
    displacement = arrow_mapped_cart - arrow_cart_points
    
    # Add simplex boundary and labels
    add_simplex_boundary_and_labels(ax)
    
    # Plot vector field with colored arrows
    # Color arrows by the class they're moving toward (dominant component)
    colors = ['red', 'green', 'blue']
    
    for i in range(len(arrow_cart_points)):
        start_point = arrow_cart_points[i]
        end_point = arrow_mapped_cart[i]
        displacement_vec = displacement[i]
        
        # Skip very small displacements to avoid clutter
        if np.linalg.norm(displacement_vec) > 0.01:
            # Color by the dominant component in the OUTPUT
            dominant_class = np.argmax(arrow_mapped_points[i])
            arrow_color = colors[dominant_class]
            
            # Plot arrow
            ax.arrow(start_point[0], start_point[1], 
                    displacement_vec[0], displacement_vec[1],
                    head_width=0.03, head_length=0.03, 
                    fc=arrow_color, ec=arrow_color, alpha=0.2, linewidth=1)
    
    # Add a light grid for reference
    # Create grid lines in barycentric coordinates
    grid_resolution = 10
    for i in range(1, grid_resolution):
        # Horizontal lines (constant p1)
        p1_val = i / grid_resolution
        p2_vals = np.linspace(0, 1 - p1_val, 20)
        p3_vals = 1 - p1_val - p2_vals
        
        # Only keep valid probability vectors
        valid_mask = (p2_vals >= 0) & (p3_vals >= 0) & (p2_vals <= 1) & (p3_vals <= 1)
        if np.any(valid_mask):
            grid_points = np.column_stack([np.full(np.sum(valid_mask), p1_val), 
                                         p2_vals[valid_mask], p3_vals[valid_mask]])
            grid_cart = barycentric_to_cartesian(grid_points)
            ax.plot(grid_cart[:, 0], grid_cart[:, 1], 'lightgray', alpha=0.3, linewidth=0.5)

    ax.set_title(f'Calibration map ({name})', fontsize=14)

    # Subplot 2-4: Contour plots for each coordinate
    colormaps = ['Reds', 'Greens', 'Blues']

    for comp_idx in range(1) if reduced else range(3):
        ax = axes[1 + comp_idx]

        # Get the mapped component values at each grid point
        component_values = mapped_points[:, comp_idx]

        # Create contour plot
        n_levels = 30
        levels = np.linspace(0-1e-8, 1+1e-8, n_levels)
        contour = ax.tricontourf(triangulation, component_values, 
                                levels=levels, cmap=colormaps[comp_idx],
                                alpha=0.8,
                                vmin=0, vmax=1, extend='neither')
        contour_lines = ax.tricontour(triangulation, component_values, 
                                     levels=8, colors='white', alpha=0.6, linewidths=0.5)

        # Add colorbar
        contour.set_clim(0, 1)
        cbar = plt.colorbar(contour, ax=ax, shrink=0.8)
        cbar.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        cbar.set_ticklabels(['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'])

        # Add simplex boundary and labels
        add_simplex_boundary_and_labels(ax)

        # Set title
        ax.set_title(f'$\\eta_{comp_idx+1}$ ({name})', fontsize=14)

    plt.tight_layout()
    return fig, axes
