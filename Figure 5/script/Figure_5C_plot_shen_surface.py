import os
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn import datasets, surface, plotting
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
from matplotlib.path import Path
from matplotlib.patches import PathPatch

# 1. Setup Paths and Data
csv_path = '../data/STA_diffOfDiff_sigparcels_FDR0050_Shen.csv'
atlas_path = '../data/shen_1mm_268_parcellation.nii.gz'
output_dir = '../results'
os.makedirs(output_dir, exist_ok=True)

# =============================================================================
# HYPERPARAMETERS - Adjust these to tune the visualization
# =============================================================================
# T-value filtering: 'positive', 'negative', or 'all'
T_VALUE_FILTER = 'negative'          # 'positive' = t > 0, 'negative' = t < 0, 'all' = no filter

# Legend appearance
LEGEND_HEIGHT_FACTOR = 0.6      # Height of legend as fraction of figure (0.5 to 1.0)
CIRCLE_ASPECT_RATIO = 3.8       # Set to a number (e.g., 3.0) to override auto-calculation
                                # Higher = wider circles, Lower = taller circles
                                # None = auto-calculate based on figure dimensions
CIRCLE_SIZE_FACTOR = 0.40       # Circle radius as fraction of row spacing (0.2 to 0.5)
LEGEND_WIDTH_RATIO = 0.4        # Width ratio for legend column in gridspec
LEGEND_VERTICAL_OFFSET = 0.08   # Push legend up (positive) or down (negative)
# =============================================================================

network_colors = {
    'VAs': {'dark': '#b31902', 'light': '#ff2200'},  
    'VI':  {'dark': '#b37202', 'light': '#fca103'},  
    'FP':  {'dark': '#b5b500', 'light': '#fcfc03'},  
    'SAL': {'dark': '#76b500', 'light': '#a7ff03'}, 
    'SC':  {'dark': '#bf0270', 'light': '#fc0394'}, 
    'MF':  {'dark': '#0267ba', 'light': '#0089fa'}, 
    'Mot': {'dark': '#7b02c2', 'light': '#9e00fa'},
    'DMN': {'dark': '#b600bd', 'light': '#f200fa'},
    'CBL': {'dark': '#00a1a1', 'light': '#00fafa'}  
}


# 3. Process CSV
df = pd.read_csv(csv_path)

# Filter by t-value sign if specified
if T_VALUE_FILTER == 'positive':
    df = df[df['t_value'] > 0]
    output_suffix = '_positive_t'
    print(f"Filtering for positive t-values: {len(df)} nodes")
elif T_VALUE_FILTER == 'negative':
    df = df[df['t_value'] < 0]
    output_suffix = '_negative_t'
    print(f"Filtering for negative t-values: {len(df)} nodes")
else:
    output_suffix = ''
    print(f"Using all t-values: {len(df)} nodes")

# We want a structured colormap where indices are grouped by network
# 0: Background
# 1, 2: Net1 Dark, Net1 Light
# 3, 4: Net2 Dark, Net2 Light
# ...
color_map_keys = ['#eeeeee'] # 0 is background
network_to_indices = {} # network -> {'dark': idx, 'light': idx}

sorted_networks = sorted(list(network_colors.keys()))
current_idx = 1
for net in sorted_networks:
    # Add dark then light
    color_map_keys.append(network_colors[net]['dark'])
    dark_idx = current_idx
    current_idx += 1
    
    color_map_keys.append(network_colors[net]['light'])
    light_idx = current_idx
    current_idx += 1
    
    network_to_indices[net] = {'dark': dark_idx, 'light': light_idx}

node_to_color_id = {}

for _, row in df.iterrows():
    node = int(row['node'])
    network = row['label']
    p_val = row['p_value_FDR']
    
    if p_val >= 0.05:
        continue

    if network not in network_to_indices:
        print(f"Warning: Network {network} not in color palette. Skipping node {node}.")
        continue
        
    shade = 'dark' if p_val < 0.01 else 'light'
    cid = network_to_indices[network][shade]
    node_to_color_id[node] = cid

print(f"Created colormap with {len(color_map_keys)} colors (including background).")

# Create the custom matplotlib colormap
custom_cmap = ListedColormap(color_map_keys)

# 4. Load Atlas and Project to Surface
print("Fetching fsaverage...")
fsaverage = datasets.fetch_surf_fsaverage('fsaverage')

print("Projecting atlas to surface (this may take a moment)...")
# We use 'nearest_most_frequent' interpolation because atlas contains discrete labels
texture_left = surface.vol_to_surf(atlas_path, fsaverage.pial_left, interpolation='nearest_most_frequent')
texture_right = surface.vol_to_surf(atlas_path, fsaverage.pial_right, interpolation='nearest_most_frequent')

# 5. Create Visualization Data
def map_texture(texture, mapping):
    new_texture = np.zeros_like(texture, dtype=int)
    unique_nodes = np.unique(texture)
    for node in unique_nodes:
        if node in mapping:
            mask = (texture == node)
            new_texture[mask] = mapping[node]
    return new_texture

print("Mapping colors to surface...")
plot_data_left = map_texture(texture_left, node_to_color_id)
plot_data_right = map_texture(texture_right, node_to_color_id)

# 6. Plotting
modes = [
    ('left', 'lateral', fsaverage.infl_left, fsaverage.sulc_left, plot_data_left, texture_left),
    ('left', 'medial', fsaverage.infl_left, fsaverage.sulc_left, plot_data_left, texture_left),
    ('right', 'lateral', fsaverage.infl_right, fsaverage.sulc_right, plot_data_right, texture_right),
    ('right', 'medial', fsaverage.infl_right, fsaverage.sulc_right, plot_data_right, texture_right)
]

# Adjust spacing
wspace = 0.05
hspace = 0.1
fig = plt.figure(figsize=(14, 10))
gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, LEGEND_WIDTH_RATIO], wspace=wspace, hspace=hspace)

# Plot Brains in first 2 columns
axes = []
axes.append(fig.add_subplot(gs[0, 0], projection='3d'))
axes.append(fig.add_subplot(gs[0, 1], projection='3d'))
axes.append(fig.add_subplot(gs[1, 0], projection='3d'))
axes.append(fig.add_subplot(gs[1, 1], projection='3d'))

for i, (hemi, view, mesh, bg_map, data, original_labels) in enumerate(modes):
    print(f"Plotting {hemi} {view}...")
    
    plotting.plot_surf(
        mesh,
        surf_map=data,
        hemi=hemi,
        view=view,
        cmap=custom_cmap,
        vmin=0,
        vmax=len(color_map_keys)-1,
        threshold=0.01,
        bg_map=bg_map,
        bg_on_data=True,
        darkness=None,
        colorbar=False,
        axes=axes[i],
        title=f"{hemi.capitalize()} {view.capitalize()}"
    )
    
    # Add contours
    unique_labels = np.unique(original_labels)
    unique_labels = unique_labels[unique_labels != 0]
    
    if len(unique_labels) > 0:
        for label in unique_labels:
            mask = (original_labels == label)
            if np.sum(mask) < 10: 
                continue
            try:
                plotting.plot_surf_contours(
                    mesh,
                    roi_map=original_labels,
                    axes=axes[i],
                    levels=[label],
                    colors=['k'],
                    legend=False
                )
            except ValueError:
                continue

# Create Custom Legend in the 3rd column
# We merge the 3rd column cells to make one long column for the legend
ax_legend = fig.add_subplot(gs[:, 2])
ax_legend.axis('off')

# Create legend handles
legend_handles = []
# We want to display: [Dark Color] [Light Color] Network Name
# Matplotlib legend usually takes one handle per text.
# We can use a custom handle (TupleHandler) or just list them vertically.
# Or better, draw rectangles manually on the axis for full control.

print("Creating custom legend...")

# Drawing manual legend on ax_legend
ax_legend.set_xlim(0, 1)
ax_legend.set_ylim(0, 1 + LEGEND_VERTICAL_OFFSET + 0.05)  # Extend to fit offset

# Calculate aspect ratio to make circles look circular
fig_width, fig_height = 14, 10
total_ratio = 1 + 1 + LEGEND_WIDTH_RATIO
legend_width_inches = fig_width * (LEGEND_WIDTH_RATIO / total_ratio)
legend_height_inches = fig_height * LEGEND_HEIGHT_FACTOR

# Use manual override if provided, otherwise auto-calculate
if CIRCLE_ASPECT_RATIO is not None:
    aspect_ratio = CIRCLE_ASPECT_RATIO
else:
    aspect_ratio = legend_height_inches / legend_width_inches

print(f"Legend settings: height_factor={LEGEND_HEIGHT_FACTOR}, aspect_ratio={aspect_ratio:.2f}, circle_size={CIRCLE_SIZE_FACTOR}")

def make_semicircle_path(cx, cy, radius_y, aspect, left=True):
    """Create a semicircle path with proper aspect ratio correction."""
    radius_x = radius_y * aspect  # Stretch x to compensate for tall/narrow axes
    n_points = 50
    if left:
        theta = np.linspace(np.pi/2, 3*np.pi/2, n_points)
    else:
        theta = np.linspace(-np.pi/2, np.pi/2, n_points)
    
    x = cx + radius_x * np.cos(theta)
    y = cy + radius_y * np.sin(theta)
    
    # Create closed path: arc + line back to center + close
    vertices = [(x[0], y[0])]
    for i in range(1, len(x)):
        vertices.append((x[i], y[i]))
    vertices.append((cx, cy))  # back to center
    vertices.append((x[0], y[0]))  # close
    
    codes = [Path.MOVETO] + [Path.LINETO] * (len(vertices) - 1)
    return Path(vertices, codes)

# Calculate layout based on number of networks
n_networks = len(sorted_networks)
top_margin = 0.85
bottom_margin = 0.05
available_height = top_margin - bottom_margin
spacing = available_height / n_networks

# Circle dimensions
radius_y = spacing * CIRCLE_SIZE_FACTOR  # Visual radius in y direction
radius_x = radius_y * aspect_ratio  # Adjusted x radius for circular appearance
center_x = 0.20

ax_legend.text(0.0, 0.95 + LEGEND_VERTICAL_OFFSET, "Network Significance", fontsize=12, fontweight='bold')
ax_legend.text(0.0, 0.89 + LEGEND_VERTICAL_OFFSET, "Left: p < 0.01\nRight: 0.01 < p < 0.05", fontsize=9, fontstyle='italic')

y_pos = top_margin + LEGEND_VERTICAL_OFFSET - spacing / 2
for net in sorted_networks:
    c_dark = network_colors[net]['dark']
    c_light = network_colors[net]['light']
    
    # Draw left semi-circle (dark)
    path_dark = make_semicircle_path(center_x, y_pos, radius_y, aspect_ratio, left=True)
    patch_dark = PathPatch(path_dark, facecolor=c_dark, edgecolor='black', linewidth=0.5)
    ax_legend.add_patch(patch_dark)
    
    # Draw right semi-circle (light)
    path_light = make_semicircle_path(center_x, y_pos, radius_y, aspect_ratio, left=False)
    patch_light = PathPatch(path_light, facecolor=c_light, edgecolor='black', linewidth=0.5)
    ax_legend.add_patch(patch_light)
    
    # Text - positioned to the right of the circle
    ax_legend.text(center_x + radius_x + 0.05, y_pos, net, 
                   va='center', fontsize=11)
    
    y_pos -= spacing

save_path_png = os.path.join(output_dir, f'shen_surface_projection{output_suffix}.png')
plt.savefig(save_path_png, dpi=300, bbox_inches='tight')
print(f"Saved visualization to {save_path_png}")

# Rasterize 3D elements to reduce PDF size
for ax in axes:
    for collection in ax.collections:
        collection.set_rasterized(True)

save_path_pdf = os.path.join(output_dir, f'shen_surface_projection{output_suffix}.pdf')
plt.savefig(save_path_pdf, bbox_inches='tight', dpi=400)
print(f"Saved visualization to {save_path_pdf}")
