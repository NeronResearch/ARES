import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider
import sys

def draw_camera_frustum(ax, camera_pos, fov_deg=60, range_m=20, orientation=None):
    """
    Draw a camera frustum (field of view) in 3D space
    """
    pos = np.array(camera_pos)
    fov_rad = np.radians(fov_deg)
    
    # Default orientation is looking along positive X axis
    if orientation is None:
        forward = np.array([1, 0, 0])
        up = np.array([0, 0, 1])
        right = np.cross(forward, up)
    else:
        # If orientation data is available, use it
        forward = np.array(orientation.get('forward', [1, 0, 0]))
        up = np.array(orientation.get('up', [0, 0, 1]))
        right = np.cross(forward, up)
    
    # Calculate frustum corners
    half_height = range_m * np.tan(fov_rad / 2)
    half_width = half_height  # Assuming square FOV
    
    # Frustum corner points in camera local space
    far_center = pos + forward * range_m
    far_tl = far_center + up * half_height - right * half_width
    far_tr = far_center + up * half_height + right * half_width
    far_bl = far_center - up * half_height - right * half_width
    far_br = far_center - up * half_height + right * half_width
    
    # Draw frustum lines
    frustum_lines = [
        [pos, far_tl], [pos, far_tr], [pos, far_bl], [pos, far_br],  # From camera to corners
        [far_tl, far_tr], [far_tr, far_br], [far_br, far_bl], [far_bl, far_tl]  # Far plane rectangle
    ]
    
    for line in frustum_lines:
        ax.plot3D(*zip(*line), 'r--', alpha=0.4, linewidth=1)
    
    return far_tl, far_tr, far_bl, far_br

def parse_unified_scene_data(json_file_path):
    """
    Parse the unified scene data JSON file and extract key information
    """
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    print("=== UNIFIED SCENE DATA PARSER ===")
    
    # Extract grid information
    grid_info = data.get('grid_info', {})
    voxel_size_m = grid_info.get('voxel_size_m', 1.0)
    grid_dimensions = grid_info.get('dimensions', [0, 0, 0])
    grid_origin_m = grid_info.get('origin_m', [0.0, 0.0, 0.0])
    total_changes = grid_info.get('total_changes', 0)
    
    print(f"Grid Info:")
    print(f"  Dimensions: {grid_dimensions} voxels")
    print(f"  Voxel size: {voxel_size_m} m")
    print(f"  Origin: {grid_origin_m} m")
    print(f"  Total changes: {total_changes}")
    
    # Extract targets information
    targets = data.get('targets', [])
    print(f"\nNumber of targets: {len(targets)}")
    
    for i, target in enumerate(targets):
        name = target.get('name', f'target_{i}')
        pos = target.get('position_m', [0, 0, 0])
        frame = target.get('frame', 0)
        print(f"Target {i} ({name}): Position {pos} at frame {frame}")
    
    # Extract targets information
    targets = data.get('targets', [])
    print(f"\nNumber of targets: {len(targets)}")
    
    for i, target in enumerate(targets):
        name = target.get('name', f'target_{i}')
        pos = target.get('position_m', [0, 0, 0])
        frame = target.get('frame', 0)
        print(f"Target {i} ({name}): Position {pos} at frame {frame}")
    
    # Extract camera information
    cameras = data.get('cameras', [])
    print(f"\nNumber of cameras: {len(cameras)}")
    
    for i, camera in enumerate(cameras[:3]):  # Show first 3 cameras
        pos = camera.get('position_m', [0, 0, 0])
        fov = camera.get('fov_deg', 0)
        print(f"Camera {i}: Position {pos}, FOV: {fov}°")
    
    # Extract voxel objects (the main post-motion analysis data)
    voxel_objects = data.get('voxels', [])  # Changed from 'voxel_objects' to 'voxels'
    print(f"\nTotal voxel objects: {len(voxel_objects)}")
    
    if voxel_objects:
        # Convert to numpy arrays for analysis
        coordinates = np.array([voxel['coordinates'] for voxel in voxel_objects])
        intensities = np.array([voxel['intensity'] for voxel in voxel_objects])
        motion_types = np.array([voxel['motion_type'] for voxel in voxel_objects])
        positions_m = np.array([voxel['position_m'] for voxel in voxel_objects])
        
        # Apply voxel size scaling to coordinates to get real-world positions
        scaled_coordinates = coordinates * voxel_size_m + np.array(grid_origin_m)
        
        print(f"Grid coordinate range: X[{coordinates[:,0].min()}-{coordinates[:,0].max()}], "
              f"Y[{coordinates[:,1].min()}-{coordinates[:,1].max()}], "
              f"Z[{coordinates[:,2].min()}-{coordinates[:,2].max()}]")
        
        print(f"Real-world position range: X[{scaled_coordinates[:,0].min():.1f}-{scaled_coordinates[:,0].max():.1f}]m, "
              f"Y[{scaled_coordinates[:,1].min():.1f}-{scaled_coordinates[:,1].max():.1f}]m, "
              f"Z[{scaled_coordinates[:,2].min():.1f}-{scaled_coordinates[:,2].max():.1f}]m")
        
        print(f"Intensity range: [{intensities.min():.6f} - {intensities.max():.6f}]")
        print(f"Motion types: {np.unique(motion_types)}")
        
        # Motion analysis
        print(f"\nMotion Analysis:")
        unique_motion_types, motion_counts = np.unique(motion_types, return_counts=True)
        motion_labels = {-1: 'Unknown', 0: 'No Change', 1: 'Appeared', 2: 'Disappeared', 3: 'Changed'}
        
        for motion_type, count in zip(unique_motion_types, motion_counts):
            label = motion_labels.get(motion_type, f'Motion Type {motion_type}')
            print(f"  {label}: {count} voxels")
        
        return {
            'cameras': cameras,
            'targets': targets,
            'voxels': voxel_objects,
            'coordinates': scaled_coordinates,  # Use scaled coordinates for visualization
            'grid_coordinates': coordinates,    # Keep original grid coordinates
            'intensities': intensities,
            'motion_types': motion_types,
            'positions_m': positions_m,
            'grid_info': grid_info,
            'voxel_size_m': voxel_size_m
        }
    
    return data

def visualize_voxel_data(parsed_data):
    """
    Create an interactive 3D visualization of voxels with intensity slider
    """
    coordinates = parsed_data['coordinates']
    intensities = parsed_data['intensities']
    cameras = parsed_data['cameras']
    targets = parsed_data.get('targets', [])
    targets = parsed_data.get('targets', [])
    
    # Create figure with space for slider
    fig = plt.figure(figsize=(10, 8))
    
    # Main 3D plot
    ax = fig.add_subplot(111, projection='3d')
    plt.subplots_adjust(bottom=0.15)  # Make room for slider
    
    # Calculate intensity percentiles for filtering
    intensity_percentiles = np.percentile(intensities, np.linspace(0, 100, 101))
    
    # Initial plot - show top 50% of voxels by intensity
    initial_threshold = 50
    threshold_value = np.percentile(intensities, 100 - initial_threshold)
    mask = intensities >= threshold_value
    
    # Plot filtered voxels
    scatter = ax.scatter(coordinates[mask,0], coordinates[mask,1], coordinates[mask,2], 
                        c=intensities[mask], cmap='viridis', alpha=0.7, s=2)
    
    # Store references for updating
    current_scatter = [scatter]
    current_camera_objects = []
    current_camera_texts = []
    current_target_objects = []
    current_target_texts = []
    
    # Add cameras to the plot
    for i, camera in enumerate(cameras):
        pos = camera.get('position_m', [0, 0, 0])
        cam_scatter = ax.scatter(pos[0], pos[1], pos[2], c='red', s=150, marker='^', 
                               label='Camera' if i == 0 else "", alpha=0.9)
        # Add camera ID text
        cam_text = ax.text(pos[0], pos[1], pos[2] + 2, f'C{i+1}', fontsize=10, color='red', 
                          ha='center', weight='bold')
        current_camera_objects.append(cam_scatter)
        current_camera_texts.append(cam_text)
    
    # Add targets to the plot
    for i, target in enumerate(targets):
        pos = target.get('position_m', [0, 0, 0])
        name = target.get('name', f'target_{i}')
        frame = target.get('frame', 0)
        target_scatter = ax.scatter(pos[0], pos[1], pos[2], c='orange', s=200, marker='*', 
                                   label='Target' if i == 0 else "", alpha=1.0, edgecolor='black', linewidth=1)
        # Add target name and frame text
        target_text = ax.text(pos[0], pos[1], pos[2] + 4, f'{name}\n(f:{frame})', fontsize=9, color='orange', 
                             ha='center', weight='bold')
        current_target_objects.append(target_scatter)
        current_target_texts.append(target_text)
    
    # Set up the plot
    voxel_size = parsed_data.get('voxel_size_m', 1.0)
    title_elements = ['Cameras']
    if targets:
        title_elements.append('Targets')
    title_suffix = ' + '.join(title_elements)
    ax.set_title(f'Top {initial_threshold}% Voxels by Intensity + {title_suffix}\n(Voxel size: {voxel_size}m)', 
                fontsize=12, weight='bold')
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_zlabel('Z (m)', fontsize=12)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, label='Intensity', shrink=0.6, pad=0.1)
    
    # Add legend for cameras and targets
    if cameras or targets:
        ax.legend(loc='upper right')
    
    # Calculate and set equal aspect ratio (using all coordinates for consistent scaling)
    all_coords = coordinates
    if cameras:
        camera_positions = np.array([cam.get('position_m', [0, 0, 0]) for cam in cameras])
        all_coords = np.vstack([coordinates, camera_positions])
    if targets:
        target_positions = np.array([target.get('position_m', [0, 0, 0]) for target in targets])
        all_coords = np.vstack([all_coords, target_positions])
    if targets:
        target_positions = np.array([target.get('position_m', [0, 0, 0]) for target in targets])
        all_coords = np.vstack([all_coords, target_positions])
    
    max_range = np.array([all_coords[:,0].max()-all_coords[:,0].min(),
                         all_coords[:,1].max()-all_coords[:,1].min(),
                         all_coords[:,2].max()-all_coords[:,2].min()]).max() / 2.0
    
    mid_x = (all_coords[:,0].max()+all_coords[:,0].min()) * 0.5
    mid_y = (all_coords[:,1].max()+all_coords[:,1].min()) * 0.5
    mid_z = (all_coords[:,2].max()+all_coords[:,2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # Add slider
    ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
    slider = Slider(ax_slider, 'Top % Intensity', 1, 100, valinit=initial_threshold, 
                   valfmt='%d%%', valstep=1)
    
    # Add text showing number of voxels
    voxel_count_text = fig.text(0.5, 0.02, f'Showing {np.sum(mask):,} of {len(coordinates):,} voxels', 
                               ha='center', fontsize=10)
    
    def update_plot(val):
        """Update the plot when slider value changes"""
        percentage = int(slider.val)
        threshold_value = np.percentile(intensities, 100 - percentage)
        mask = intensities >= threshold_value
        
        # Remove old scatter plot
        if current_scatter[0]:
            current_scatter[0].remove()
        
        # Create new scatter plot with filtered data
        if np.any(mask):
            new_scatter = ax.scatter(coordinates[mask,0], coordinates[mask,1], coordinates[mask,2], 
                                   c=intensities[mask], cmap='viridis', alpha=0.7, s=2)
            current_scatter[0] = new_scatter
            
            # Update colorbar
            cbar.update_normal(new_scatter)
        else:
            current_scatter[0] = None
        
        # Update title and voxel count
        voxel_size = parsed_data.get('voxel_size_m', 1.0)
        title_elements = ['Cameras']
        if targets:
            title_elements.append('Targets')
        title_suffix = ' + '.join(title_elements)
        ax.set_title(f'Top {percentage}% Voxels by Intensity + {title_suffix}\n(Voxel size: {voxel_size}m)', 
                    fontsize=12, weight='bold')
        voxel_count_text.set_text(f'Showing {np.sum(mask):,} of {len(coordinates):,} voxels')
        
        # Redraw
        fig.canvas.draw_idle()
    
    # Connect slider to update function
    slider.on_changed(update_plot)
    
    # Add instructions
    fig.text(0.5, 0.92, 'Use the slider below to filter voxels by intensity percentile', 
             ha='center', fontsize=12, style='italic')
    
    plt.tight_layout()
    plt.show()

def analyze_motion_patterns(parsed_data):
    """
    Analyze motion patterns in the voxel data
    """
    coordinates = parsed_data['coordinates']  # These are now scaled to real-world coordinates
    motion_types = parsed_data['motion_types']
    intensities = parsed_data['intensities']
    voxel_size_m = parsed_data.get('voxel_size_m', 1.0)
    
    print("\n=== MOTION PATTERN ANALYSIS ===")
    
    # Find clusters of motion
    motion_labels = {-1: 'Unknown', 0: 'No Change', 1: 'Appeared', 2: 'Disappeared', 3: 'Changed'}
    
    for motion_type in np.unique(motion_types):
        if motion_type == 0:  # Skip "no change" voxels
            continue
            
        mask = motion_types == motion_type
        motion_coords = coordinates[mask]
        motion_intensities = intensities[mask]
        
        if len(motion_coords) > 0:
            center = np.mean(motion_coords, axis=0)
            intensity_avg = np.mean(motion_intensities)
            
            label = motion_labels.get(motion_type, f'Motion Type {motion_type}')
            print(f"{label}:")
            print(f"  Count: {len(motion_coords)} voxels")
            print(f"  Center: [{center[0]:.1f}m, {center[1]:.1f}m, {center[2]:.1f}m]")
            print(f"  Avg Intensity: {intensity_avg:.6f}")
            print(f"  Volume: {len(motion_coords) * voxel_size_m**3:.2f} m³")

# Example usage
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python VoxelViewer.py <path_to_scene_json_file>")
        print("Example: python VoxelViewer.py Scenarios/Scenario2/scenario.json")
        sys.exit(1)
    
    # Parse the unified scene data
    json_file = sys.argv[1]
    print(f"Loading scene data from: {json_file}")
    
    try:
        parsed_data = parse_unified_scene_data(json_file)
        
        # Analyze motion patterns and create visualization
        if 'coordinates' in parsed_data and len(parsed_data['coordinates']) > 0:
            analyze_motion_patterns(parsed_data)
            
            # Create 3D visualization with cameras and voxels
            print("\n=== GENERATING 3D VISUALIZATION (Voxels + Cameras) ===")
            visualize_voxel_data(parsed_data)
        else:
            print("\nNo voxel data found to visualize, but cameras are available.")
            
        # Example: Extract specific data for ML
        print("\n=== ML-READY DATA EXTRACTION ===")
        if 'coordinates' in parsed_data and len(parsed_data['coordinates']) > 0:
            # Feature matrix: [x, y, z, intensity, motion_type]
            features = np.column_stack([
                parsed_data['coordinates'],
                parsed_data['intensities'].reshape(-1, 1),
                parsed_data['motion_types'].reshape(-1, 1)
            ])
            print(f"Feature matrix shape: {features.shape}")
            print(f"Features: [x, y, z, intensity, motion_type]")
            
            # Save for ML processing
            # np.save('voxel_features.npy', features)
            # print("Saved features to 'voxel_features.npy'")
            
    except FileNotFoundError:
        print(f"Error: Could not find {json_file}")
        print("Make sure you're in the correct directory or update the file path")
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in {json_file}")