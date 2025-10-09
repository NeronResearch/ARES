bl_info = {
    "name": "ARES Scenario Generator",
    "author": "Connor",
    "version": (1, 0),
    "blender": (4, 0, 0),
    "location": "View3D > Sidebar > ARES Scenario Generator",
    "description": "Procedurally generate cameras and targets with random placement, motion paths, and animation",
    "category": "Object",
}

import bpy
import bmesh
import random
import math
from mathutils import Vector, Euler

# -------------------------------------------------------------------
# UI Panel for Scenario Generation
# -------------------------------------------------------------------
class ARES_PT_GeneratorPanel(bpy.types.Panel):
    bl_label = "Scenario Generator"
    bl_idname = "ARES_PT_generator_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "ARES Scenario Generator"

    def draw(self, context):
        layout = self.layout
        scene = context.scene

        # Generate Target Nodes section (moved to top)
        layout.separator()
        box = layout.box()
        
        # Collapsible header for Generate Target Nodes
        row = box.row()
        row.prop(scene, "ares_expand_target_nodes", 
                icon="TRIA_DOWN" if scene.ares_expand_target_nodes else "TRIA_RIGHT", 
                text="Generate Target Nodes", emboss=False, icon_only=False)
        
        if scene.ares_expand_target_nodes:
            # Camera visibility distance setting
            box.prop(scene, "ares_camera_visibility_distance")
            
            # Generate Box button
            box.operator("ares.generate_camera_nodes_box", icon='MESH_CUBE')
            
            # Bounding box selector
            box.prop(scene, "ares_target_bounding_box")
            
            # Node spacing setting
            box.prop(scene, "ares_target_node_spacing")
            
            # Generate Target Nodes button
            box.operator("ares.generate_target_nodes", icon='PLUS')

        # Generator Parameters section (moved below)
        layout.separator()
        box = layout.box()
        
        # Collapsible header for Generator Parameters
        row = box.row()
        row.prop(scene, "ares_expand_generator_params", 
                icon="TRIA_DOWN" if scene.ares_expand_generator_params else "TRIA_RIGHT", 
                text="Generator Parameters", emboss=False, icon_only=False)
        
        if scene.ares_expand_generator_params:
            # Generation toggles
            row = box.row()
            row.prop(scene, "ares_gen_enable_cameras")
            row.prop(scene, "ares_gen_enable_targets")
            
            # Camera settings
            if scene.ares_gen_enable_cameras:
                box.separator()
                box.label(text="Camera Settings:", icon='CAMERA_DATA')
                box.prop(scene, "ares_gen_camera_count")
                box.prop(scene, "ares_gen_camera_fov")
            
            # Target settings
            if scene.ares_gen_enable_targets:
                box.separator()
                box.label(text="Target Settings:", icon='EMPTY_DATA')
                box.prop(scene, "ares_gen_target_count")
                box.prop(scene, "ares_gen_target_min_speed")
                box.prop(scene, "ares_gen_target_max_speed")
                box.prop(scene, "ares_gen_target_size")
                box.prop(scene, "ares_gen_target_type")

            layout.separator()
            layout.operator("ares.generate_scenario", icon='OUTLINER_OB_EMPTY')


# -------------------------------------------------------------------
# Scenario Generator Operator
# -------------------------------------------------------------------
class ARES_OT_GenerateScenario(bpy.types.Operator):
    bl_idname = "ares.generate_scenario"
    bl_label = "Generate Scenario"
    bl_description = "Generate random cameras and targets with animation between nodes"

    def execute(self, context):
        scene = context.scene

        # Check what's enabled and find required nodes
        camera_nodes = []
        target_starting_nodes = []
        target_nodes = []
        
        if scene.ares_gen_enable_cameras:
            camera_nodes = [obj for obj in scene.objects if obj.name.startswith("camera_node")]
            if not camera_nodes:
                self.report({'ERROR'}, "Camera generation enabled but no camera_node empties found in the scene.")
                return {'CANCELLED'}
        
        if scene.ares_gen_enable_targets:
            target_nodes = [obj for obj in scene.objects if obj.name.startswith("target_node")]
            
            if not target_nodes:
                self.report({'ERROR'}, "Target generation enabled but no target_node empties found in the scene.")
                return {'CANCELLED'}
        
        # Check if at least one generation type is enabled
        if not scene.ares_gen_enable_cameras and not scene.ares_gen_enable_targets:
            self.report({'ERROR'}, "Both camera and target generation are disabled. Enable at least one to generate a scenario.")
            return {'CANCELLED'}

        # ------------------------------------------------------------
        # SETUP COLLECTIONS
        # ------------------------------------------------------------
        camera_collection = None
        target_collection = None
        
        # Create or get camera collection if cameras are enabled
        if scene.ares_gen_enable_cameras:
            if "cameras" in bpy.data.collections:
                camera_collection = bpy.data.collections["cameras"]
                # Remove existing objects from collection
                for obj in camera_collection.objects:
                    bpy.data.objects.remove(obj, do_unlink=True)
            else:
                camera_collection = bpy.data.collections.new("cameras")
                scene.collection.children.link(camera_collection)

        # Create or get target collection if targets are enabled
        if scene.ares_gen_enable_targets:
            if "targets" in bpy.data.collections:
                target_collection = bpy.data.collections["targets"]
                # Remove existing objects from collection
                for obj in target_collection.objects:
                    bpy.data.objects.remove(obj, do_unlink=True)
            else:
                target_collection = bpy.data.collections.new("targets")
                scene.collection.children.link(target_collection)

        # ------------------------------------------------------------
        # CAMERA PLACEMENT
        # ------------------------------------------------------------
        if scene.ares_gen_enable_cameras:
            # Ensure we don't try to place more cameras than available nodes
            max_cameras = min(scene.ares_gen_camera_count, len(camera_nodes))
            if max_cameras < scene.ares_gen_camera_count:
                self.report({'WARNING'}, f"Only {len(camera_nodes)} camera nodes available. Generating {max_cameras} cameras instead of {scene.ares_gen_camera_count}.")
            
            # Get unique camera nodes for placement
            available_camera_nodes = camera_nodes.copy()
            
            for i in range(max_cameras):
                base_node = random.choice(available_camera_nodes)
                available_camera_nodes.remove(base_node)  # Remove to prevent reuse
                
                cam_data = bpy.data.cameras.new(f"Camera_{i}")
                cam_obj = bpy.data.objects.new(f"Camera_{i}", cam_data)
                camera_collection.objects.link(cam_obj)

                # Place at node
                cam_obj.location = base_node.location.copy()

                # Random rotation with specified constraints
                x_rotation = random.uniform(math.radians(125), math.radians(180))  # X between 125° and 180°
                y_rotation = random.uniform(math.radians(-10), math.radians(10))   # Y between -10° and 10°
                z_rotation = random.uniform(0, math.tau)                          # Z any random rotation (0° to 360°)
                cam_obj.rotation_euler = Euler((x_rotation, y_rotation, z_rotation), 'XYZ')

                # Set FOV
                cam_data.angle = math.radians(scene.ares_gen_camera_fov)

        # ------------------------------------------------------------
        # TARGET PLACEMENT & ANIMATION
        # ------------------------------------------------------------
        if scene.ares_gen_enable_targets:
            # Check for drone object if drone type is selected
            drone_object = None
            if scene.ares_gen_target_type == "DRONE":
                drone_object = scene.objects.get("Drone")
                if not drone_object:
                    self.report({'ERROR'}, "Drone target type selected but 'Drone' object not found in scene.")
                    return {'CANCELLED'}
            
            for t in range(scene.ares_gen_target_count):
                start_node = random.choice(target_nodes)
                target = bpy.data.objects.new(f"target_{t}", None)
                target.empty_display_size = scene.ares_gen_target_size
                target.empty_display_type = 'SPHERE'
                target.location = start_node.location.copy()
                target_collection.objects.link(target)
                
                # Handle drone type
                if scene.ares_gen_target_type == "DRONE" and drone_object:
                    # Create a duplicate of the drone for this target
                    drone_copy = drone_object.copy()
                    if drone_object.data:
                        drone_copy.data = drone_object.data.copy()
                    drone_copy.name = f"Drone_{t}"
                    target_collection.objects.link(drone_copy)
                    
                    # Parent the drone to the target sphere first
                    drone_copy.parent = target
                    drone_copy.parent_type = 'OBJECT'
                    
                    # Clear the drone's local transform to snap it to the parent's location
                    drone_copy.location = (0, 0, 0)
                    drone_copy.rotation_euler = (0, 0, 0)
                    drone_copy.scale = (1, 1, 1)

                # Pick 10 random flight nodes for the path
                waypoints = random.sample(target_nodes, min(10, len(target_nodes)))

                # Set keyframe at starting position
                target.keyframe_insert(data_path="location", frame=scene.frame_start)

                # Animate linear motion between waypoints
                current_frame = scene.frame_start
                current_location = target.location.copy()
                
                for wp in waypoints:
                    speed = random.uniform(scene.ares_gen_target_min_speed, scene.ares_gen_target_max_speed)
                    dist = (wp.location - current_location).length
                    
                    if dist > 0:  # Only animate if there's distance to cover
                        frames_to_next = max(1, int(dist / speed * scene.render.fps))
                        current_frame += frames_to_next
                        
                        target.location = wp.location.copy()
                        target.keyframe_insert(data_path="location", frame=current_frame)
                        current_location = wp.location.copy()

        # Hide camera_nodes_box if it exists
        camera_nodes_box = bpy.data.objects.get("camera_nodes_box")
        if camera_nodes_box:
            camera_nodes_box.hide_viewport = True
            camera_nodes_box.hide_render = True

        self.report({'INFO'}, "Scenario generated successfully.")
        return {'FINISHED'}


# -------------------------------------------------------------------
# Generate Camera Nodes Box Operator
# -------------------------------------------------------------------
class ARES_OT_GenerateCameraNodesBox(bpy.types.Operator):
    bl_idname = "ares.generate_camera_nodes_box"
    bl_label = "Generate Box"
    bl_description = "Generate a bounding box containing all points cameras can see within specified distance"

    def execute(self, context):
        scene = context.scene
        
        # Remove existing camera_nodes_box if it exists
        existing_box = bpy.data.objects.get("camera_nodes_box")
        if existing_box:
            bpy.data.objects.remove(existing_box, do_unlink=True)
        
        # Find all camera objects
        cameras = [obj for obj in scene.objects if obj.type == 'CAMERA']
        
        if not cameras:
            self.report({'ERROR'}, "No camera objects found in the scene.")
            return {'CANCELLED'}
        
        # Calculate bounding box based on camera positions and visibility distance
        visibility_distance = scene.ares_camera_visibility_distance
        
        # Get all camera positions
        positions = [camera.location for camera in cameras]
        
        # Find the min and max coordinates
        min_x = min(pos.x for pos in positions) - visibility_distance
        max_x = max(pos.x for pos in positions) + visibility_distance
        min_y = min(pos.y for pos in positions) - visibility_distance
        max_y = max(pos.y for pos in positions) + visibility_distance
        min_z = min(pos.z for pos in positions) - visibility_distance
        max_z = max(pos.z for pos in positions) + visibility_distance
        
        # Create the wireframe bounding box mesh
        mesh = bpy.data.meshes.new("camera_nodes_box_mesh")
        box_obj = bpy.data.objects.new("camera_nodes_box", mesh)
        
        # Calculate box dimensions
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        center_z = (min_z + max_z) / 2
        
        size_x = max_x - min_x
        size_y = max_y - min_y
        size_z = max_z - min_z
        
        # Create wireframe box vertices (8 corners)
        verts = [
            (min_x, min_y, min_z),  # 0: bottom-back-left
            (max_x, min_y, min_z),  # 1: bottom-back-right
            (max_x, max_y, min_z),  # 2: bottom-front-right
            (min_x, max_y, min_z),  # 3: bottom-front-left
            (min_x, min_y, max_z),  # 4: top-back-left
            (max_x, min_y, max_z),  # 5: top-back-right
            (max_x, max_y, max_z),  # 6: top-front-right
            (min_x, max_y, max_z),  # 7: top-front-left
        ]
        
        # Create wireframe edges (12 edges of a cube)
        edges = [
            # Bottom face edges
            (0, 1), (1, 2), (2, 3), (3, 0),
            # Top face edges
            (4, 5), (5, 6), (6, 7), (7, 4),
            # Vertical edges
            (0, 4), (1, 5), (2, 6), (3, 7)
        ]
        
        # Create the mesh with vertices and edges only (no faces for wireframe)
        mesh.from_pydata(verts, edges, [])
        mesh.update()
        
        # Ensure the box stays in the main scene collection
        scene.collection.objects.link(box_obj)
        
        # Make it the active object
        context.view_layer.objects.active = box_obj
        box_obj.select_set(True)
        
        # Set display mode to wireframe to make it clearly visible as wireframe
        box_obj.display_type = 'WIRE'
        
        # Auto-populate the bounding box selector
        scene.ares_target_bounding_box = box_obj
        
        self.report({'INFO'}, f"Camera nodes box generated with visibility distance {visibility_distance}m")
        return {'FINISHED'}


# -------------------------------------------------------------------
# Generate Target Nodes Operator
# -------------------------------------------------------------------
class ARES_OT_GenerateTargetNodes(bpy.types.Operator):
    bl_idname = "ares.generate_target_nodes"
    bl_label = "Generate Target Nodes"
    bl_description = "Generate randomly placed target nodes within the selected bounding box"

    def execute(self, context):
        scene = context.scene
        
        # Check if bounding box is selected
        if not scene.ares_target_bounding_box:
            self.report({'ERROR'}, "No bounding box selected. Please select a bounding box object.")
            return {'CANCELLED'}
        
        bounding_box = scene.ares_target_bounding_box
        
        # Check if the bounding box object still exists in the scene
        if bounding_box.name not in bpy.data.objects:
            self.report({'ERROR'}, "Selected bounding box object no longer exists. Please select a valid bounding box.")
            return {'CANCELLED'}
        
        node_spacing = scene.ares_target_node_spacing
        
        # Remove existing target_nodes collection if it exists
        existing_collection = bpy.data.collections.get("target_nodes")
        if existing_collection:
            # Remove all objects in the collection
            for obj in existing_collection.objects:
                bpy.data.objects.remove(obj, do_unlink=True)
            # Remove the collection itself
            bpy.data.collections.remove(existing_collection)
        
        # Create new target_nodes collection
        target_nodes_collection = bpy.data.collections.new("target_nodes")
        scene.collection.children.link(target_nodes_collection)
        
        # Calculate bounding box dimensions and position using actual mesh bounds
        # This ensures we use the current geometry, not just the object transform
        mesh = bounding_box.data
        
        # Get the bounding box from the mesh vertices in world space
        matrix_world = bounding_box.matrix_world
        bbox_corners = [matrix_world @ Vector(corner) for corner in bounding_box.bound_box]
        
        # Calculate actual bounds from the mesh
        min_x = min(corner.x for corner in bbox_corners)
        max_x = max(corner.x for corner in bbox_corners)
        min_y = min(corner.y for corner in bbox_corners)
        max_y = max(corner.y for corner in bbox_corners)
        min_z = min(corner.z for corner in bbox_corners)
        max_z = max(corner.z for corner in bbox_corners)
        
        # Calculate how many nodes we can fit in each dimension
        box_size_x = max_x - min_x
        box_size_y = max_y - min_y
        box_size_z = max_z - min_z
        
        # Use a more reasonable approach for node count calculation
        # Calculate nodes per dimension based on spacing
        nodes_x = max(1, int(box_size_x / node_spacing))
        nodes_y = max(1, int(box_size_y / node_spacing))
        nodes_z = max(1, int(box_size_z / node_spacing))
        target_node_count = min(100, nodes_x * nodes_y * nodes_z // 4)  # Cap at 100 nodes and use 1/4 density
        
        # Generate random positions within the bounding box
        node_count = 0
        max_attempts = target_node_count * 50  # More reasonable attempt limit
        attempts = 0
        consecutive_failures = 0
        
        while node_count < target_node_count and attempts < max_attempts and consecutive_failures < 1000:
            # Generate random position within bounds
            x = random.uniform(min_x, max_x)
            y = random.uniform(min_y, max_y)
            z = random.uniform(min_z, max_z)
            
            # Check if this position is far enough from existing nodes
            position_valid = True
            for obj in target_nodes_collection.objects:
                if obj.name.startswith("target_node"):
                    distance = (Vector((x, y, z)) - obj.location).length
                    if distance < node_spacing:
                        position_valid = False
                        break
            
            if position_valid:
                # Create target node
                bpy.ops.object.empty_add(type='SPHERE', location=(x, y, z))
                node_obj = context.active_object
                node_obj.name = f"target_node_{node_count:03d}"
                node_obj.empty_display_size = 0.5
                
                # Move to target_nodes collection safely
                # Remove from all collections first
                for collection in node_obj.users_collection:
                    collection.objects.unlink(node_obj)
                
                # Add to target_nodes collection
                target_nodes_collection.objects.link(node_obj)
                
                node_count += 1
                consecutive_failures = 0  # Reset failure counter
            else:
                consecutive_failures += 1
            
            attempts += 1
            
            # Progress update for user feedback (every 100 attempts)
            if attempts % 100 == 0:
                bpy.context.window_manager.progress_update(attempts / max_attempts * 100)
        
        # Hide camera_nodes_box if it exists
        camera_nodes_box = bpy.data.objects.get("camera_nodes_box")
        if camera_nodes_box:
            camera_nodes_box.hide_viewport = True
            camera_nodes_box.hide_render = True

        self.report({'INFO'}, f"Generated {node_count} target nodes with {node_spacing}m spacing")
        return {'FINISHED'}


# -------------------------------------------------------------------
# Registration
# -------------------------------------------------------------------
classes = [
    ARES_PT_GeneratorPanel,
    ARES_OT_GenerateScenario,
    ARES_OT_GenerateCameraNodesBox,
    ARES_OT_GenerateTargetNodes,
]

def register():
    for cls in classes:
        bpy.utils.register_class(cls)

    bpy.types.Scene.ares_gen_enable_cameras = bpy.props.BoolProperty(
        name="Generate Cameras", default=True, description="Enable camera generation")
    bpy.types.Scene.ares_gen_enable_targets = bpy.props.BoolProperty(
        name="Generate Targets", default=True, description="Enable target generation")
    bpy.types.Scene.ares_gen_camera_count = bpy.props.IntProperty(
        name="Camera Count", default=3, min=1, max=50)
    bpy.types.Scene.ares_gen_camera_fov = bpy.props.FloatProperty(
        name="Camera FOV (deg)", default=60.0, min=1.0, max=179.0)
    bpy.types.Scene.ares_gen_target_count = bpy.props.IntProperty(
        name="Target Count", default=1, min=1, max=10)
    bpy.types.Scene.ares_gen_target_min_speed = bpy.props.FloatProperty(
        name="Min Speed (m/s)", default=1.0, min=0.1)
    bpy.types.Scene.ares_gen_target_max_speed = bpy.props.FloatProperty(
        name="Max Speed (m/s)", default=5.0, min=0.1)
    bpy.types.Scene.ares_gen_target_size = bpy.props.FloatProperty(
        name="Target Size (m)", default=1.0, min=0.01)
    bpy.types.Scene.ares_gen_target_type = bpy.props.EnumProperty(
        name="Target Type",
        items=[
            ("SPHERE", "Sphere", "Spherical target"),
            ("DRONE", "Drone", "Drone target with Quadcopter Drone model")
        ],
        default="SPHERE"
    )
    
    # Generate Target Nodes properties
    bpy.types.Scene.ares_camera_visibility_distance = bpy.props.FloatProperty(
        name="Camera Visibility Distance (m)", 
        default=50.0, 
        min=1.0, 
        description="Distance within which cameras can see targets"
    )
    bpy.types.Scene.ares_target_bounding_box = bpy.props.PointerProperty(
        name="Bounding Box",
        type=bpy.types.Object,
        description="Select the bounding box object for target node generation"
    )
    bpy.types.Scene.ares_target_node_spacing = bpy.props.FloatProperty(
        name="Node Spacing (m)",
        default=5.0,
        min=0.1,
        description="Average distance between target nodes"
    )
    
    # UI expand properties for collapsible sections
    bpy.types.Scene.ares_expand_target_nodes = bpy.props.BoolProperty(
        name="Expand Target Nodes",
        default=True,
        description="Show/hide Generate Target Nodes section"
    )
    bpy.types.Scene.ares_expand_generator_params = bpy.props.BoolProperty(
        name="Expand Generator Parameters", 
        default=False,
        description="Show/hide Generator Parameters section"
    )

def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)

    del bpy.types.Scene.ares_gen_enable_cameras
    del bpy.types.Scene.ares_gen_enable_targets
    del bpy.types.Scene.ares_gen_camera_count
    del bpy.types.Scene.ares_gen_camera_fov
    del bpy.types.Scene.ares_gen_target_count
    del bpy.types.Scene.ares_gen_target_min_speed
    del bpy.types.Scene.ares_gen_target_max_speed
    del bpy.types.Scene.ares_gen_target_size
    del bpy.types.Scene.ares_gen_target_type
    del bpy.types.Scene.ares_camera_visibility_distance
    del bpy.types.Scene.ares_target_bounding_box
    del bpy.types.Scene.ares_target_node_spacing
    del bpy.types.Scene.ares_expand_target_nodes
    del bpy.types.Scene.ares_expand_generator_params

if __name__ == "__main__":
    register()
