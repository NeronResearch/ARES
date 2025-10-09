bl_info = {
    "name": "ARES Scenario Renderer",
    "author": "Connor",
    "version": (5, 0),
    "blender": (4, 0, 0),
    "location": "View3D > Sidebar > ARES Scenario Renderer",
    "description": "Render camera test scenarios with metadata export, multi-target support, and full scenario tracking",
    "category": "Render",
}

import bpy
import os
import json
import uuid
import datetime
import math
from mathutils import Matrix, Vector

# -------------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------------
def is_target_visible_from_camera(target_obj, camera_obj, max_distance):
    """Check if target is visible from a camera within max distance and not occluded"""
    # Get target and camera positions
    target_pos = target_obj.matrix_world.translation
    camera_pos = camera_obj.matrix_world.translation
    
    # Check distance
    distance = (target_pos - camera_pos).length
    if distance > max_distance:
        return False
    
    # Use raycast to check for occlusion
    scene = bpy.context.scene
    depsgraph = bpy.context.evaluated_depsgraph_get()
    
    # Cast ray from camera to target
    direction = (target_pos - camera_pos).normalized()
    
    # Raycast
    result, location, normal, index, hit_object, matrix = scene.ray_cast(
        depsgraph, camera_pos, direction, distance=distance
    )
    
    # If ray hits the target object or no hit, target is visible
    if not result or hit_object == target_obj:
        return True
    
    return False

def calculate_target_speed(target_obj, frame, fps):
    """Calculate target speed between current and previous frame"""
    if frame <= 1:
        return 0.0
    
    scene = bpy.context.scene
    
    # Get current position
    scene.frame_set(frame)
    current_pos = target_obj.matrix_world.translation.copy()
    
    # Get previous position
    scene.frame_set(frame - 1)
    prev_pos = target_obj.matrix_world.translation.copy()
    
    # Calculate distance and speed
    distance = (current_pos - prev_pos).length
    time_delta = 1.0 / fps
    speed = distance / time_delta
    
    return speed

# -------------------------------------------------------------------
# Target Property Group
# -------------------------------------------------------------------
class ARESTarget(bpy.types.PropertyGroup):
    name: bpy.props.StringProperty(name="Target Name")
    object: bpy.props.PointerProperty(name="Target Object", type=bpy.types.Object)

# -------------------------------------------------------------------
# UI Panel
# -------------------------------------------------------------------
class ARES_PT_MainPanel(bpy.types.Panel):
    bl_label = "ARES Scenario Renderer"
    bl_idname = "ARES_PT_main_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "ARES Scenario Renderer"

    def draw(self, context):
        scene = context.scene
        layout = self.layout

        box = layout.box()
        box.label(text="Scenario Management", icon='SCENE_DATA')
        box.prop(scene, "ares_scenario_name", text="Scenario Name")
        box.prop(scene, "ares_scenario_dir", text="Scenario Directory")

        row = box.row()
        row.operator("ares.clear_scenario", icon='TRASH')

        box.separator()
        box.label(text="Scene Time (Start)", icon='TIME')
        col = box.column(align=True)
        col.prop(scene, "ares_year")
        col.prop(scene, "ares_month")
        col.prop(scene, "ares_day")
        col.prop(scene, "ares_hour")
        col.prop(scene, "ares_minute")
        col.prop(scene, "ares_second")

        box.separator()
        box.label(text="Scene Settings", icon='RENDER_ANIMATION')
        box.prop(scene, "ares_origin_lat")
        box.prop(scene, "ares_origin_lon")
        box.prop(scene, "ares_origin_alt")
        box.prop(scene, "ares_random_noise", text="Random Noise Per Frame")

        box.separator()
        box.label(text="Target Detection", icon='VIEWZOOM')
        box.prop(scene, "ares_target_max_distance", text="Max Detection Distance (m)")
        box.prop(scene, "ares_target_min_speed", text="Min Detection Speed (m/s)")

        box.separator()
        box.label(text="Framerate", icon='RENDER_RESULT')
        box.prop(scene.render, "fps", text="Framerate")

        box.separator()
        box.label(text="Targets", icon='EMPTY_ARROWS')
        row = box.row()
        row.operator("ares.add_target", icon='ADD')
        row.operator("ares.remove_target", icon='REMOVE')

        for tgt in scene.ares_targets:
            tgt_box = box.box()
            tgt_box.prop(tgt, "name", text="Target Name")
            tgt_box.prop(tgt, "object")

        # Check if rendering is in progress
        total_progress = getattr(scene, "ares_total_progress", 0)
        is_rendering = total_progress > 0
        
        if is_rendering:
            progress_box = layout.box()
            progress_box.label(text="Rendering in Progress", icon='RENDER_ANIMATION')
            progress = getattr(scene, "ares_rendering_progress", 0)
            current_camera = getattr(scene, "ares_current_camera", "")
            
            progress_box.label(text=f"Camera: {current_camera}")
            
            # Prevent division by zero
            if total_progress > 0:
                percentage = (progress / total_progress * 100)
                progress_box.label(text=f"Progress: {progress}/{total_progress} ({percentage:.1f}%)")
            else:
                progress_box.label(text=f"Progress: {progress}/0 (0.0%)")
            
            # Show cancel button during rendering
            progress_box.operator("ares.cancel_render", icon='CANCEL', text="Cancel Rendering")
        else:
            # Render controls
            row = layout.row(align=True)
            row.operator("ares.render_all_cameras", icon='RENDER_ANIMATION')
            row.prop(scene, "ares_overwrite_frames", text="Overwrite")


# -------------------------------------------------------------------
# Clear Scenario
# -------------------------------------------------------------------
class ARES_OT_ClearScenario(bpy.types.Operator):
    bl_idname = "ares.clear_scenario"
    bl_label = "Clear Scenario"

    def execute(self, context):
        scene = context.scene
        scene.ares_scenario_name = ""
        scene.ares_scenario_dir = ""
        scene.ares_targets.clear()
        self.report({'INFO'}, "Scenario cleared.")
        return {'FINISHED'}


# -------------------------------------------------------------------
# Cancel Render Operator
# -------------------------------------------------------------------
class ARES_OT_CancelRender(bpy.types.Operator):
    bl_idname = "ares.cancel_render"
    bl_label = "Cancel Rendering"
    bl_description = "Cancel the current rendering operation"

    def execute(self, context):
        # Stop any running modal operators
        try:
            # Try to cancel any active render operations
            bpy.ops.render.render('INVOKE_DEFAULT')
        except:
            pass
        
        # Clear progress tracking immediately
        context.scene.ares_rendering_progress = 0
        context.scene.ares_total_progress = 0 
        context.scene.ares_current_camera = ""
        
        # Force UI redraw to show the render button again
        for area in context.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()
        
        self.report({'INFO'}, "Rendering cancelled")
        return {'FINISHED'}


# -------------------------------------------------------------------
# Target Management
# -------------------------------------------------------------------
class ARES_OT_AddTarget(bpy.types.Operator):
    bl_idname = "ares.add_target"
    bl_label = "Add Target"

    def execute(self, context):
        context.scene.ares_targets.add()
        return {'FINISHED'}

class ARES_OT_RemoveTarget(bpy.types.Operator):
    bl_idname = "ares.remove_target"
    bl_label = "Remove Target"

    def execute(self, context):
        targets = context.scene.ares_targets
        if targets:
            targets.remove(len(targets) - 1)
        return {'FINISHED'}


# -------------------------------------------------------------------
# Render & Export Operator
# -------------------------------------------------------------------
class ARES_OT_RenderAll(bpy.types.Operator):
    bl_idname = "ares.render_all_cameras"
    bl_label = "Render All Cameras"
    bl_description = "Render all cameras and export scenario data"

    _timer = None
    _cameras = []
    _current_camera_idx = 0
    _current_frame = 0
    _frame_start = 0
    _frame_end = 0
    _fps = 0
    _scenario_dir = ""
    _scenario_uuid = ""
    _scenario_version = 0
    _scene_unix_start = 0
    _original_camera = None
    _cameras_info = []
    _targets_info = []
    _is_rendering = False
    _total_frames = 0
    _rendered_frames = 0

    def modal(self, context, event):
        if event.type == 'TIMER':
            if not self._is_rendering:
                # Continue with the next render step
                return self._process_next_step(context)
        
        return {'PASS_THROUGH'}

    def _process_next_step(self, context):
        scene = context.scene
        
        # Check if we've finished all cameras
        if self._current_camera_idx >= len(self._cameras):
            return self._finish_rendering(context)
        
        # Get current camera
        current_camera = self._cameras[self._current_camera_idx]
        
        # Check if we've finished all frames for this camera
        if self._current_frame > self._frame_end:
            # Move to next camera
            self._current_camera_idx += 1
            self._current_frame = self._frame_start
            
            # Update progress
            progress = (self._current_camera_idx / len(self._cameras)) * 100
            self.report({'INFO'}, f"Camera {self._current_camera_idx}/{len(self._cameras)} completed ({progress:.1f}%)")
            
            # Update UI progress
            context.scene.ares_current_camera = self._cameras[self._current_camera_idx-1].name if self._current_camera_idx > 0 else ""
            
            return {'RUNNING_MODAL'}
        
        # Render current frame
        self._render_frame(context, current_camera)
        
        # Move to next frame
        self._current_frame += 1
        
        return {'RUNNING_MODAL'}

    def _render_frame(self, context, camera_obj):
        scene = context.scene
        
        # Set frame and camera
        scene.frame_set(self._current_frame)
        scene.camera = camera_obj
        
        # Create camera directory if needed
        cam_name = camera_obj.name
        cam_dir = os.path.join(self._scenario_dir, cam_name)
        os.makedirs(cam_dir, exist_ok=True)
        
        # Check if frame already exists and overwrite is disabled
        frame_path = os.path.join(cam_dir, f"{self._current_frame:04d}.png")
        meta_path = os.path.join(cam_dir, f"{self._current_frame:04d}.json")
        
        # Skip if both files exist and overwrite is disabled
        if not scene.ares_overwrite_frames and os.path.exists(frame_path) and os.path.exists(meta_path):
            # Still need to add camera info if this is the first frame
            if self._current_frame == self._frame_start:
                self._cameras_info.append({
                    "name": cam_name,
                    "frames_dir": cam_dir
                })
            
            # Update progress tracking for skipped frame
            self._rendered_frames += 1
            context.scene.ares_rendering_progress = self._rendered_frames
            context.scene.ares_current_camera = cam_name
            return
        
        # Render frame
        scene.render.filepath = frame_path
        
        # Use viewport render for speed and responsiveness
        bpy.ops.render.opengl(write_still=True)
        
        # Generate metadata
        loc = camera_obj.matrix_world.translation
        rot_mat = camera_obj.matrix_world.to_3x3()
        sensor_size = camera_obj.data.sensor_width
        fov_rad = camera_obj.data.angle
        fov_deg = math.degrees(fov_rad)

        proj_matrix = camera_obj.calc_matrix_camera(
            depsgraph=bpy.context.evaluated_depsgraph_get(),
            x=scene.render.resolution_x,
            y=scene.render.resolution_y,
            scale_x=1,
            scale_y=1
        )

        frame_meta = {
            "frame": self._current_frame,
            "timestamp_unix": self._scene_unix_start + int((self._current_frame - self._frame_start) / self._fps),
            "position_m": [loc.x, loc.y, loc.z],
            "rotation_matrix": [list(r) for r in rot_mat],
            "sensor_size_mm": sensor_size,
            "fov_deg": fov_deg,
            "projection_matrix": [list(row) for row in proj_matrix]
        }

        meta_path = os.path.join(cam_dir, f"{self._current_frame:04d}.json")
        with open(meta_path, 'w') as f:
            json.dump(frame_meta, f, indent=4)
        
        # Add camera info if this is the first frame
        if self._current_frame == self._frame_start:
            self._cameras_info.append({
                "name": cam_name,
                "frames_dir": cam_dir
            })
        
        # Update progress tracking
        self._rendered_frames += 1
        context.scene.ares_rendering_progress = self._rendered_frames
        context.scene.ares_current_camera = cam_name

    def _finish_rendering(self, context):
        scene = context.scene
        
        # Restore original camera
        if self._original_camera:
            scene.camera = self._original_camera
        
        # Export target metadata
        self._export_targets(context)
        
        # Save scenario.json
        self._save_scenario_json(context)
        
        # Cleanup
        wm = context.window_manager
        wm.event_timer_remove(self._timer)
        
        # Clear progress tracking
        context.scene.ares_rendering_progress = 0
        context.scene.ares_total_progress = 0
        context.scene.ares_current_camera = ""
        
        self.report({'INFO'}, f"Scenario rendering completed! Exported to {self._scenario_dir}")
        return {'FINISHED'}

    def _export_targets(self, context):
        scene = context.scene
        
        # Get detection parameters
        max_distance = scene.ares_target_max_distance
        min_speed = scene.ares_target_min_speed
        
        for tgt in scene.ares_targets:
            tgt_obj = tgt.object
            if not tgt_obj:
                continue

            tgt_data = []

            for frame in range(self._frame_start, self._frame_end + 1):
                scene.frame_set(frame)
                loc = tgt_obj.matrix_world.translation
                
                # Calculate target speed
                speed = calculate_target_speed(tgt_obj, frame, self._fps)
                
                # Check visibility from cameras
                visible_cameras = []
                for camera in self._cameras:
                    if is_target_visible_from_camera(tgt_obj, camera, max_distance):
                        visible_cameras.append(camera.name)
                
                # Determine if target is detected based on criteria:
                # - Visible by at least 2 cameras
                # - Moving faster than min speed
                # - Within max distance (already checked in visibility function)
                is_detected = len(visible_cameras) >= 2 and speed >= min_speed

                tgt_data.append({
                    "frame": frame,
                    "timestamp_unix": self._scene_unix_start + int((frame - self._frame_start) / self._fps),
                    "position_m": [loc.x, loc.y, loc.z],
                    "speed_mps": speed,
                    "visible_cameras": visible_cameras,
                    "is_detected": is_detected,
                    "detection_criteria": {
                        "min_cameras_required": 2,
                        "cameras_seeing_target": len(visible_cameras),
                        "min_speed_required_mps": min_speed,
                        "current_speed_mps": speed,
                        "max_detection_distance_m": max_distance
                    }
                })

            # Get target dimensions
            dimensions = tgt_obj.dimensions
            
            # Calculate target length (scale as magnitude of dimensions)
            target_length = math.sqrt(dimensions.x**2 + dimensions.y**2 + dimensions.z**2)

            target_metadata = {
                "target_id": tgt.name,
                "size_xyz_m": [dimensions.x, dimensions.y, dimensions.z],
                "scale_length_m": target_length,
                "detection_settings": {
                    "max_detection_distance_m": max_distance,
                    "min_detection_speed_mps": min_speed,
                    "min_cameras_required": 2
                },
                "frames": tgt_data
            }

            tgt_path = os.path.join(self._scenario_dir, f"target_{tgt.name}.json")
            with open(tgt_path, 'w') as f:
                json.dump(target_metadata, f, indent=4)

            self._targets_info.append({
                "name": tgt.name,
                "file": tgt_path
            })

    def _save_scenario_json(self, context):
        scenario_meta = {
            "scenario_uuid": self._scenario_uuid,
            "scenario_version": self._scenario_version,
            "build_timestamp_unix": int(datetime.datetime.now().timestamp()),
            "scenario_name": context.scene.ares_scenario_name,
            "frame_rate": self._fps,
            "total_frames": self._frame_end - self._frame_start + 1,
            "origin_lla": [context.scene.ares_origin_lat, context.scene.ares_origin_lon, context.scene.ares_origin_alt],
            "scenario_dir": self._scenario_dir,
            "cameras": self._cameras_info,
            "targets": self._targets_info
        }

        scenario_path = os.path.join(self._scenario_dir, "scenario.json")
        with open(scenario_path, 'w') as f:
            json.dump(scenario_meta, f, indent=4)

    def execute(self, context):
        scene = context.scene
        
        # Initialize rendering state
        self._scenario_dir = bpy.path.abspath(scene.ares_scenario_dir)
        if not self._scenario_dir:
            self.report({'ERROR'}, "Please specify a scenario directory")
            return {'CANCELLED'}
            
        os.makedirs(self._scenario_dir, exist_ok=True)

        # Generate scenario metadata
        self._scenario_uuid = str(uuid.uuid4())
        self._scenario_version = getattr(scene, "ares_scenario_version", 1)
        setattr(scene, "ares_scenario_version", self._scenario_version + 1)

        # Convert GUI time to Unix
        scene_start = datetime.datetime(
            scene.ares_year, scene.ares_month, scene.ares_day,
            scene.ares_hour, scene.ares_minute, scene.ares_second
        )
        self._scene_unix_start = int(scene_start.timestamp())

        # Initialize rendering parameters
        self._frame_start = scene.frame_start
        self._frame_end = scene.frame_end
        self._fps = scene.render.fps
        self._current_frame = self._frame_start
        
        # Collect all cameras
        self._cameras = [obj for obj in scene.objects if obj.type == 'CAMERA']
        if not self._cameras:
            self.report({'ERROR'}, "No cameras found in the scene")
            return {'CANCELLED'}
        
        self._current_camera_idx = 0
        self._original_camera = scene.camera
        self._cameras_info = []
        self._targets_info = []
        self._is_rendering = False
        
        # Initialize progress tracking
        self._total_frames = len(self._cameras) * (self._frame_end - self._frame_start + 1)
        self._rendered_frames = 0
        
        # Set UI progress tracking
        scene.ares_rendering_progress = 0
        scene.ares_total_progress = self._total_frames
        scene.ares_current_camera = ""
        
        # Start modal operation
        wm = context.window_manager
        self._timer = wm.event_timer_add(0.01, window=context.window)
        wm.modal_handler_add(self)
        
        self.report({'INFO'}, f"Starting render of {len(self._cameras)} cameras, {self._total_frames} total frames...")
        
        return {'RUNNING_MODAL'}

    def cancel(self, context):
        # Cleanup on cancellation
        wm = context.window_manager
        if self._timer:
            wm.event_timer_remove(self._timer)
        
        # Clear progress tracking
        context.scene.ares_rendering_progress = 0
        context.scene.ares_total_progress = 0
        context.scene.ares_current_camera = ""
        
        # Restore original camera
        if self._original_camera:
            context.scene.camera = self._original_camera


# -------------------------------------------------------------------
# Registration
# -------------------------------------------------------------------
classes = [
    ARESTarget,
    ARES_PT_MainPanel,
    ARES_OT_ClearScenario,
    ARES_OT_CancelRender,
    ARES_OT_AddTarget,
    ARES_OT_RemoveTarget,
    ARES_OT_RenderAll
]

def register():
    for cls in classes:
        bpy.utils.register_class(cls)

    bpy.types.Scene.ares_scenario_name = bpy.props.StringProperty(name="Scenario Name", default="")
    bpy.types.Scene.ares_scenario_dir = bpy.props.StringProperty(name="Scenario Directory", subtype='DIR_PATH')
    bpy.types.Scene.ares_scenario_version = bpy.props.IntProperty(default=1)

    bpy.types.Scene.ares_year = bpy.props.IntProperty(name="Year", default=2025)
    bpy.types.Scene.ares_month = bpy.props.IntProperty(name="Month", default=1, min=1, max=12)
    bpy.types.Scene.ares_day = bpy.props.IntProperty(name="Day", default=1, min=1, max=31)
    bpy.types.Scene.ares_hour = bpy.props.IntProperty(name="Hour", default=0, min=0, max=23)
    bpy.types.Scene.ares_minute = bpy.props.IntProperty(name="Minute", default=0, min=0, max=59)
    bpy.types.Scene.ares_second = bpy.props.IntProperty(name="Second", default=0, min=0, max=59)

    bpy.types.Scene.ares_origin_lat = bpy.props.FloatProperty(name="Origin Latitude", default=0.0)
    bpy.types.Scene.ares_origin_lon = bpy.props.FloatProperty(name="Origin Longitude", default=0.0)
    bpy.types.Scene.ares_origin_alt = bpy.props.FloatProperty(name="Origin Altitude (m)", default=0.0)

    bpy.types.Scene.ares_random_noise = bpy.props.BoolProperty(name="Random Noise Per Frame", default=False)
    
    # Target detection parameters
    bpy.types.Scene.ares_target_max_distance = bpy.props.FloatProperty(
        name="Target Max Detection Distance", 
        description="Maximum distance in meters for target detection",
        default=200.0, 
        min=1.0, 
        max=10000.0
    )
    bpy.types.Scene.ares_target_min_speed = bpy.props.FloatProperty(
        name="Target Min Detection Speed", 
        description="Minimum speed in m/s for target detection",
        default=3.0, 
        min=0.0, 
        max=1000.0
    )

    bpy.types.Scene.ares_targets = bpy.props.CollectionProperty(type=ARESTarget)
    
    # Render settings
    bpy.types.Scene.ares_overwrite_frames = bpy.props.BoolProperty(
        name="Overwrite Frames",
        description="When enabled, re-render all frames. When disabled, skip existing frames",
        default=True
    )
    
    # Progress tracking properties
    bpy.types.Scene.ares_rendering_progress = bpy.props.IntProperty(default=0)
    bpy.types.Scene.ares_total_progress = bpy.props.IntProperty(default=0)
    bpy.types.Scene.ares_current_camera = bpy.props.StringProperty(default="")

def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)

    del bpy.types.Scene.ares_scenario_name
    del bpy.types.Scene.ares_scenario_dir
    del bpy.types.Scene.ares_scenario_version
    del bpy.types.Scene.ares_year
    del bpy.types.Scene.ares_month
    del bpy.types.Scene.ares_day
    del bpy.types.Scene.ares_hour
    del bpy.types.Scene.ares_minute
    del bpy.types.Scene.ares_second
    del bpy.types.Scene.ares_origin_lat
    del bpy.types.Scene.ares_origin_lon
    del bpy.types.Scene.ares_origin_alt

    del bpy.types.Scene.ares_random_noise
    del bpy.types.Scene.ares_target_max_distance
    del bpy.types.Scene.ares_target_min_speed
    del bpy.types.Scene.ares_targets
    del bpy.types.Scene.ares_overwrite_frames
    del bpy.types.Scene.ares_rendering_progress
    del bpy.types.Scene.ares_total_progress
    del bpy.types.Scene.ares_current_camera


if __name__ == "__main__":
    register()
