import gradio as gr
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import json
import os
from datetime import datetime
from PIL import Image, ImageDraw
import io

class SegmentCombinerApp:
    def __init__(self):
        self.h5_path = None
        self.log_path = None
        self.current_image_key = None
        self.current_data = None
        self.selected_segment = None
        self.segments_to_combine = set()
        self.h5_file = None
        self.debug = False
        self.centroid_edit_mode = False
        self.temp_centroid = None

    def load_h5_file(self, h5_path):
        """Load H5 file and return list of available image keys"""
        try:
            if self.h5_file:
                self.h5_file.close()
            self.h5_file = h5py.File(h5_path, 'r+')
            self.h5_path = h5_path
            return list(self.h5_file.keys())
        except Exception as e:
            return f"Error loading H5 file: {str(e)}"

    def set_log_path(self, log_path):
        """Set the log file path and create if doesn't exist"""
        if not log_path or log_path.strip() == "":
            self.log_path = None
            return "Logging disabled - no path provided"

        self.log_path = log_path.strip()
        try:
            log_dir = os.path.dirname(self.log_path)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)

            if not os.path.exists(self.log_path):
                with open(self.log_path, 'w') as f:
                    json.dump({"logs": []}, f)
            else:
                with open(self.log_path, 'r') as f:
                    content = f.read().strip()
                    if content:
                        json.loads(content)
                    else:
                        with open(self.log_path, 'w') as f2:
                            json.dump({"logs": []}, f2)

            return f"Log file set to: {self.log_path}"
        except Exception as e:
            self.log_path = None
            return f"Error setting log file (logging disabled): {str(e)}"

    def load_image(self, image_id):
        """Load image data from H5 file"""
        if not self.h5_file:
            return None, "Please load H5 file first"

        # Find matching key
        matching_keys = [k for k in self.h5_file.keys() if image_id in k]
        if not matching_keys:
            return None, f"No image found with ID: {image_id}"

        self.current_image_key = matching_keys[0]

        available_keys = list(self.h5_file[self.current_image_key].keys())

        self.current_data = {
            'rgb': self.h5_file[self.current_image_key]['rgb'][:],
            'segment': self.h5_file[self.current_image_key]['segment'][:],
            'filename': self.h5_file[self.current_image_key]['filename'][()].decode() if 'filename' in self.h5_file[
                self.current_image_key] else self.current_image_key
        }

        assert 'centroid' in self.h5_file[self.current_image_key]
        self.current_data['centroid'] = self.h5_file[self.current_image_key]['centroid'][:]

        # Reset selection and modes
        self.selected_segment = None
        self.segments_to_combine = set()
        self.centroid_edit_mode = False
        self.temp_centroid = None

        return self.create_left_panel_image(), f"Loaded: {self.current_image_key} (keys: {', '.join(available_keys)})"

    def create_left_panel_image(self):
        """Create the RGB image with segment overlay for left panel"""
        if self.current_data is None:
            return None

        rgb = self.current_data['rgb']
        segments = self.current_data['segment']

        overlay = rgb.copy()

        # segment overlays
        for i, seg in enumerate(segments):
            mask = seg > 0
            if np.any(mask):
                color = plt.cm.tab10(i % 10)[:3]
                # transparencies
                alpha = 0.8
                beta = 1.0 - alpha
                overlay[mask] = (
                        rgb[mask] * beta
                        + np.array(color) * 255 * alpha
                ).astype(np.uint8)

        return Image.fromarray(overlay) # PIL!

    def create_right_panel_image(self):
        """Create the selection view for right panel with centroid visualization"""
        if self.current_data is None or self.selected_segment is None:
            return None

        rgb = self.current_data['rgb']
        segments = self.current_data['segment']
        centroids = self.current_data.get('centroid', [])

        overlay = rgb.copy()

        # red overlay to all segments
        for i, seg in enumerate(segments):
            if i != self.selected_segment and i not in self.segments_to_combine:
                mask = seg > 0
                overlay[mask] = (overlay[mask] * 0.5 + np.array([255, 0, 0]) * 0.5).astype(np.uint8)

        # Apply green overlay to segments to combine (except selected)
        for seg_idx in self.segments_to_combine:
            if seg_idx != self.selected_segment:
                mask = segments[seg_idx] > 0
                overlay[mask] = (overlay[mask] * 0.2 + np.array([0, 255, 0]) * 0.8).astype(np.uint8)

        # Apply yellow overlay to selected segment (last so it's on top)
        selected_mask = segments[self.selected_segment] > 0
        overlay[selected_mask] = (overlay[selected_mask] * 0.2 + np.array([255, 255, 0]) * 0.8).astype(np.uint8)

        # PIL for drawing
        pil_image = Image.fromarray(overlay)
        draw = ImageDraw.Draw(pil_image)

        # Draw centroid for selected segment
        if self.selected_segment < len(centroids):
            # Use temporary centroid if in edit mode and it exists
            if self.centroid_edit_mode and self.temp_centroid is not None:
                cx, cy = self.temp_centroid
                draw.ellipse([cx - 8, cy - 8, cx + 8, cy + 8], fill=(255, 0, 255), outline=(255, 255, 255), width=2)
                draw.line([cx - 12, cy, cx + 12, cy], fill=(255, 255, 255), width=2)
                draw.line([cx, cy - 12, cx, cy + 12], fill=(255, 255, 255), width=2)

            # draw the current centroid
            cx, cy = centroids[self.selected_segment]
            cx, cy = int(cx), int(cy)

            # w/ crosshair
            if self.centroid_edit_mode:
                # Current centroid in cyan when in edit mode
                draw.ellipse([cx - 6, cy - 6, cx + 6, cy + 6], fill=(0, 255, 255), outline=(0, 0, 0), width=2)
            else:
                # Current centroid in blue when not in edit mode
                draw.ellipse([cx - 6, cy - 6, cx + 6, cy + 6], fill=(0, 0, 255), outline=(255, 255, 255), width=2)

            # Draw crosshair
            draw.line([cx - 10, cy, cx + 10, cy], fill=(255, 255, 255), width=1)
            draw.line([cx, cy - 10, cx, cy + 10], fill=(255, 255, 255), width=1)

        return pil_image

    def handle_left_click(self, evt: gr.SelectData):
        """Handle click on left panel to select initial segment"""
        if self.current_data is None:
            return None, "No image loaded"

        # Reset centroid edit mode
        self.centroid_edit_mode = False
        self.temp_centroid = None

        # gradio SelectData.index is [x, y]
        x_raw, y_raw = evt.index[0], evt.index[1]

        # Get actual image dimensions
        segments = self.current_data['segment']
        h, w = segments.shape[1], segments.shape[2]

        # Handle coordinate order
        if x_raw < w and y_raw < h:
            x, y = x_raw, y_raw
        elif y_raw < w and x_raw < h:
            x, y = y_raw, x_raw
        else:
            return None, f"Click out of bounds: ({x_raw}, {y_raw}) exceeds image size ({w}x{h})"

        # Find which segment was clicked
        clicked_segments = []
        for i, seg in enumerate(segments):
            if seg[y, x] > 0:
                clicked_segments.append(i)

        # If no segment found at exact pixel, check nearby pixels
        if not clicked_segments:
            search_radius = 3
            for dy in range(-search_radius, search_radius + 1):
                for dx in range(-search_radius, search_radius + 1):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w:
                        for i, seg in enumerate(segments):
                            if seg[ny, nx] > 0 and i not in clicked_segments:
                                clicked_segments.append(i)

        if clicked_segments:
            # Select the first found segment
            self.selected_segment = clicked_segments[0]
            self.segments_to_combine = {clicked_segments[0]}
            return self.create_right_panel_image(), f"Selected segment {clicked_segments[0]}"

        return None, f"No segment found near clicked location ({x}, {y})"

    def handle_right_click(self, evt: gr.SelectData):
        """Handle click on right panel to add/remove segments or set new centroid"""
        if self.current_data is None or self.selected_segment is None:
            return None, "Please select a segment from the left panel first"

        # Get click coordinates
        x_raw, y_raw = evt.index[0], evt.index[1]

        # Get actual image dimensions
        segments = self.current_data['segment']
        h, w = segments.shape[1], segments.shape[2]

        # Handle coordinate order
        if x_raw < w and y_raw < h:
            x, y = x_raw, y_raw
        elif y_raw < w and x_raw < h:
            x, y = y_raw, x_raw
        else:
            return self.create_right_panel_image(), f"Click out of bounds"

        # If in centroid edit mode, set new centroid position
        if self.centroid_edit_mode:
            self.temp_centroid = (x, y)
            return self.create_right_panel_image(), f"New centroid position: ({x}, {y}). Click 'Confirm Centroid' to save."

        # Otherwise, handle segment selection as before
        clicked_segment = None
        for i, seg in enumerate(segments):
            if seg[y, x] > 0:
                clicked_segment = i
                break

        # If no segment found at exact pixel, check nearby pixels
        if clicked_segment is None:
            search_radius = 3
            for dy in range(-search_radius, search_radius + 1):
                for dx in range(-search_radius, search_radius + 1):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w:
                        for i, seg in enumerate(segments):
                            if seg[ny, nx] > 0:
                                clicked_segment = i
                                break
                        if clicked_segment is not None:
                            break
                if clicked_segment is not None:
                    break

        if clicked_segment is not None and clicked_segment != self.selected_segment:
            if clicked_segment in self.segments_to_combine:
                self.segments_to_combine.remove(clicked_segment)
                status = f"Removed segment {clicked_segment} from combination"
            else:
                self.segments_to_combine.add(clicked_segment)
                status = f"Added segment {clicked_segment} to combination (total: {len(self.segments_to_combine)})"

            return self.create_right_panel_image(), status

        return self.create_right_panel_image(), f"No valid segment found near ({x}, {y})"

    def toggle_centroid_edit(self):
        """Toggle centroid edit mode"""
        if self.selected_segment is None:
            return None, "Please select a segment first"

        self.centroid_edit_mode = not self.centroid_edit_mode
        self.temp_centroid = None

        if self.centroid_edit_mode:
            status = "Centroid edit mode ON - click on the right panel to set new centroid position"
        else:
            status = "Centroid edit mode OFF"

        return self.create_right_panel_image(), status

    def confirm_centroid_change(self):
        """Save the new centroid position to the H5 file"""
        if not self.centroid_edit_mode or self.temp_centroid is None:
            return None, "No new centroid position set. Click on the right panel first."

        if self.selected_segment is None or self.h5_file is None:
            return None, "No segment selected"

        try:
            # Update centroid in memory
            self.current_data['centroid'][self.selected_segment] = np.array(self.temp_centroid, dtype=np.float32)

            # Save to H5 file
            if 'centroid' in self.h5_file[self.current_image_key]:
                del self.h5_file[self.current_image_key]['centroid']
            self.h5_file[self.current_image_key].create_dataset('centroid', data=self.current_data['centroid'])
            self.h5_file.flush()

            # Log the operation
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "filename": self.current_data.get('filename', self.current_image_key),
                "image_key": self.current_image_key,
                "action": "update_centroid",
                "segment": self.selected_segment,
                "old_centroid": list(self.current_data['centroid'][self.selected_segment]),
                "new_centroid": list(self.temp_centroid)
            }

            if self.log_path:
                with open(self.log_path, 'r') as f:
                    log_data = json.load(f)
                log_data['logs'].append(log_entry)
                with open(self.log_path, 'w') as f:
                    json.dump(log_data, f, indent=2)

            # Exit edit mode
            self.centroid_edit_mode = False
            self.temp_centroid = None

            status = f"Updated centroid for segment {self.selected_segment} to {list(self.temp_centroid)}"
            return self.create_right_panel_image(), status

        except Exception as e:
            import traceback
            traceback.print_exc()
            return None, f"Error updating centroid: {str(e)}"

    def segments_to_rgb(self, segments_list):
        """Convert list of segments to RGB seg_map"""
        if not segments_list:
            return np.zeros((256, 256, 3), dtype=np.uint8)

        # Create RGB segmentation map
        seg_map = np.zeros((256, 256, 3), dtype=np.uint8)

        for i, seg in enumerate(segments_list):
            mask = seg > 0
            if np.any(mask):
                # Use distinct colors for each segment
                color = plt.cm.tab10(i % 10)[:3]
                seg_map[mask] = (np.array(color) * 255).astype(np.uint8)

        return seg_map

    def delete_segment(self):
        """Delete the selected segment from the H5 file"""
        if self.selected_segment is None:
            return None, None, "No segment selected to delete"

        if self.current_data is None or self.h5_file is None:
            return None, None, "No image loaded"

        try:
            # Get current data
            segments = self.current_data['segment']

            # Check what data exists
            has_centroids = 'centroid' in self.h5_file[self.current_image_key]
            has_segment_ids = 'segment_ids' in self.h5_file[self.current_image_key]

            # Get existing data if available
            centroids = self.h5_file[self.current_image_key]['centroid'][:] if has_centroids else None
            segment_ids = self.h5_file[self.current_image_key]['segment_ids'][:] if has_segment_ids else None

            # Create lists without the deleted segment
            new_segments_list = []
            new_centroids_list = []
            new_segment_ids_list = []

            for i in range(len(segments)):
                if i != self.selected_segment:
                    new_segments_list.append(segments[i])
                    if centroids is not None and i < len(centroids):
                        new_centroids_list.append(centroids[i])
                    if segment_ids is not None and i < len(segment_ids):
                        new_segment_ids_list.append(segment_ids[i])

            # Convert lists to arrays
            if new_segments_list:
                new_segments = np.array(new_segments_list)
                new_seg_map = self.segments_to_rgb(new_segments_list)
            else:
                # If no segments left, create empty arrays
                new_segments = np.zeros((0, 256, 256), dtype=np.uint8)
                new_seg_map = np.zeros((256, 256, 3), dtype=np.uint8)

            # Delete and recreate segment dataset
            del self.h5_file[self.current_image_key]['segment']
            self.h5_file[self.current_image_key].create_dataset('segment', data=new_segments)

            # Delete and recreate seg_map
            if 'seg_map' in self.h5_file[self.current_image_key]:
                del self.h5_file[self.current_image_key]['seg_map']
            self.h5_file[self.current_image_key].create_dataset('seg_map', data=new_seg_map)

            # Handle centroids if they exist
            if has_centroids:
                del self.h5_file[self.current_image_key]['centroid']
                if new_centroids_list:
                    new_centroids = np.array(new_centroids_list)
                    self.h5_file[self.current_image_key].create_dataset('centroid', data=new_centroids)
                else:
                    self.h5_file[self.current_image_key].create_dataset('centroid',
                                                                        data=np.zeros((0, 2), dtype=np.float32))

            # Handle segment_ids if they exist
            if has_segment_ids:
                del self.h5_file[self.current_image_key]['segment_ids']
                if new_segment_ids_list:
                    new_segment_ids = np.array(new_segment_ids_list)
                    self.h5_file[self.current_image_key].create_dataset('segment_ids', data=new_segment_ids)
                else:
                    self.h5_file[self.current_image_key].create_dataset('segment_ids',
                                                                        data=np.array([], dtype=segment_ids.dtype))

            self.h5_file.flush()

            # Update current data
            self.current_data['segment'] = new_segments

            # Log the operation
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "filename": self.current_data.get('filename', self.current_image_key),
                "image_key": self.current_image_key,
                "action": "delete",
                "deleted_segment": self.selected_segment
            }

            if self.log_path:
                with open(self.log_path, 'r') as f:
                    log_data = json.load(f)
                log_data['logs'].append(log_entry)
                with open(self.log_path, 'w') as f:
                    json.dump(log_data, f, indent=2)

            # Reset selection
            deleted_segment = self.selected_segment
            self.selected_segment = None
            self.segments_to_combine = set()
            self.centroid_edit_mode = False
            self.temp_centroid = None

            status = f"Deleted segment {deleted_segment}"
            return self.create_left_panel_image(), None, status

        except Exception as e:
            import traceback
            traceback.print_exc()
            return None, None, f"Error deleting segment: {str(e)}"

    def reset_selection(self):
        """Reset additional segment selections but keep the base segment"""
        if self.selected_segment is not None:
            # Keep only the base segment in the combination set
            self.segments_to_combine = {self.selected_segment}
            # Also reset centroid edit mode
            self.centroid_edit_mode = False
            self.temp_centroid = None
            return self.create_right_panel_image(), "Reset to base segment only"
        return None, "No base segment selected"

    def combine_segments(self):
        """Combine selected segments and save to H5 file"""
        if not self.segments_to_combine or len(self.segments_to_combine) < 2:
            return None, None, "Please select at least 2 segments to combine"

        if self.current_data is None or self.h5_file is None or self.selected_segment is None:
            return None, None, "No image or segment selected"

        try:
            segments = self.current_data['segment']
            combined_mask = np.zeros_like(segments[0], dtype=bool)

            # Combine all selected segments (from the RIGHT panel)
            for seg_idx in self.segments_to_combine:
                combined_mask |= segments[seg_idx] > 0

            # Update ONLY the selected segment (from LEFT panel) with combined mask
            new_segments = segments.copy()
            new_segments[self.selected_segment] = combined_mask.astype(np.uint8) * 255

            # Update centroid to be the centroid of the combined mask
            if 'centroid' in self.current_data:
                y_coords, x_coords = np.where(combined_mask)
                if len(x_coords) > 0:
                    new_centroid = [np.mean(x_coords), np.mean(y_coords)]
                    self.current_data['centroid'][self.selected_segment] = np.array(new_centroid, dtype=np.float32)

                    # Save updated centroids
                    if 'centroid' in self.h5_file[self.current_image_key]:
                        del self.h5_file[self.current_image_key]['centroid']
                    self.h5_file[self.current_image_key].create_dataset('centroid', data=self.current_data['centroid'])

            # Save to H5 file
            del self.h5_file[self.current_image_key]['segment']
            self.h5_file[self.current_image_key].create_dataset('segment', data=new_segments)
            self.h5_file.flush()

            # Update current data
            self.current_data['segment'] = new_segments

            # Log the operation
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "filename": self.current_data['filename'],
                "image_key": self.current_image_key,
                "segments_combined": sorted(list(self.segments_to_combine)),
                "combined_into": self.selected_segment
            }

            if self.log_path:
                with open(self.log_path, 'r') as f:
                    log_data = json.load(f)
                log_data['logs'].append(log_entry)
                with open(self.log_path, 'w') as f:
                    json.dump(log_data, f, indent=2)

            # Reset selection
            self.selected_segment = None
            self.segments_to_combine = set()
            self.centroid_edit_mode = False
            self.temp_centroid = None

            status = f"Combined segments {sorted(list(log_entry['segments_combined']))} into segment {log_entry['combined_into']}"
            return self.create_left_panel_image(), None, status

        except Exception as e:
            return None, None, f"Error combining segments: {str(e)}"

    def __del__(self):
        if self.h5_file:
            self.h5_file.close()


app = SegmentCombinerApp()

# GRADIO INTERFACE
with gr.Blocks(title="Segment Combiner") as demo:
    gr.Markdown("# Segment Combination Tool")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Configuration")

            # H5 and Log file inputs side by side
            with gr.Row():
                with gr.Column():
                    h5_path_input = gr.Textbox(label="H5 File Path", placeholder="/path/to/consolidated.h5")
                    load_h5_btn = gr.Button("Load H5 File")
                with gr.Column():
                    log_path_input = gr.Textbox(label="Log File Path (Optional)", placeholder="/path/to/log.json")
                    set_log_btn = gr.Button("Set Log File")

            available_images = gr.Textbox(label="Available Images", interactive=False, lines=3)

            image_id_input = gr.Textbox(label="Image ID/Filename", placeholder="image1007")
            load_image_btn = gr.Button("Load Image")

            status_text = gr.Textbox(label="Status", interactive=False, lines=2)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Original Image with Segments")
            gr.Markdown("*Click on a segment to select it*")
            left_image = gr.Image(label="RGB + All Segments", interactive=True)
            with gr.Row():
                clear_selection_btn = gr.Button("Clear Selection", variant="secondary", scale=1)
                delete_segment_btn = gr.Button("🗑️ Delete Segment", variant="stop", scale=1)

        with gr.Column(scale=1):
            gr.Markdown("### Segment Selection")
            gr.Markdown("*Yellow: Selected | Green: To Combine | Red: Available*")
            gr.Markdown("*Blue dot: Current centroid | Magenta dot: New centroid (edit mode)*")
            gr.Markdown("*Click red segments to add to combination*")
            right_image = gr.Image(label="Selection View", interactive=True)

            # Buttons under the right panel
            with gr.Row():
                reset_btn = gr.Button("Reset Selection", variant="secondary", scale=1)
                combine_btn = gr.Button("Combine Segments", variant="primary", scale=1)

            # Centroid editing buttons
            with gr.Row():
                change_centroid_btn = gr.Button("Change Centroid", variant="secondary", scale=1)
                confirm_centroid_btn = gr.Button("Confirm Centroid", variant="primary", scale=1)

    # Event handlers
    load_h5_btn.click(
        fn=lambda path: (app.load_h5_file(path),
                         str(app.load_h5_file(path))),
        inputs=[h5_path_input],
        outputs=[status_text, available_images]
    )

    set_log_btn.click(
        fn=app.set_log_path,
        inputs=[log_path_input],
        outputs=[status_text]
    )

    load_image_btn.click(
        fn=lambda img_id: (*app.load_image(img_id), None),
        inputs=[image_id_input],
        outputs=[left_image, status_text, right_image]
    )

    left_image.select(
        fn=app.handle_left_click,
        inputs=[],
        outputs=[right_image, status_text]
    )

    right_image.select(
        fn=app.handle_right_click,
        inputs=[],
        outputs=[right_image, status_text]
    )

    clear_selection_btn.click(
        fn=app.reset_selection,
        inputs=[],
        outputs=[right_image, status_text]
    )

    reset_btn.click(
        fn=app.reset_selection,
        inputs=[],
        outputs=[right_image, status_text]
    )

    delete_segment_btn.click(
        fn=app.delete_segment,
        inputs=[],
        outputs=[left_image, right_image, status_text]
    )

    combine_btn.click(
        fn=app.combine_segments,
        inputs=[],
        outputs=[left_image, right_image, status_text]
    )

    change_centroid_btn.click(
        fn=app.toggle_centroid_edit,
        inputs=[],
        outputs=[right_image, status_text]
    )

    confirm_centroid_btn.click(
        fn=app.confirm_centroid_change,
        inputs=[],
        outputs=[right_image, status_text]
    )

if __name__ == "__main__":
    demo.launch(share=True)