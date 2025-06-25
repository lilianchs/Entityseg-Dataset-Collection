import gradio as gr
import json
import os
import h5py
import numpy as np
import cv2
from pycocotools import mask as mask_utils
import random
from pathlib import Path
import colorsys
from PIL import Image

class EntitySegAnnotator:
    """
    Annotation tool for EntitySeg dataset.
    """
    def __init__(self):
        self.data = None
        self.image_dir = None
        self.save_dir = None
        self.discard_dir = None
        self.current_batch = []
        self.current_idx = 0
        self.current_image_data = None
        self.current_segments = []
        self.deleted_segments = set()
        self.deletion_history = []  # Track order of deletions for undo
        self.jumped_to_specific = False  # Track if we jumped to a specific image
        self.segment_map = None

    def get_filename_without_extension(self, filename):
        return os.path.splitext(filename)[0]

    def get_file_extension(self, filename):
        return os.path.splitext(filename)[1]

    def load_json_data(self, json_path, image_dir, save_dir, discard_dir):
        if not os.path.exists(json_path):
            return "Error: JSON file does not exist", None, None
        if not os.path.exists(image_dir):
            return "Error: Image directory does not exist", None, None

        # Create directories if they don't exist
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(discard_dir, exist_ok=True)

        # Load data
        with open(json_path, 'r') as f:
            self.data = json.load(f)

        self.image_dir = image_dir
        self.save_dir = save_dir
        self.discard_dir = discard_dir

        # Get already processed files from both save and discard directories
        self.processed = set()

        for h5_file in Path(save_dir).glob("*.h5"):
            self.processed.add(h5_file.stem)
        for h5_file in Path(discard_dir).glob("*.h5"):
            self.processed.add(h5_file.stem)

        # initial batch
        batch_info = self.load_batch()
        if batch_info.startswith("Error"):
            return batch_info, None, None

        # first image
        if self.current_batch:
            self.current_image_data = self.current_batch[0]

        return batch_info, *self.display_current_image()

    def is_supported_image_format(self, filename):
        """Check if the file has a supported image extension"""
        ext = self.get_file_extension(filename).lower()
        return ext in ['.jpg', '.jpeg', '.png']

    def load_batch(self):
        """Load 20 random unprocessed images"""
        # Filter available images
        available_images = []
        for img in self.data['images']:
            filename = img['file_name']
            # Check if it's a supported format
            if not self.is_supported_image_format(filename):
                print(f"Skipping unsupported format: {filename}")
                continue
            filename_no_ext = self.get_filename_without_extension(filename)
            if filename_no_ext not in self.processed:
                # Also check if the actual image file exists
                img_path = os.path.join(self.image_dir, filename)
                if os.path.exists(img_path):
                    available_images.append(img)

        if not available_images:
            return "Error: No more images to process"

        # Sample up to 20 images
        sample_size = min(20, len(available_images))
        self.current_batch = random.sample(available_images, sample_size)
        self.current_idx = 0
        self.deleted_segments = set()
        self.deletion_history = []  # Reset deletion history
        self.jumped_to_specific = False  # Reset jump flag

        # Set first image from batch
        if self.current_batch:
            self.current_image_data = self.current_batch[0]

        return f"Loaded {sample_size} images. Showing image 1/{sample_size}"

    def get_image_annotations(self, image_id):
        """Get all annotations for a specific image"""
        return [ann for ann in self.data['annotations'] if ann['image_id'] == image_id]

    def decode_rle_mask(self, segmentation, height, width):
        """Decode RLE mask from COCO format"""
        if isinstance(segmentation, dict):
            rle = segmentation
        else:
            # If it's polygon format, convert to RLE
            rle = mask_utils.frPyObjects(segmentation, height, width)
            if isinstance(rle, list):
                rle = mask_utils.merge(rle)

        mask = mask_utils.decode(rle)
        return mask

    def generate_colors(self, n):
        """Generate n distinct colors"""
        colors = []
        for i in range(n):
            hue = i / n
            rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
            colors.append(tuple(int(x * 255) for x in rgb))
        return colors

    def display_current_image(self):
        """Display current image and its segments"""
        # Check if we have an image to display
        if not self.current_image_data:
            # Try to get from batch if available
            if self.current_batch and 0 <= self.current_idx < len(self.current_batch):
                self.current_image_data = self.current_batch[self.current_idx]
            else:
                return None, None

        # Ensure the image format is supported
        if not self.is_supported_image_format(self.current_image_data['file_name']):
            return None, None

        image_id = self.current_image_data['id']
        filename = self.current_image_data['file_name']
        height = self.current_image_data['height']
        width = self.current_image_data['width']

        # Load actual image
        img_path = os.path.join(self.image_dir, filename)
        original_img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if original_img is None:
            # Fallback to placeholder if image can't be loaded
            print(f"Warning: Could not load image {img_path}")
            original_img = np.ones((height, width, 3), dtype=np.uint8) * 200
        else:
            original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

        # Get annotations
        self.current_segments = self.get_image_annotations(image_id)

        # Create overlay image
        overlay_img = original_img.copy()
        colors = self.generate_colors(len(self.current_segments))

        # Create a mask to track which segment each pixel belongs to
        segment_map = np.zeros((height, width), dtype=np.int32) - 1

        for idx, (ann, color) in enumerate(zip(self.current_segments, colors)):
            if ann['id'] not in self.deleted_segments:
                mask = self.decode_rle_mask(ann['segmentation'], height, width)
                # Apply color overlay
                overlay_img[mask > 0] = (
                        overlay_img[mask > 0] * 0.1 + np.array(color) * 0.9
                ).astype(np.uint8)
                # Update segment map
                segment_map[mask > 0] = idx

        # Store segment map for click handling
        self.segment_map = segment_map

        # Convert to PIL for Gradio
        original_pil = Image.fromarray(original_img)
        overlay_pil = Image.fromarray(overlay_img)

        return original_pil, overlay_pil

    def handle_click(self, evt: gr.SelectData):
        """Handle click on overlay image to delete segment"""
        if self.segment_map is None or not hasattr(self, 'segment_map'):
            _, overlay = self.display_current_image()
            return overlay

        x, y = evt.index
        # Get segment at clicked position
        segment_idx = self.segment_map[y, x]

        if segment_idx >= 0:
            # Add segment to deleted set
            segment_id = self.current_segments[segment_idx]['id']
            if segment_id not in self.deleted_segments:
                self.deleted_segments.add(segment_id)
                self.deletion_history.append(segment_id)  # Track deletion order

            # Redraw overlay
            _, overlay = self.display_current_image()
            return overlay

        _, overlay = self.display_current_image()
        return overlay

    def undo_last_deletion(self):
        """Undo the last segment deletion"""
        if not self.deletion_history:
            _, overlay = self.display_current_image()
            return "No deletions to undo", overlay

        # Get the last deleted segment ID
        last_deleted_id = self.deletion_history.pop()

        # Remove from deleted set
        self.deleted_segments.discard(last_deleted_id)

        # Redraw overlay
        _, overlay = self.display_current_image()
        return f"Restored segment {last_deleted_id}", overlay

    def jump_to_image(self, identifier):
        """Jump to a specific image by ID or filename"""
        if not identifier or not self.data:
            return "Please enter an image ID or filename", None, None

        identifier = identifier.strip()
        target_image = None

        # Check if it's a filename ending with .h5
        if identifier.endswith('.h5'):
            # Get the base filename without .h5
            base_filename = identifier.replace('.h5', '')
            # Find image by matching the base filename (without extension)
            for img in self.data['images']:
                if self.get_filename_without_extension(img['file_name']) == base_filename:
                    target_image = img
                    break
        else:
            # Try to parse as image ID
            try:
                image_id = int(identifier)
                # Find image by ID
                for img in self.data['images']:
                    if img['id'] == image_id:
                        target_image = img
                        break
            except ValueError:
                return "Invalid input: Enter either an image ID (number) or filename.h5", None, None

        if not target_image:
            return f"Image not found: {identifier}", None, None

        # Check if it's a supported format
        if not self.is_supported_image_format(target_image['file_name']):
            return f"Unsupported image format: {target_image['file_name']}", None, None

        # Check if image file exists
        img_path = os.path.join(self.image_dir, target_image['file_name'])
        if not os.path.exists(img_path):
            return f"Image file not found: {target_image['file_name']}", None, None

        # Set current image data
        self.current_image_data = target_image
        self.deleted_segments = set()
        self.deletion_history = []
        self.jumped_to_specific = True

        # Load the segments for this image
        self.current_segments = self.get_image_annotations(target_image['id'])

        # If this image was previously processed, we can still load it for editing
        status = f"Loaded image: {target_image['file_name']} (ID: {target_image['id']})"
        filename_no_ext = self.get_filename_without_extension(target_image['file_name'])
        if filename_no_ext in self.processed:
            status += " - Previously processed (will overwrite if saved)"

        # Display the image
        orig, overlay = self.display_current_image()
        return status, orig, overlay

    def save_current(self):
        """Save current image and remaining segments to H5"""
        if not self.current_image_data:
            return "No image to save", None, None

        # Get valid segment IDs
        valid_segment_ids = [
            seg['id'] for seg in self.current_segments
            if seg['id'] not in self.deleted_segments
        ]

        # Create H5 filename
        filename = self.current_image_data['file_name']
        filename_no_ext = self.get_filename_without_extension(filename)
        h5_filename = filename_no_ext + '.h5'
        h5_path = os.path.join(self.save_dir, h5_filename)

        # Save to H5
        with h5py.File(h5_path, 'w') as f:
            f.create_dataset('image_id', data=self.current_image_data['id'])
            # Handle empty list case for H5
            if valid_segment_ids:
                f.create_dataset('valid_segment_ids', data=valid_segment_ids)
            else:
                # Create empty dataset with proper dtype
                f.create_dataset('valid_segment_ids', data=np.array([], dtype=np.int64))

        # Mark as processed (store without extension)
        self.processed.add(filename_no_ext)
        print(f"Discarded {h5_filename}")

        # Clear current state</        self.deleted_segments = set()
        self.deletion_history = []

        # If we jumped to a specific image
        if self.jumped_to_specific:
            self.jumped_to_specific = False
            self.current_image_data = None
            # If we have a batch in progress, continue from where we left off
            if self.current_batch and self.current_idx < len(self.current_batch):
                self.current_image_data = self.current_batch[self.current_idx]
                status = f"Image saved ({len(valid_segment_ids)} segments kept). Returning to batch - showing image {self.current_idx + 1}/{len(self.current_batch)}"
                orig, overlay = self.display_current_image()
                return status, orig, overlay
            else:
                return f"Image saved ({len(valid_segment_ids)} segments kept). Load a new batch or jump to another image.", None, None

        # Normal batch mode - move to next image
        return self.next_image()

    def discard_current(self):
        """Discard current image by creating empty H5 in discard folder"""
        if not self.current_image_data:
            return "No image to discard", None, None

        # Create empty H5 file in discard folder
        filename = self.current_image_data['file_name']
        filename_no_ext = self.get_filename_without_extension(filename)
        h5_filename = filename_no_ext + '.h5'
        h5_path = os.path.join(self.discard_dir, h5_filename)

        # Create empty H5 file
        try:
            with h5py.File(h5_path, 'w') as f:
                # Just create an empty file as a marker
                pass
            print(f"Successfully discarded {h5_filename}")
        except Exception as e:
            print(f"Error discarding {h5_filename}: {str(e)}")
            return f"Error discarding file: {str(e)}", None, None

        # Mark as processed
        self.processed.add(filename)

        # Clear current state
        self.deleted_segments = set()
        self.deletion_history = []

        # If we jumped to a specific image
        if self.jumped_to_specific:
            self.jumped_to_specific = False
            self.current_image_data = None
            # If we have a batch in progress, continue from where we left off
            if self.current_batch and self.current_idx < len(self.current_batch):
                self.current_image_data = self.current_batch[self.current_idx]
                status = f"Image discarded. Returning to batch - showing image {self.current_idx + 1}/{len(self.current_batch)}"
                orig, overlay = self.display_current_image()
                return status, orig, overlay
            else:
                return "Image discarded. Load a new batch or jump to another image.", None, None

        # Normal batch mode - move to next image
        return self.next_image()

    def next_image(self):
        """Move to next image in batch"""
        # Clear current state
        self.deleted_segments = set()
        self.deletion_history = []
        self.jumped_to_specific = False

        # Increment index
        self.current_idx += 1

        if self.current_idx >= len(self.current_batch):
            # Clear current image data when batch is complete
            self.current_image_data = None
            return "Batch complete. Click 'Refresh' to load more images.", None, None

        # Set current image from batch
        self.current_image_data = self.current_batch[self.current_idx]

        status = f"Showing image {self.current_idx + 1}/{len(self.current_batch)}"
        orig, overlay = self.display_current_image()
        return status, orig, overlay

    def refresh_batch(self):
        """Load a new batch of images"""
        # Reset states
        self.jumped_to_specific = False
        self.current_image_data = None

        batch_info = self.load_batch()
        if batch_info.startswith("Error"):
            return batch_info, None, None

        # Set the first image from the new batch
        if self.current_batch:
            self.current_image_data = self.current_batch[0]

        # Display first image of new batch
        return batch_info, *self.display_current_image()


# Create annotator instance
annotator = EntitySegAnnotator()

# Create Gradio interface
with gr.Blocks(title="EntitySeg Annotation Tool") as app:
    gr.Markdown("# EntitySeg Dataset Annotation Tool")
    gr.Markdown("Click on segments in the right panel to delete them. Save only the segments you want to keep.")
    gr.Markdown("**Supported image formats:** .jpg, .JPEG, .png")

    with gr.Row():
        json_path = gr.Textbox(label="Path to JSON annotations file", placeholder="/path/to/annotations.json")
        image_dir = gr.Textbox(label="Path to Image Directory", placeholder="/path/to/entityseg/images")

    with gr.Row():
        save_dir = gr.Textbox(label="Path to H5 save directory", placeholder="/path/to/save/directory")
        discard_dir = gr.Textbox(label="Path to discard directory", placeholder="/path/to/discard/directory")

    load_btn = gr.Button("Load Data", variant="primary")

    with gr.Row():
        jump_input = gr.Textbox(
            label="Jump to specific image",
            placeholder="Enter image ID (e.g., 42) or filename.h5 (e.g., coco_000000091500.h5)"
        )
        jump_btn = gr.Button("Jump")

    status_text = gr.Textbox(label="Status", interactive=False)

    with gr.Row():
        with gr.Column():
            original_image = gr.Image(label="Original Image", interactive=False)
        with gr.Column():
            overlay_image = gr.Image(label="Segmented Image (Click to delete segments)", interactive=False)

    with gr.Row():
        save_btn = gr.Button("Save Image and Segments", variant="primary")
        discard_btn = gr.Button("Discard Entire Image", variant="stop")
        undo_btn = gr.Button("Undo Last Deletion")
        refresh_btn = gr.Button("Refresh (Load New Batch)")

    # Bottom panel showing batch preview (simplified version)
    batch_preview = gr.Gallery(label="Current Batch Preview", columns=10, height=100)

    # Event handlers
    load_btn.click(
        fn=lambda j, i, s, d: annotator.load_json_data(j, i, s, d),
        inputs=[json_path, image_dir, save_dir, discard_dir],
        outputs=[status_text, original_image, overlay_image]
    )

    jump_btn.click(
        fn=lambda j: annotator.jump_to_image(j),
        inputs=[jump_input],
        outputs=[status_text, original_image, overlay_image]
    ).then(
        fn=lambda: "",
        inputs=[],
        outputs=[jump_input]
    )

    jump_input.submit(
        fn=lambda j: annotator.jump_to_image(j),
        inputs=[jump_input],
        outputs=[status_text, original_image, overlay_image]
    ).then(
        fn=lambda: "",
        inputs=[],
        outputs=[jump_input]
    )

    overlay_image.select(
        fn=annotator.handle_click,
        inputs=[],
        outputs=[overlay_image]
    )

    save_btn.click(
        fn=lambda: annotator.save_current(),
        inputs=[],
        outputs=[status_text, original_image, overlay_image]
    )

    discard_btn.click(
        fn=annotator.discard_current,
        inputs=[],
        outputs=[status_text, original_image, overlay_image]
    )

    undo_btn.click(
        fn=annotator.undo_last_deletion,
        inputs=[],
        outputs=[status_text, overlay_image]
    )

    refresh_btn.click(
        fn=annotator.refresh_batch,
        inputs=[],
        outputs=[status_text, original_image, overlay_image]
    )

if __name__ == "__main__":
    app.launch()