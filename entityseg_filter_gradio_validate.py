import h5py
import glob
import os
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils
import argparse

def overlay_segments_on_image(img_np, annotations):
    overlaid = img_np.copy()
    for ann in annotations:
        seg = ann['segmentation']
        mask = mask_utils.decode(seg)
        if mask.ndim == 3:
            mask = np.any(mask, axis=2)
        color = np.random.randint(0, 256, size=3, dtype=np.uint8)
        overlaid[mask == 1] = color
    return overlaid


def visualize_and_save(img_np, overlaid_img, save_path, image_id):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(img_np)
    axes[0].set_title(f"Image (ID: {image_id})")
    axes[0].axis("off")

    axes[1].imshow(overlaid_img)
    axes[1].set_title("Image + Colored Segments")
    axes[1].axis("off")

    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Visualize and overlay entity segments on RGB images.")
    parser.add_argument('--h5_dir', required=True, help="Directory containing .h5 files")
    parser.add_argument('--save_dir', required=True, help="Directory to save visualizations")
    parser.add_argument('--ann_path', required=True, help="Path to the annotation JSON file")
    parser.add_argument('--img_dir', required=True, help="Directory containing original RGB images")
    args = parser.parse_args()

    # Validations
    assert os.path.exists(args.ann_path), f"Annotation JSON file not found: {args.ann_path}"
    assert os.path.exists(args.img_dir), f"Image directory not found: {args.img_dir}"
    assert os.path.exists(args.h5_dir), f"H5 directory not found: {args.h5_dir}"
    os.makedirs(args.save_dir, exist_ok=True)

    h5_files = glob.glob(os.path.join(args.h5_dir, '*.h5'))
    assert h5_files, f"No .h5 files found in directory: {args.h5_dir}"

    entity_seg = json.load(open(args.ann_path))
    annotations = entity_seg['annotations']

    valid_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']

    for h5_file in h5_files:
        with h5py.File(h5_file, 'r') as data:
            image_id = data['image_id'][()]
            valid_segment_ids = data['valid_segment_ids'][()]

        img_key = os.path.splitext(os.path.basename(h5_file))[0]

        # Find the matching image with any of the valid extensions
        for ext in valid_extensions:
            img_path = os.path.join(args.img_dir, f'{img_key}{ext}')
            if os.path.exists(img_path):
                break

        assert os.path.exists(img_path), f"Image file not found: {img_path}"

        valid_annotations = [ann for ann in annotations if ann['id'] in valid_segment_ids]
        assert len(valid_annotations) == len(valid_segment_ids), \
            f"Mismatch in valid annotations for image {img_key}"

        img = Image.open(img_path).convert("RGB")
        img_np = np.array(img)

        overlaid_img = overlay_segments_on_image(img_np, valid_annotations)
        save_path = os.path.join(args.save_dir, f"{img_key}.png")
        visualize_and_save(img_np, overlaid_img, save_path, image_id)

if __name__ == "__main__":
    main()
