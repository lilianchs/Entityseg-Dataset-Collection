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
from collection.visualize_entityseg import segments_to_rgb, crop
import cv2

def get_seg_map(img_path, annotations, size=256):
    image_og = cv2.imread(img_path)[:, :, ::-1]  # BGR to RGB
    h, w, _ = image_og.shape
    all_segs = []

    image, cropped_segs, annotations = crop(image_og, annotations, size=size)

    for id, seg in cropped_segs.items():
        if (seg.sum() / (size * size)) * 100 <= 0.2:
            continue
        all_segs.append(seg)

    seg_map = segments_to_rgb(all_segs)

    return image, seg_map


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

    return 1


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
    ct_vis = 0 # counter
    for h5_file in h5_files:
        if ct_vis % 25 == 0:
            print(f'Visualized {ct_vis}/{len(h5_files)}')
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

        # valid_annotations = [ann for ann in valid_segment_ids if ann['id'] in valid_segment_ids]
        valid_annotations = [annotations[id] for id in valid_segment_ids]
        assert len(valid_annotations) == len(valid_segment_ids), \
            f"Mismatch in valid annotations for image {img_key}"

        rgb_img, seg_map = get_seg_map(img_path, valid_annotations)

        save_path = os.path.join(args.save_dir, f"{img_key}.png")
        ct_vis += visualize_and_save(rgb_img, seg_map, save_path, image_id)

    print(f'Visualized {ct_vis} Examples for {len(h5_files)} H5 Files')

if __name__ == "__main__":
    main()
