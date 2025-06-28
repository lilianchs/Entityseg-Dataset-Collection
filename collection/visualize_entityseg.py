import os
import json
import numpy as np
import math
import matplotlib.pyplot as plt
import glob
import cv2
import random
from pycocotools import mask as mask_util
from PIL import Image, ImageOps
import h5py

def segments_to_rgb(segments):
    """
    Takes in list of segments, returns rgb image of segments in one image
    """
    # Assume all segment maps have the same shape
    height, width = segments[0].shape
    rgb_image = np.zeros((height, width, 3), dtype=np.float32)  # Use float32 for calculations

    # Assign a unique color to each segment
    unique_colors = [tuple(np.random.randint(0, 256, 3)) for _ in range(len(segments))]

    for idx, segment in enumerate(segments):
        color = unique_colors[idx]
        # Apply the color to the RGB image where the binary map is 1
        for channel in range(3):
            rgb_image[:, :, channel] += segment.astype(np.float32) * color[channel]

    # Clip values to valid range and convert back to uint8
    rgb_image = np.clip(rgb_image, 0, 255).astype(np.uint8)

    return rgb_image

def crop(image, annotations_image_id, size=256):
    """
    Aplp images and associated segments to square crop
    """
    cropped = False
    h, w, _ = image.shape

    # Resize and crop
    scale = max(size / h, size / w)
    new_h, new_w = int(round(h * scale)), int(round(w * scale))
    image_resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    start_h = (new_h - size) // 2
    start_w = (new_w - size) // 2
    image_cropped = image_resized[start_h:start_h + size, start_w:start_w + size]

    cropped_segs = {}
    all_segs = []

    for annot in annotations_image_id:
        seg = annot['segmentation']
        id = (annot['iscrowd'], annot['area'], annot['image_id'], annot['category_id'], annot['attribute'], annot['id'])
        seg_mask = mask_util.decode(seg)
        seg_mask_resized = cv2.resize(seg_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        seg_mask_cropped = seg_mask_resized[start_h:start_h + size, start_w:start_w + size]

        # Skip if cropped segment mask is completely out of frame
        if seg_mask_cropped.sum() == 0:
            continue

        # if passes filtering, add it
        cropped_segs[id] = seg_mask_cropped
        all_segs.append(annot)

    return image_cropped, cropped_segs, all_segs

def get_seg_map(images_dir, filename, image_id, annotation, categories, size=256):
    image_path = os.path.join(images_dir, filename)
    image_og = cv2.imread(image_path)[:, :, ::-1]  # BGR to RGB
    h, w, _ = image_og.shape
    all_segs = []
    all_ids = []
    exclude_segs = []

    annotations_image_id = [x for x in annotation if x['image_id'] == image_id]
    image, cropped_segs, annotations = crop(image_og, annotations_image_id)

    for annot in annotations:
        id = (annot['iscrowd'], annot['area'], annot['image_id'], annot['category_id'], annot['attribute'], annot['id'])
        seg_mask = cropped_segs[id]
        category_id = annot['category_id']

        for category in categories:
            if category['id'] == category_id:
                if category['type'] == 'thing':
                    if (seg_mask.sum() / (size * size)) * 100 > 0.2:
                        all_segs.append(seg_mask)
                        all_ids.append(annot['id'])

    exclude_rgb = None
    # if len(exclude_segs) != 0:
    #     exclude_rgb = segments_to_rgb(exclude_segs)
    seg_map = None
    if len(all_segs) != 0:
        seg_map = segments_to_rgb(all_segs)

    return image, seg_map, len(all_segs), all_ids, all_segs

def process_and_merge(images_dir, image_data, annotations, vis_dir, categories):
    # global vis_info
    np.random.seed(24)

    filename = image_data['file_name']
    image_id = image_data['id']

    # Generate segmentation map and related data
    rgb_img, seg_map, seg_len, all_ids, all_segs = get_seg_map(images_dir, filename, image_id, annotations, categories)
    assert len(all_ids) == len(all_segs)

    if len(all_segs) > 15:
        return
    if len(all_segs) < 3:
        return
    if len(all_segs) == 0:
        return

    # set up visual!
    try:
        fig, ax = plt.subplots(1, 2, figsize=(8, 4))

        # Show original image8ax[0].imshow(rgb_img)
        ax[0].imshow(rgb_img)
        ax[0].set_title(f'Image (id: {image_id})')
        ax[1].imshow(seg_map)
        ax[1].set_title(f'Segment Map: {seg_len}')
        # if exclude_len != 0:
        #     ax[2].set_title(f'Non-Stuff Excluded Segment: {exclude_len}')
        #     ax[2].imshow(exclude_map)

        merged_path = os.path.join(vis_dir, f"{filename}" + ".png")
        fig.savefig(merged_path, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
    except:
        print(f"Error: visualizing seg len of {seg_len}")
    return

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    # dataset
    parser.add_argument("--ann_path", type=str,
                        required=True,
                        # default="/ccn2/u/lilianch/data/entityseg/entityseg_train_01.json",
                        help="path to annotations json")
    parser.add_argument("--images_dir", type=str,
                        required=True,
                        #default="/ccn2/u/rmvenkat/data/entity_seg_dataset/entity_seg_images/entity_01_11580",
                        help="path to directory of entityseg images")

    # number to load/visualize
    parser.add_argument("--start", type=int,
                        required=True,
                        help="Image ID to begin visualizing from")
    parser.add_argument("--end", type=int,
                        required=True,
                        help="Image ID to end visualizing from")

    # save paths
    parser.add_argument("--save_dir", type=str,
                        required=True,
                        #default="/ccn2/u/lilianch/share/Entityseg-Dataset-Collection/visualize_dataset",
                        )

    args = parser.parse_args()
    np.random.seed(24)

    # Paths
    ann_path = args.ann_path  # Path to JSON
    images_dir = args.images_dir  # Directory containing the images
    vis_dir = args.save_dir
    os.makedirs(vis_dir, exist_ok=True)

    with open(ann_path, 'r') as f:
        data = json.load(f)

    entity_seg = json.load(open(ann_path))
    annotation = entity_seg['annotations']
    categories = entity_seg['categories']

    print(f'Printing from Image {args.start} to {args.end} for {args.end-args.start} total images to visualize')

    for ct, images in enumerate(data['images'][args.start:args.end]):
        process_and_merge(images_dir, images, annotation, vis_dir, categories)
        if ct % 100 == 0:
            print(f'Image {ct} of {args.end-args.start} visualized')

    print(f"Entityseg visualized saved to {vis_output_path}")
