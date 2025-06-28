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
from collection.visualize_entityseg import segments_to_rgb, crop, get_seg_map
from ccwm.utils.segment import compute_segment_centroids
import cv2
import torch
import h5py as h5
import sys

SPLIT_1 = {'imgs': '/ccn2/u/rmvenkat/data/entity_seg_dataset/entity_seg_images/entity_01_11580',
           'anns': '/ccn2/u/lilianch/data/entityseg/entityseg_train_01.json', 'num': 1}
SPLIT_2 = {'imgs': '/ccn2/u/rmvenkat/data/entity_seg_dataset/entity_seg_images/entity_02_11598',
           'anns': '/ccn2/u/lilianch/data/entityseg/entityseg_train_02.json', 'num': 2}
SPLIT_3 = {'imgs': '/ccn2/u/rmvenkat/data/entity_seg_dataset/entity_seg_images/entity_03_10049',
           'anns': '/ccn2/u/lilianch/data/entityseg/entityseg_train_03.json', 'num': 3}
SPLITS = [SPLIT_1, SPLIT_2, SPLIT_3]

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

    return image, seg_map, all_segs

def main():
    parser = argparse.ArgumentParser(
        description="Save h5 files in consolidated single h5 files. Visualization optional.")
    parser.add_argument("--h5_dirs", type=str, nargs='+', required=True,
                        help="All directories containing h5 files to consolidate")
    parser.add_argument("--h5_save", type=str, required=True, help="Where to save h5 consolidated file")
    args = parser.parse_args()

    h5_dirs = args.h5_dirs
    h5_save = args.h5_save

    # validations
    if os.path.exists(args.h5_save):
        response = input(f"File '{args.h5_save}' already exists. Overwrite? [y/N]: ").strip().lower()
        if response not in ("y", "yes"):
            print("Aborting to avoid overwriting.")
            sys.exit(0)

    # ASSUMES each h5 directory contains data for only one split and only h5 files
    h5_dir_split = []
    for h5_dir in h5_dirs:
        # Sample h5 file to get basename
        h5_files = [f for f in os.listdir(h5_dir) if f.endswith('.h5')]
        if not h5_files:
            raise ValueError(f"No .h5 files found in {h5_dir}")
        sample_file = h5_files[0]
        basename = os.path.splitext(sample_file)[0]

        # Check which split contains this basename
        split_found = False
        for i, split in enumerate(SPLITS):
            img_files = {os.path.splitext(f)[0] for f in os.listdir(split['imgs'])}
            if basename in img_files:
                h5_dir_split.append({'split': split, 'h5_dir': h5_dir})
                split_found = True
                break

        if not split_found:
            raise ValueError(f"File '{basename}' not found in any split")

    # Create consolidated h5 file
    ct_vis = 0
    with h5py.File(h5_save, 'w') as consolidated_h5:

        for item in h5_dir_split:
            h5_dir = item['h5_dir']
            img_dir = item['split']['imgs']
            anns = item['split']['anns']
            split_num = item['split']['num']

            h5_files = glob.glob(os.path.join(h5_dir, '*.h5'))
            assert h5_files, f"No .h5 files found in directory: {h5_dir}"

            entity_seg_split = json.load(open(anns))
            annotations = entity_seg_split['annotations']

            for i, h5_file in enumerate(h5_files):
                if i % 25 == 0:
                    print(f'Processing {i}/{len(h5_files)} files from split {split_num}')

                with h5py.File(h5_file, 'r') as data:
                    image_id = data['image_id'][()]
                    valid_segment_ids = data['valid_segment_ids'][()]

                filename = os.path.splitext(os.path.basename(h5_file))[0]
                img_path = glob.glob(os.path.join(img_dir, filename + ".*"))[0]
                assert img_path, f"Image file not found for {filename}"

                valid_annotations = [annotations[id] for id in valid_segment_ids]
                assert len(valid_annotations) == len(valid_segment_ids), \
                    f"Mismatch in valid annotations for image {filename}"

                rgb_img, seg_map, all_segs = get_seg_map(img_path, valid_annotations)
                assert all_segs, f"Empty valid segments"

                # H5 save format
                top_level_key = f"{split_num}_image{image_id}"
                img_group = consolidated_h5.create_group(top_level_key)

                img_group.create_dataset('filename', data=filename.encode('utf-8'))
                img_group.create_dataset('rgb', data=rgb_img)
                img_group.create_dataset('seg_map', data=seg_map)

                stacked_segments = np.stack(all_segs, axis=0)  # Shape: (num_segments, height, width)
                img_group.create_dataset('segment', data=stacked_segments)

                centroids = compute_segment_centroids(torch.from_numpy(stacked_segments))
                img_group.create_dataset('centroid', data=centroids)

                img_group.create_dataset('segment_ids', data=valid_segment_ids)

                ct_vis += 1

    print(f'Consolidated {ct_vis} examples from {len(h5_dirs)} directories into {h5_save}')

if __name__ == "__main__":
    main()