# Consolidation
Run make_dataset_h5.py to consolidate data from multiple h5 directory sources into a format suitable for evaluation

Additionally added centroid to h5 to help standardize

## TO RUN: 
```python
python make_dataset_h5.py --h5_dirs {list of h5 directories to consolidate} --h5_save {path to h5 file to save (will make if not existing)}
```

## OPTIONAL: Also visualize your consolidated h5 file
If you want to visualize what is saved in the consolidated h5 file automatically after running the above, just add the following arguments to the end of the command
```python
--vis --vis_dir {directory to save visualizations}
```
It will save the rgb image, segment map, and each individual segment with its centroids for each image saved in the consolidated dataset
See example output vis [here](http://node4-ccn2cluster.stanford.edu:8666/cgi-bin/file-explorer/?dir=%2Fccn2%2Fu%2Flilianch%2Fshare%2FEntityseg-Dataset-Collection%2Fconsolidation%2Fvis&patterns_show=*&patterns_highlight=&w=1600&h=600&n=1&showmedia=1&mr=)

# ANNOTATION - Merging, Deleting Segments, and Re-Annotating Centroids
# Segment Re-annotation Tool

A Gradio-based application for re-annotating image segments and modifying centroids in H5 datasets.

## Setup

### 1. Copy Dataset
First, copy the current updated dataset H5 file to your working directory:

```bash
cp /ccn2/u/lilianch/share/Entityseg-Dataset-Collection/consolidation/centroid_test.h5 <your_directory>
```

### 2. Run the Application
Launch the Gradio app:

```bash
python gradio_merge_and_centroids.py
```

## Usage Guide

### Loading Data

1. **Load H5 File**
   - Enter the file path of your copied dataset H5 file
   - Click "Load H5 File"

2. **Log File (Optional)**
   - Currently not functional - skip this step

3. **Load Image**
   - Enter the image ID or name to re-annotate
   - Examples: `1007` or `1_image1007` (for image `1_image1007.png`)
   - Click "Load Image"

### Re-annotation Features

#### Deleting Segments
1. Click on the segment you want to delete in the **left panel**
2. Click "Delete Segment" button
3. The segment will be permanently removed from the dataset

#### Modifying Centroids
1. Click on the segment whose centroid you want to change in the **left panel**
2. Click "Change Centroid" button in the **right panel**
3. Click anywhere on the image in the **right panel** to set the new centroid location
4. Click "Confirm Centroid" to save the changes

### Visual Indicators

- **Left Panel**: Shows all segments overlaid on the RGB image
- **Right Panel**: Shows selected segment with:
  - **Yellow**: Currently selected segment
  - **Blue dot**: Current centroid position
  - **Magenta dot**: New centroid position (in edit mode)

## Notes

- All changes are saved directly to the H5 file
- The application supports segment deletion and centroid modification
- Make sure to work on a copy of your dataset to preserve the original
