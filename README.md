# Entityseg-Dataset-Collection
Dataset collection gradio for Entityseg Segments for a more spelke-aligned dataset

## SETUP + RUN
```python
pip install -e . 
```
```python
python entityseg_filter_gradio.py
```

## How to use:
### Enter in paths:
* Path to JSON annotations file (base: /ccn2/u/lilianch/data/entityseg/)
** Then one of either (entityseg_train_01.json, entityseg_train_02.json, entityseg_train_03.json)
* Path to Image Directory (base: /ccn2/u/rmvenkat/data/entity_seg_dataset/)
** Then one of either (entity_01_11580, entity_02_11598, entity_03_10049)
** This is just the associated rgb images to load with the json annotations, so ensure it lines up with the correct json from the previous point
* Path to H5 save directory (your choice)
* Path to discard folder (your choice)

### Press Load Data
* It will load in one sample image with original rgb on left and with all segments overlaid on the right

### Annotate
* The image on the right starts off with all segments–to discard a segment, just press the segment on the right image to discard it (it should update the visualizations); repeat this with all undesired segments until you have the final set
* Press Save Image and Segments

### Feel free to discard entire example
* Some (actually, most) examples may be too messy or not have ideal layout, feel free to press discard on those–don’t worry about discarding too many, just search for good examples

## Validate
* To make sure everything is aligned, feel free to periodically run the validation like so:
```python
python entityseg_filter_gradio_validate.py --h5_dir {path to h5_dir} --save_dir {path to save_dir} --ann_pat {associated ann path that you used for gradio} --img_dir {associated image dir} 
```
* It should visualize the image and only the segments that you chose (as saved in the h5 directory)

## Visual Examples
![alt text](https://github.com/lilianchs/Entityseg-Dataset-Collection/images/entityseg_example_segments.png "Visual Examples")