from torch.utils.data import Dataset
import os
import pandas as pd
import numpy as np
import cv2
import json

from skimage.draw import polygon, polygon2mask


"""
Dataset class for evaluation of the wildlife crossing dataset

"""
class Wildbrueck(Dataset):
  """
    PyTorch-compatible dataset class for loading and processing the wildlife crossing dataset.
    The class handles loading of annotations, bounding boxes, segmentation masks, and optional cross-validation folds.

    
  """
  def __init__(self, dataset_path, cross_val_fold = None, train=False):
    self.dset_path = dataset_path

    self.annot_path = os.path.join(self.dset_path, 'animals_tracking.csv')
    self.image_size = None

    self.vids = list(map(str,list(np.arange(0,41))))

    self.fold = cross_val_fold
    self.split_path = None
    if self.fold is not None:
      fold_name = os.path.join('/content/drive/MyDrive/MA/Datasets/Wildbrück', f"WildbrueckenInstTrackFold{str(self.fold)}.xlsx")
      fold_frame = pd.read_excel(fold_name)

      trn = 'train' if train == True else 'test'
      fold_frame = fold_frame[fold_frame["Train_Test"] == trn]
      self.vids = list(fold_frame['Video_number'])

    mega_detections_file = os.path.join(self.dset_path, 'megadetector.json')
    with open(mega_detections_file, 'r') as f:
      data = json.load(f)
      self.mega_annotations = data['annotations']
    for i in range(len(self.mega_annotations)):
      self.mega_annotations[i]['img_id'] = self.mega_annotations[i]['img_id'].split('/')[-1]

    self.dframe = pd.read_csv(self.annot_path)

    vid_ids = self.dframe['file_attributes']

    vid_ids = [int(item.split('"')[3]) for item in vid_ids]

    vid_arr = np.asarray(vid_ids)
    vids = self.vids
    full_file_names = self.dframe["filename"]

    ann_ids = list(np.arange(len(full_file_names)))
    self.dframe.insert(0, 'ann_id', ann_ids)

    self.item_mapping = {}

    for vid in vids:
      img_names = []
      for p in range(len(vid_ids)):
        if vid_ids[p] == vid:
          img_names.append(full_file_names[p])
      img_names = list(set(img_names))
      img_names.sort()

      current_vid = []
      for elem in img_names:
        current_elements = []
        for l in range(len(vid_ids)):
          if full_file_names[l] == elem:
            current_elements.append(l)
        current_vid.append(current_elements)

      self.item_mapping[vid] = current_vid

    self.double_mapping = {}

    for i in range(len(self.item_mapping.keys())):
      self.double_mapping[i] = list(self.item_mapping.keys())[i]

    self.dframe.insert(2, "vid_id", vid_ids)
    self.dframe.drop("file_attributes", axis=1, inplace=True)

    tracks = self.dframe["region_attributes"].to_list()
    tracks = [int(item.split(":")[2].strip('"{}')) for item in tracks]
    self.dframe.insert(5, "track_id", tracks)

    self.dframe.drop("region_attributes", axis=1, inplace=True)
    self.dframe.drop("region_count", axis=1, inplace=True)

    poly = self.dframe["region_shape_attributes"]

    xy_points = [[item.split("all_points")[1], item.split("all_points")[2]] for item in poly]

    x_points = [item[0].strip('_xy":[],{}') for item in xy_points]
    y_points = [item[1].strip('_xy":[],{}') for item in xy_points]

    xy_strings = []
    for i in range(len(x_points)):
      c_string = x_points[i] + 'xy' + y_points[i]
      xy_strings.append(c_string)
    self.dframe.insert(5, "poly_points", xy_strings)

    x_arrays = [np.asarray([int(elem) for elem in items.split(',')]) for items in x_points]
    y_arrays = [np.asarray([int(elem) for elem in items.split(',')]) for items in y_points]

    x_min = np.asarray([item.min() for item in x_arrays])
    x_max = np.asarray([item.max() for item in x_arrays])
    y_min = np.asarray([item.min() for item in y_arrays])
    y_max = np.asarray([item.max() for item in y_arrays])

    boxes = np.asarray([x_min, y_min, x_max-x_min, y_max-y_min])

    box_strings = []
    for i in range(boxes.shape[1]):
      c_box = boxes[:,i]
      c_string = f'{c_box[0]},{c_box[1]},{c_box[2]},{c_box[3]}'
      box_strings.append(c_string)

    self.dframe.insert(4, "bbox", box_strings)

    self.dframe.drop("region_shape_attributes", axis=1, inplace=True)

  def __len__(self):
    return len(self.item_mapping.keys())

  def __getitem__(self, idx):
    elements = self.item_mapping[self.double_mapping[idx]]

    c_id = self.double_mapping[idx]

    img_paths = []
    full_ids = []
    full_boxes = []
    full_masks = []

    for item in elements:
        dframe_img = self.dframe.iloc[item]

        c_img_name = dframe_img['filename'].iloc[0]
        c_img_path = os.path.join(self.dset_path, 'images', c_img_name)
        c_image = cv2.imread(c_img_path)
        y_s, x_s, channel = c_image.shape

        ids = dframe_img['track_id'].to_numpy()
        boxes = dframe_img['bbox']
        c_masks = dframe_img['poly_points']

        bbx = []
        bmasks = []

        for i in range(len(boxes)):
            box = boxes.iloc[i]
            items = list(map(int,box.split(',')))
            bbx.append(items)

            x_points, y_points = c_masks.iloc[i].split('xy')
            x_points = np.asarray(list(map(int, x_points.split(','))))
            y_points = np.asarray(list(map(int, y_points.split(','))))
            points = np.stack([x_points, y_points], axis = 1)
            c_mask = polygon2mask((x_s, y_s), points).transpose()
            bmasks.append(c_mask)

        bbox = np.asarray(bbx)
        bmasks = np.asarray(bmasks)

        full_boxes.append(bbox)
        full_masks.append(bmasks)
        full_ids.append(ids)
        img_paths.append(c_img_path)

    return img_paths, full_boxes, full_masks