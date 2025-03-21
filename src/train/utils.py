from torch.utils.data import Dataset
import os
import pandas as pd
import numpy as np
import cv2
from PIL import Image

from skimage.draw import polygon, polygon2mask


"""
This class is used as a helper dataset class to load ground truth data from the wildlife crossing dataset to use
as input for a kalman tracker.
"""
class Precompute_helper(Dataset):
  def __init__(self, dataset_path, dset_split = 0):
    self.dset_path = dataset_path

    self.annot_path = os.path.join(self.dset_path, 'animals_tracking.csv')
    if dset_split != 0:
      self.split_path = os.path.join(self.dset_path, 'WildbrueckenInstTrackFold1.xlsx')

    self.dframe = pd.read_csv(self.annot_path)

    vid_ids = self.dframe['file_attributes']

    vid_ids = [int(item.split('"')[3]) for item in vid_ids]

    vid_arr = np.asarray(vid_ids)
    vids = list(np.unique(vid_arr))

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


    self.dframe.insert(2, "vid_id", vid_ids)
    self.dframe.drop("file_attributes", axis=1, inplace=True)

    tracks = self.dframe["region_attributes"].to_list()
    tracks = [int(item.split(":")[2].strip('"{}')) for item in tracks]
    self.dframe.insert(5, "track_id", tracks)

    self.dframe.drop("region_attributes", axis=1, inplace=True)
    # number of elements per image also not needed
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
    elements = self.item_mapping[idx]

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

    return img_paths, full_ids, full_boxes, full_masks
  

def precompute_gt_box(d_set, save_path, mask_predictor, extractor):
  for e in range(d_set.__len__()):
    elem = e
    print(f"{elem+1}/{d_set.__len__()}")
    img_paths, full_ids, full_boxes, full_masks = d_set.__getitem__(elem)

    for i in range(len(img_paths)):

      current_img_path = img_paths[i]
      current_ids = full_ids[i]
      print(f"{i+1}/{len(img_paths)}  - ", current_img_path)

      current_boxes = full_boxes[i].copy()

      curr_img = cv2.imread(current_img_path)
      image_rgb = cv2.cvtColor(curr_img, cv2.COLOR_BGR2RGB)
      mask_predictor.set_image(image_rgb)

      megadesc_img = Image.open(current_img_path)

      img_name = current_img_path.split('/')[-1].split('.')[0]

      for j in range(current_boxes.shape[0]):
        c_box = current_boxes[j,:].copy()
        c_box[2] = c_box[0] + c_box[2]
        c_box[3] = c_box[1] + c_box[3]

        gt_box = current_boxes[j,:].copy()

        c_id = current_ids[j]

        pred_mask, _, _ = mask_predictor.predict(box=c_box, multimask_output=False)
        pred_mask = pred_mask[0,:,:].astype(np.int8)

        current_crop = megadesc_img.crop(c_box)
        feature_vect = extractor(current_crop)
        feature_vect = feature_vect.detach().cpu().numpy().reshape(-1)

        mask_path = os.path.join(save_path, f"mask_{img_name}_{str(c_id)}.npy")
        feature_path = os.path.join(save_path, f"feature_{img_name}_{str(c_id)}.npy")

        box_path = os.path.join(save_path, f"box_{img_name}_{str(c_id)}.npy")

        np.save(box_path, gt_box)

        np.save(mask_path, pred_mask)
        np.save(feature_path, feature_vect)

import sys
sys.path.insert(0, '..')
from track import KalmanBoxTracker

def precompute_pred_box(dataset_path, pre_path, pred_save_path):

  s_dset_path = dataset_path
  s_pre_path = pre_path
  s_annot_path = os.path.join(s_dset_path, 'animals_tracking.csv')
  s_vids = ['deer1', 'boar1', 'fox1', 'deer2', 'deer3', 'boar2', 'fox3', 'hare1', 'hare3', 'fox4', 'fox5', 'hare4', 'hare6', 'fox6', 'fox7', 'fox8', 'hare9', 'hare10', 'hare15', 'boar6', 'hare17', 'boar9', 'boar10', 'boar12', 'boar16', 'boar19', 'deer8', 'deer9', 'deer16', 'deer25', 'deer38', 'hare20', 'deer446', 'boar15', 'boarfox', 'deer58', 'hare23', 'deer74', 'deer452', 'hare21', 'deer10']

  # load all relevant masks and feature vectors
  elements = os.listdir(s_pre_path)
  masks = [item for item in elements if item.split('_')[0] == 'mask']
  features = [item for item in elements if item.split('_')[0] == 'feature']

  # here same as in normal Wildbrück: load relevant infos and save in dataframe
  s_dframe = pd.read_csv(s_annot_path)

  vid_ids = s_dframe['file_attributes']

  vid_ids = [int(item.split('"')[3]) for item in vid_ids]

  vid_arr = np.asarray(vid_ids)
  vids = list(np.unique(vid_arr))

  full_file_names = s_dframe["filename"]

  ann_ids = list(np.arange(len(full_file_names)))
  s_dframe.insert(0, 'ann_id', ann_ids)

  s_item_mapping = {}

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

    s_item_mapping[vid] = current_vid


  s_dframe.insert(2, "vid_id", vid_ids)
  # video number now saved in vid_id, so file_attributes not needed
  s_dframe.drop("file_attributes", axis=1, inplace=True)


  tracks = s_dframe["region_attributes"].to_list()
  tracks = [int(item.split(":")[2].strip('"{}')) for item in tracks]
  s_dframe.insert(5, "track_id", tracks)

  s_dframe.drop("region_attributes", axis=1, inplace=True)
  # number of elements per image also not needed
  s_dframe.drop("region_count", axis=1, inplace=True)

  poly = s_dframe["region_shape_attributes"]

  xy_points = [[item.split("all_points")[1], item.split("all_points")[2]] for item in poly]

  x_points = [item[0].strip('_xy":[],{}') for item in xy_points]
  y_points = [item[1].strip('_xy":[],{}') for item in xy_points]

  xy_strings = []
  for i in range(len(x_points)):
    c_string = x_points[i] + 'xy' + y_points[i]
    xy_strings.append(c_string)
  s_dframe.insert(5, "poly_points", xy_strings)

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
  #print(box_strings)
  s_dframe.insert(4, "bbox", box_strings)

  s_dframe.drop("region_shape_attributes", axis=1, inplace=True)
  
  """
  now for each tracklet kalman filter is applied and
  """

  s_elements = []

  counter = 0
  for vid in s_vids:

    counter = counter + 1
    print(f"{counter}/{len(s_vids)} vids")

    all_frames = list(s_dframe['filename'])

    relevant_frames = list(set([item for item in all_frames if item.split('-')[0] == vid]))

    relevant_frames.sort()
  
    fr_masks = [item for item in masks if item.split('_')[1].split('-')[0] == vid]

    fr_masks.sort()

    fr_features = [item for item in features if item.split('_')[1].split('-')[0] == vid]
  
    ids = list(set([item.split('_')[2].split('.')[0] for item in fr_masks]))
    ids.sort()

    c1 = 0
    for id in ids:

      spec_masks = [item for item in fr_masks if item.split('_')[2].split('.')[0] == id]
      spec_masks.sort()
      spec_features = [item for item in fr_features if item.split('_')[2].split('.')[0] == id]
      spec_features.sort()

      c1 = c1 + 1
      print(f"     {c1}/{len(ids)} ids ({len(spec_masks)} detections)")


      tracker = None

      for i in range(len(relevant_frames)):
        name = relevant_frames[i]

        box = s_dframe[s_dframe['filename'] == name]
        box = box[box['track_id'] == int(id)]

        reid_name = name.split('.')[0]

        c_spec_masks = [item for item in spec_masks if item.split('_')[1] == reid_name]
        c_spec_features = [item for item in spec_features if item.split('_')[1] == reid_name]

        if len(c_spec_masks) > 0:
          curr_mask_path = os.path.join(s_pre_path, c_spec_masks[0])
          curr_feature_path = os.path.join(s_pre_path, c_spec_features[0])

          curr_mask = np.load(curr_mask_path)
          curr_feature = np.load(curr_feature_path)


          bbox = list(map(int, list(box['bbox'])[0].split(',')))
          bbox[2] = bbox[0] + bbox[2]
          bbox[3] = bbox[1] + bbox[3]

          if tracker is None:
            tracker = KalmanBoxTracker(bbox, curr_mask, curr_feature)
          else:
            tracker.predict()

            pred_box, pred_mask, pred_feature = tracker.get_full_state()

            c_box = pred_box.copy().reshape((4)).astype(np.int32)
            c_box[2] = c_box[2]-c_box[0]
            c_box[3] = c_box[3]-c_box[1]

            pred_mask_path = os.path.join(pred_save_path, f"mask_{name.split('.')[0]}_{str(id)}.npy")
            pred_feature_path = os.path.join(pred_save_path, f"feature_{name.split('.')[0]}_{str(id)}.npy")

            pred_box_path = os.path.join(pred_save_path, f"box_{name.split('.')[0]}_{str(id)}.npy")
            n_last_path = os.path.join(pred_save_path, f"n_{name.split('.')[0]}_{str(id)}.npy")


            n_last_seen = np.asarray([tracker.time_since_update/10])


            np.save(n_last_path, n_last_seen)
            np.save(pred_mask_path, pred_mask)
            np.save(pred_feature_path, pred_feature)
            np.save(pred_box_path, pred_box)

            tracker.update(bbox, curr_mask, curr_feature)

        else:
          if tracker is not None:
            tracker.predict()
            n_last_seen = tracker.time_since_update
            # drop element, if it was not seen for more than 24 frames
            if n_last_seen > 24:
              tracker = None