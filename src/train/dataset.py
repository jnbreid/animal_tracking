import os
import torch
import gc

import pandas as pd
import numpa as np

from torch.utils.data import Dataset
from sklearn.metrics.pairwise import cosine_similarity

class Wildbrueck_Sim(Dataset):
  """
    Dataset class for training DistNet on the (wildlife crossing) dataset.
    Further information on the dataset can be found in the readme file.

    This dataset simulates object tracking with a Kalman filter for annotated tracklets.
    For each frame (excluding the first), it includes:
        - predicted and ground truth bounding boxes,
        - segmentation masks,
        - feature vectors,
        - time since last seen for tracked instances.

    Each data point consists of a feature vector with 3 values:
        - Cosine similarity between predicted and ground truth feature vectors.
        - IoU between predicted and ground truth segmentation masks.
        - Normalized frame distance since the object was last seen.

    Positive (same instance) and negative (different instance) samples are generated
    for each frame.

    Args:
        dataset_path (str): Root directory containing the dataset folders `predictions/` and `gt_data/`.
        cross_val_fold (int, optional): Fold number to use for cross-validation. If None, all data is used.
        train (bool): If using cross-validation, specifies whether to use the train or test split.

    Attributes:
        elements (list): A list of (input_tensor, label) pairs for training.
  """
  def __init__(self, dataset_path, cross_val_fold = None, train = True):
    self.dset_path = dataset_path

    self.vids = ['deer1', 'boar1', 'fox1', 'deer2', 'deer3', 'boar2', 'fox3', 'hare1', 'hare3', 'fox4', 'fox5', 'hare4', 'hare6', 'fox6', 'fox7', 'fox8', 'hare9', 'hare10', 'hare15', 'boar6', 'hare17', 'boar9', 'boar10', 'boar12', 'boar16', 'boar19', 'deer8', 'deer9', 'deer16', 'deer25', 'deer38', 'hare20', 'deer446', 'boar15', 'boarfox', 'deer58', 'hare23', 'deer74', 'deer452', 'hare21', 'deer10']

    self.fold = cross_val_fold
    self.split_path = None
    if self.fold is not None:
      fold_name = os.path.join('/content/drive/MyDrive/MA/Datasets/Wildbrück', f"WildbrueckenInstTrackFold{str(self.fold)}.xlsx")
      fold_frame = pd.read_excel(fold_name)

      trn = 'train' if train == True else 'test'
      fold_frame = fold_frame[fold_frame["Train_Test"] == trn]
      self.vids = list(fold_frame['Video_name'])

    self.vids.sort()
    elements = []

    self.predictions_path = os.path.join(self.dset_path, 'predictions')
    self.gt_path = os.path.join(self.dset_path, 'gt_data')

    predictions = os.listdir(self.predictions_path)

    for vid in self.vids:
      relevant_predictions =  [item for item in predictions if item.split('_')[1].split('-')[0] == vid]

      p_masks = [item for item in relevant_predictions if item.split('_')[0] == 'mask']
      p_features = [item for item in relevant_predictions if item.split('_')[0] == 'feature']
      p_boxes = [item for item in relevant_predictions if item.split('_')[0] == 'box']
      p_lastseen = [item for item in relevant_predictions if item.split('_')[0] == 'n']

      frames = list(set([item.split('_')[1] for item in p_masks]))
      frames.sort()

      tracklet = []
      for frame in frames:
        frame_masks = [item for item in p_masks if item.split('_')[1] == frame]
        frame_masks.sort()
        frame_features = [item for item in p_features if item.split('_')[1] == frame]
        frame_features.sort()
        frame_boxes = [item for item in p_boxes if item.split('_')[1] == frame]
        frame_boxes.sort()
        frame_lastseen = [item for item in p_lastseen if item.split('_')[1] == frame]
        frame_lastseen.sort()

        ids = list(set([item.split('_')[2].split('.')[0] for item in frame_masks]))
        ids.sort()


        for id in ids:

          id_masks = [item for item in frame_masks if item.split('_')[2].split('.')[0] == id]
          id_masks.sort()
          id_features = [item for item in frame_features if item.split('_')[2].split('.')[0] == id]
          id_features.sort()
          id_boxes = [item for item in frame_boxes if item.split('_')[2].split('.')[0] == id]
          id_boxes.sort()
          id_lastseen = [item for item in frame_lastseen if item.split('_')[2].split('.')[0] == id]
          id_lastseen.sort()

          for i in range(len(id_masks)):

            other_pos_inst = frame_masks.copy()
            other_pos_inst.remove(id_masks[i])

            tracklet.append([id_masks[i], id_features[i], id_boxes[i], id_lastseen[i], other_pos_inst])

      elements.append(tracklet)
    gc.collect()

    # list that contains elements used for training 
    self.elements = []

    id_mapping = {}
    itemid = 0
    for i in range(len(elements)):
      element = elements[i]
      for j in range(len(element)):

        id_mapping[itemid] = [i,j]

        itemid = itemid + 1

    ln = 0
    for item in elements:
      ln  = ln + len(item)

    for id in range(ln):
      i, j = id_mapping[id]
      pc_mask, pc_feature, pc_box, pc_idlastseen, other_elem_list = elements[i][j]

      pc_mask_path = os.path.join(self.predictions_path, pc_mask)
      pc_feature_path = os.path.join(self.predictions_path, pc_feature)
      pc_box_path = os.path.join(self.predictions_path, pc_box)
      pc_idlastseen_path = os.path.join(self.predictions_path, pc_idlastseen)

      pred_mask = np.load(pc_mask_path)
      pred_feature = np.load(pc_feature_path)
      pred_box = np.load(pc_box_path).astype(np.int32).reshape((-1))
      # if work with box, rememder other formating (true_gt_box[2] = true_gt_box[0]+true_gt_box[2]     true_gt_box[3] = true_gt_box[1]+true_gt_box[3])
      pred_lastseen = np.load(pc_idlastseen_path)#[0] #

      gt_mask_path = os.path.join(self.gt_path, pc_mask)
      gt_feature_path = os.path.join(self.gt_path, pc_feature)
      gt_box_path = os.path.join(self.gt_path, pc_box)

      gt_mask = np.load(gt_mask_path)
      gt_feature = np.load(gt_feature_path)
      gt_box = np.load(gt_box_path)

      # for positive samples
      feature_dist = cosine_similarity(pred_feature.reshape((1,-1)), gt_feature.reshape((1,-1)))[0,0]
      gt_box[2] = gt_box[2]+gt_box[0]
      gt_box[3] = gt_box[3]+gt_box[1]
      IoU = iou_mask([gt_box], [gt_mask], [pred_box], [pred_mask])[0,0]

      tens = np.asarray([feature_dist, IoU, pred_lastseen[0]], dtype = np.float32)
      tens = torch.from_numpy(tens)
      label = np.asarray([0], dtype = np.float32)
      label = torch.from_numpy(label)

      self.elements.append([tens, label])

      for item in other_elem_list:

        other_mask = item#random.choice(other_elem_list)
        other_feature = f"feature_{other_mask.split('_')[1]}_{other_mask.split('_')[2]}"
        other_box = f"box_{other_mask.split('_')[1]}_{other_mask.split('_')[2]}"

        gt_mask_path = os.path.join(self.gt_path, other_mask)
        gt_feature_path = os.path.join(self.gt_path, other_feature)
        gt_box_path = os.path.join(self.gt_path, other_box)


        gt_mask = np.load(gt_mask_path)
        gt_feature = np.load(gt_feature_path)
        gt_box = np.load(gt_box_path).copy()

        feature_dist = cosine_similarity(pred_feature.reshape((1,-1)), gt_feature.reshape((1,-1)))[0,0]
        gt_box[2] = gt_box[2]+gt_box[0]
        gt_box[3] = gt_box[3]+gt_box[1]
        IoU = iou_mask([gt_box], [gt_mask], [pred_box], [pred_mask])[0,0]

        tens = np.asarray([feature_dist, IoU, pred_lastseen[0]], dtype = np.float32)
        tens = torch.from_numpy(tens)
        label = np.asarray([1], dtype = np.float32)
        label = torch.from_numpy(label)

        self.elements.append([tens, label])
    gc.collect()

  #def full_set(self):
  #  for i in range(self.len)

  def __len__(self):
    return len(self.elements)


  def __getitem__(self, id):

    output, label = self.elements[id]

    return output, label
