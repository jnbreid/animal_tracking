from torch.utils.data import Dataset
import os
import pandas as pd
import numpy as np
import random
import torch
from math import isnan

from sklearn.metrics.pairwise import cosine_similarity

from utils import iou_mask

import gc

def key_fkt(x):
  return x[0][1]

class G_MOT40_Sim(Dataset):
  def __init__(self, dataset_path):
    self.dset_path = dataset_path
    #self.pre_path = pre_path
    #self.annot_path = os.path.join(self.dset_path, 'animals_tracking.csv')
    #self.vids = ['deer1', 'boar1', 'fox1', 'deer2', 'deer3', 'boar2', 'fox3', 'hare1', 'hare3', 'fox4', 'fox5', 'hare4', 'hare6', 'fox6', 'fox7', 'fox8', 'hare9', 'hare10', 'hare15', 'boar6', 'hare17', 'boar9', 'boar10', 'boar12', 'boar16', 'boar19', 'deer8', 'deer9', 'deer16', 'deer25', 'deer38', 'hare20', 'deer446', 'boar15', 'boarfox', 'deer58', 'hare23', 'deer74', 'deer452', 'hare21', 'deer10']

    self.vids = ['bird-0', 'bird-1', 'bird-2', 'bird-3','stock-0', 'stock-2',  'fish-2', 'fish-3']#['bird-0', 'bird-1', 'bird-2', 'bird-3']#['stock-0', 'stock-2']#['fish-0', 'fish-2']#['bird-1', 'bird-2', 'bird-3', 'fish-0', 'fish-2', 'stock-0', 'stock-2']#'bird-0',


    self.vids.sort()
    elements = []

    self.predictions_path = os.path.join(self.dset_path, 'content/predictions')
    self.gt_path = os.path.join(self.dset_path, 'masks')

    predictions = os.listdir(self.predictions_path)

    #gts = os.listdir(self.gt_path)


    for vid in self.vids:
      relevant_predictions = [item for item in predictions if item.split('_')[1] == vid] #[item for item in predictions if item.split('_')[1].split('-')[0] == vid]

      #test = [item.split('_')[0] for item in relevant_predictions]

      p_masks = [item for item in relevant_predictions if item.split('_')[0] == 'mask']
      p_features = [item for item in relevant_predictions if item.split('_')[0] == 'feature']
      p_boxes = [item for item in relevant_predictions if item.split('_')[0] == 'box']
      p_lastseen = [item for item in relevant_predictions if item.split('_')[0] == 'n']



      frames = list(set([item.split('_')[2] for item in p_masks]))
      frames.sort()

      """
      c_mask_names = list(set([item.split('_')[1] for item in p_masks]))
      c_mask_names.sort()
      c_feature_names = list(set([item.split('_')[1] for item in p_features]))
      c_feature_names.sort()
      c_box_names = list(set([item.split('_')[1] for item in p_boxes]))
      c_box_names.sort()

      for x in range(len(c_mask_names)):
        print(c_mask_names[x], c_feature_names[x], c_box_names[x])

      return
      """
      tracklet = []
      for frame in frames:
        frame_masks = [item for item in p_masks if item.split('_')[2] == frame]
        frame_masks.sort()
        frame_features = [item for item in p_features if item.split('_')[2] == frame]
        frame_features.sort()
        frame_boxes = [item for item in p_boxes if item.split('_')[2] == frame]
        frame_boxes.sort()
        frame_lastseen = [item for item in p_lastseen if item.split('_')[2] == frame]
        frame_lastseen.sort()

        #print(frame_masks)
        #print(frame_features)
        #print(frame_boxes)

        #continue
        ids = list(set([item.split('_')[3].split('.')[0] for item in frame_masks]))
        ids.sort()



        for id in ids:

          id_masks = [item for item in frame_masks if item.split('_')[3].split('.')[0] == id]
          id_masks.sort()
          id_features = [item for item in frame_features if item.split('_')[3].split('.')[0] == id]
          id_features.sort()
          id_boxes = [item for item in frame_boxes if item.split('_')[3].split('.')[0] == id]
          id_boxes.sort()
          id_lastseen = [item for item in frame_lastseen if item.split('_')[3].split('.')[0] == id]
          id_lastseen.sort()

          #print(id_masks)
          #print(len(id_masks))

          """
          print(id_masks)
          print(id_features)
          print(id_boxes)
          print(id_lastseen)
          """
          for i in range(len(id_masks)):


            other_pos_inst = frame_masks.copy()
            other_pos_inst.remove(id_masks[i])

            tracklet.append([id_masks[i], id_features[i], id_boxes[i], id_lastseen[i], other_pos_inst])

      elements.append(tracklet)
    gc.collect()

    print('loading elements finished')

    """
    """
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
      if id%250 == 0:
        print(f"{id}/{ln}")

      i, j = id_mapping[id]
      pc_mask, pc_feature, pc_box, pc_idlastseen, other_elem_list = elements[i][j]

      pc_mask_path = os.path.join(self.predictions_path, pc_mask)
      pc_feature_path = os.path.join(self.predictions_path, pc_feature)
      pc_box_path = os.path.join(self.predictions_path, pc_box)
      pc_idlastseen_path = os.path.join(self.predictions_path, pc_idlastseen)

      pred_mask = np.load(pc_mask_path).astype(np.uint8)
      pred_feature = np.load(pc_feature_path)
      pred_box = np.load(pc_box_path).astype(np.int32).reshape((-1))

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

      if isnan(feature_dist)==False or isnan(IoU)==False or isnan(pred_lastseen[0])==False:
        for extr in range(4):
          self.elements.append([tens, label])


      other_candidates = []
      for item in other_elem_list:

        other_mask = item#random.choice(other_elem_list))

        other_feature = f"feature_{other_mask.split('_')[1]}_{other_mask.split('_')[2]}_{other_mask.split('_')[3]}"
        other_box = f"box_{other_mask.split('_')[1]}_{other_mask.split('_')[2]}_{other_mask.split('_')[3]}"

        gt_mask_path = os.path.join(self.gt_path, other_mask)
        gt_feature_path = os.path.join(self.gt_path, other_feature)
        gt_box_path = os.path.join(self.gt_path, other_box)


        gt_mask = np.load(gt_mask_path)
        gt_feature = np.load(gt_feature_path)
        gt_box = np.load(gt_box_path).copy()

        feature_dist = cosine_similarity(pred_feature.reshape((1,-1)), gt_feature.reshape((1,-1)))[0,0]

        #print(np.unique(gt_mask), np.unique(pred_mask))
        gt_box[2] = gt_box[2]+gt_box[0]
        gt_box[3] = gt_box[3]+gt_box[1]
        IoU = iou_mask([gt_box], [gt_mask], [pred_box], [pred_mask])[0,0]

        tens = np.asarray([feature_dist, IoU, pred_lastseen[0]], dtype = np.float32)
        tens = torch.from_numpy(tens)
        label = np.asarray([1], dtype = np.float32)
        label = torch.from_numpy(label)

        if isnan(feature_dist) or isnan(IoU) or isnan(pred_lastseen[0]):
          continue

        other_candidates.append([tens,label])

      random.shuffle(other_candidates)
      other_candidates.sort(reverse=True, key=key_fkt)
      if len(other_candidates) > 40:
        other_candidates = other_candidates[:40]
 

      for chosen_item in other_candidates:
        self.elements.append(chosen_item)
    del elements
    gc.collect()



  def __len__(self):
    return len(self.elements)


  def __getitem__(self, id):
    output, label = self.elements[id]

    return output, label