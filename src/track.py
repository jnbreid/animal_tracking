import numpy as np
import time
from filterpy.kalman import KalmanFilter

from src.utils import linear_assignment, iou_batch, iou_mask, decision_matr, convert_bbox_to_z, convert_x_to_bbox, insert_mask, cutout_mask


class KalmanBoxTracker(object):
  """
  This class represents the internal state of individual tracked objects observed as bbox.
  (this class is taken from https://github.com/abewley/sort/blob/master/sort.py and extended to work with feature vectors and segmentation masks)
  """
  count = 0
  def __init__(self,bbox, mask, feature_vect):
    """
    Initialises a tracker using initial bounding box.
    """
    #define constant velocity model
    self.kf = KalmanFilter(dim_x=7, dim_z=4)
    self.kf.F = np.array([[1,0,0,0,1,0,0],[0,1,0,0,0,1,0],[0,0,1,0,0,0,1],[0,0,0,1,0,0,0],  [0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1]])
    self.kf.H = np.array([[1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0]])

    self.kf.R[2:,2:] *= 10.
    self.kf.P[4:,4:] *= 1000. #give high uncertainty to the unobservable initial velocities
    self.kf.P *= 10.
    self.kf.Q[-1,-1] *= 0.01
    self.kf.Q[4:,4:] *= 0.01

    self.kf.x[:4] = convert_bbox_to_z(bbox)
    self.time_since_update = 0
    self.id = KalmanBoxTracker.count
    KalmanBoxTracker.count += 1
    self.history = []
    self.hits = 0
    self.hit_streak = 0
    self.age = 0

    self.mask = mask
    self.img_size = mask.shape
    self.feature_vect = feature_vect


  def update_mask(self, old_box, new_box):
    c_mask = self.mask
    cut_mask = cutout_mask(c_mask, old_box[0])
    new_mask = np.zeros(c_mask.shape)
    new_mask = insert_mask(cut_mask, new_box[0], new_mask)
    self.mask = new_mask

  def update_feature_vect(self, new_feature_vect, alpha_val = 0.6):
    exp_dec = alpha_val*self.feature_vect + (1-alpha_val)*new_feature_vect
    self.feature_vect = exp_dec


  def update(self,bbox, mask, feature_vect):
    """
    Updates the state vector with observed bbox.
    """
    self.mask = mask

    self.time_since_update = 0
    self.history = []
    self.hits += 1
    self.hit_streak += 1
    self.kf.update(convert_bbox_to_z(bbox))

    new_box = self.get_state()
    self.update_mask([bbox], new_box)
    self.update_feature_vect(feature_vect)

  def predict(self):
    """
    Advances the state vector and returns the predicted bounding box estimate.
    """
    old_box = self.get_state()

    if((self.kf.x[6]+self.kf.x[2])<=0):
      self.kf.x[6] *= 0.0
    self.kf.predict()
    self.age += 1
    if(self.time_since_update>0):
      self.hit_streak = 0
    self.time_since_update += 1
    self.history.append(convert_x_to_bbox(self.kf.x))

    new_box = self.get_state()
    self.update_mask(old_box, new_box)

    return self.history[-1], self.mask

  def get_state(self):
    """
    Returns the current bounding box estimate.
    """
    return convert_x_to_bbox(self.kf.x)

  def get_full_state(self):
    return convert_x_to_bbox(self.kf.x), self.mask, self.feature_vect


def associate_detections_to_trackers(model, 
                                     detections, 
                                     detect_masks,
                                     trackers, 
                                     pred_masks, 
                                     feature_vect_detect, 
                                     feature_vect, 
                                     n_last_seen, 
                                     device, 
                                     iou_threshold = 0.3, 
                                     dist_mode='default'):
  """
  Assigns detections to tracked object (both represented as bounding boxes)
  (this class is taken from https://github.com/abewley/sort/blob/master/sort.py and extended 
  to work with feature vectors and segmentation masks as well as different modes for 
  distance calculation)

  Returns 3 lists of matches, unmatched_detections and unmatched_trackers
  """
  if(len(trackers)==0):
    return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,5),dtype=int)

  if dist_mode == 'mask':
    iou_matrix = iou_mask(detections, detect_masks, trackers, pred_masks)
  elif dist_mode == 'box':
    iou_matrix = iou_matrix = iou_batch(detections,  trackers)
  else:
    iou_matrix = decision_matr(model, 
                               detections, 
                               detect_masks, 
                               trackers, 
                               pred_masks, 
                               feature_vect_detect, 
                               feature_vect, 
                               n_last_seen, 
                               device)
 
  if min(iou_matrix.shape) > 0:
    a = (iou_matrix > iou_threshold).astype(np.int32)

    if a.sum(1).max() == 1 and a.sum(0).max() == 1:
        matched_indices = np.stack(np.where(a), axis=1)
    else:
      matched_indices = linear_assignment(-iou_matrix)
  else:
    matched_indices = np.empty(shape=(0,2))

  unmatched_detections = []
  for d, det in enumerate(detections):
    if(d not in matched_indices[:,0]):
      unmatched_detections.append(d)
  unmatched_trackers = []
  for t, trk in enumerate(trackers):
    if(t not in matched_indices[:,1]):
      unmatched_trackers.append(t)

  #filter out matched with low IOU
  matches = []
  for m in matched_indices:
    if(iou_matrix[m[0], m[1]]<iou_threshold):
      unmatched_detections.append(m[0])
      unmatched_trackers.append(m[1])
    else:
      matches.append(m.reshape(1,2))
  if(len(matches)==0):
    matches = np.empty((0,2),dtype=int)
  else:
    matches = np.concatenate(matches,axis=0)

  return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


class Tracker(object):
  def __init__(self, model, device=None, max_age=5, min_hits=3, iou_threshold=0.3, dist_mode = 'default'):
    
    self.model = model
    self.model.eval()
    self.max_age = max_age
    self.min_hits = min_hits
    self.iou_threshold = iou_threshold
    self.trackers = []
    self.frame_count = 0
    self.dist_mode = dist_mode
    if device is None:
      self.device = 'cpu'
    else:
      self.device = device

  def update(self, dets=np.empty((0, 5)), masks = False, feature_vect_detect = False):
    self.frame_count += 1
    # get predicted locations from existing trackers.
    trks = np.zeros((len(self.trackers), 5))
    to_del = []
    ret = []

    pred_masks = []
    feature_vects = []
    n_last_seen = []

    for t, trk in enumerate(trks):
      tracker_feature_vect = self.trackers[t].feature_vect
      feature_vects.append(tracker_feature_vect)
      hist, pred_mask = self.trackers[t].predict()
      time_since_update = self.trackers[t].time_since_update
      n_last_seen.append(time_since_update)
      pred_masks.append(pred_mask)
      pos = hist[0]
      #pos = self.trackers[t].predict()[0]
      trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
      if np.any(np.isnan(pos)):
        to_del.append(t)
    trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
    for t in reversed(to_del):
      self.trackers.pop(t)

    matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(self.model, dets,masks,trks,pred_masks, feature_vect_detect, feature_vects,  n_last_seen, self.device, iou_threshold=self.iou_threshold, dist_mode = self.dist_mode)
    # update matched trackers with assigned detections
    for m in matched:
      self.trackers[m[1]].update(dets[m[0], :], masks[m[0]], feature_vect_detect[m[0]])

    # create and initialise new trackers for unmatched detections
    for i in unmatched_dets:
        trk = KalmanBoxTracker(dets[i,:], masks[i], feature_vect_detect[i])
        self.trackers.append(trk)
    i = len(self.trackers)
    for trk in reversed(self.trackers):
        d = trk.get_state()[0]
        if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
          ret.append(np.concatenate((d,[trk.id+1])).reshape(1,-1)) # +1 as MOT benchmark requires positive
        i -= 1
        # remove dead tracklet
        if(trk.time_since_update > self.max_age):
          self.trackers.pop(i)
    if(len(ret)>0):
      return np.concatenate(ret)
    return np.empty((0,5))