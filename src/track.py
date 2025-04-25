import numpy as np
import time
from filterpy.kalman import KalmanFilter

from src.utils import linear_assignment, iou_batch, iou_mask, decision_matr, convert_bbox_to_z, convert_x_to_bbox, insert_mask, cutout_mask

class KalmanBoxTracker(object):
  """
  This class represents the internal state of individual tracked objects observed as bounding boxes.
  It extends the original tracker from https://github.com/abewley/sort/blob/master/sort.py to support 
  feature vectors and segmentation masks, in addition to the bounding box information.
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
    """
        Updates the segmentation mask based on the change in bounding box.

        Args:
            old_box (list or np.array): The previous bounding box.
            new_box (list or np.array): The new bounding box.

        Returns:
            None: The mask is updated in place.
    """
    c_mask = self.mask
    cut_mask = cutout_mask(c_mask, old_box[0])
    new_mask = np.zeros(c_mask.shape)
    new_mask = insert_mask(cut_mask, new_box[0], new_mask)
    self.mask = new_mask

  def update_feature_vect(self, new_feature_vect, alpha_val = 0.6):
    """
        Updates the feature vector using an exponential moving average.

        Args:
            new_feature_vect (np.array): The new feature vector to be incorporated.
            alpha_val (float, optional): The smoothing factor (default is 0.6). Controls how much of the old feature vector is retained.

        Returns:
            None: The feature vector is updated in place.
    """
    exp_dec = alpha_val*self.feature_vect + (1-alpha_val)*new_feature_vect
    self.feature_vect = exp_dec


  def update(self,bbox, mask, feature_vect):
    """
        Updates the tracker state with a new bounding box, mask, and feature vector.

        Args:
            bbox (list or np.array): The new bounding box.
            mask (np.array): The new segmentation mask for the detected and assigned object.
            feature_vect (np.array): The new feature vector for the detected and assigned object.

        Returns:
            None: The state, mask, and feature vector are updated in place.
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
        Advances the state vector and returns the predicted bounding box and mask estimate.

        Args:
            None: No input arguments.

        Returns:
            tuple: 
                - The predicted bounding box [x, y, width, height] as a list.
                - The updated segmentation mask.
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
        Returns the current bounding box estimate from the Kalman filter state.

        Args:
            None: This method does not require any arguments.

        Returns:
            np.array: The current bounding box estimate in the form [x, y, width, height].
    """
    return convert_x_to_bbox(self.kf.x)

  def get_full_state(self):
    """
        Returns the full state of the tracked object, including the bounding box, segmentation mask, and feature vector.

        Args:
            None: This method does not require any arguments.

        Returns:
            tuple: A tuple containing:
                - np.array: The bounding box estimate [x, y, width, height].
                - np.array: The segmentation mask of the object.
                - np.array: The feature vector of the object.
    """
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
    Assigns detections to tracked objects (both represented as bounding boxes). 
    This method is an extended version of the original from https://github.com/abewley/sort/blob/master/sort.py, 
    supporting feature vectors, segmentation masks, and different distance calculation modes.

    Args:
        model (DistNet): The model used for distance predictions.
        detections (np.ndarray): A numpy array of shape (n, 4) representing the bounding boxes of the detections 
                                  (each row is [x, y, width, height]).
        detect_masks (list of np.ndarray): A list of n numpy array of shape ( N, M) representing the masks associated with the detections.
        trackers (list of KalmanBoxTracker): A list of m KalmanBoxTracker objects that represent the current trackers.
        pred_masks (list of np.ndarray): A list of m numpy array of shape (N, M) representing the predicted masks for the trackers.
        feature_vect_detect (list of np.ndarray): A list of n 1x1535 numpy arrays representing feature vectors of the detections.
        feature_vect (list of np.ndarray): A list of m 1x1535 numpy arrays representing feature vectors of the trackers.
        n_last_seen (list of floats): A list of m floats representing the last seen time for each tracker.
        device (torch.device): The device (CPU or GPU) used for computation.
        iou_threshold (float, optional): The threshold for intersection over union (IoU) used to match detections and trackers. Default is 0.3.
        dist_mode (str, optional): The method used to calculate the distance. It can be one of:
            - 'mask': Use IoU of segmentation masks for distance calculation.
            - 'box': Use IoU of bounding boxes for distance calculation.
            - 'default': Use Use IoU of segmentation masks and feature vector-based distance calculation.

    Returns:
        tuple: 
            - np.ndarray: A numpy array of matched elements of shape (k, 2), where each row is [detection_index, tracker_index] and k is the number of matches.
            - np.ndarray: A numpy array of unmatched detection indices.
            - np.ndarray: A numpy array of unmatched tracker indices.
  """

  if(len(trackers)==0):
    return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,5),dtype=int)

  if dist_mode == 'mask':
    iou_matrix = iou_mask(detections, detect_masks, trackers, pred_masks)
  elif dist_mode == 'box':
    iou_matrix = iou_matrix = iou_batch(detections,  trackers)
  else:
    iou_matrix = decision_matr(model, detections, detect_masks, trackers, pred_masks, 
                               feature_vect_detect, feature_vect, n_last_seen, device)


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
  """
    Tracker for managing multiple tracked objects across video frames.
    This class extends the original tracker from https://github.com/abewley/sort/blob/master/sort.py to support
    feature vectors, segmentation masks, and different modes for distance calculation.

    Attributes:
        model (DistNet): The model used for distance predictions.
        max_age (int): The maximum number of frames a tracker can remain unmatched before being deleted.
        min_hits (int): The minimum number of hits (frames with a match) required to consider a tracker valid.
        iou_threshold (float): The threshold for the predictted distance to match detections with trackers.
        trackers (list): A list of KalmanBoxTracker objects that represent the active trackers.
        frame_count (int): The current frame number.
        dist_mode (str): The method used for distance calculation. It can be 'mask', 'box', or 'default'.
        device (str): The device (CPU or GPU) used for computation.
    """

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
    """
        Updates the tracker with new detections and feature vectors. Performs the following steps:
        1. Predicts the current positions of the tracked objects.
        2. Associates detections with trackers based on the distance calculation.
        3. Updates the trackers with matched detections.
        4. Creates new trackers for unmatched detections.
        5. Removes dead trackers that have not been updated for too long.

        Args:
            dets (np.ndarray, optional): A numpy array of shape (n, 5) representing the bounding boxes and IDs of the detections. 
                                         Each row is [x, y, width, height, ID]. Defaults to an empty array. If there are no detections use an empty array as input
            masks (list of np.ndarray, optional): A list of n numpy array of shape (N, M) representing the segmentation masks for the detections. Defaults to False.
            feature_vect_detect (list of np.ndarray, optional): A list of n 1x1535 numpy arrays representing the feature vectors of the detections. Defaults to False.

        Returns:
            np.ndarray: A numpy array of shape (m, 5) where each row is [x, y, width, height, ID] for the tracked objects.
                        If no detections are matched or tracked, an empty array is returned.
    """
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