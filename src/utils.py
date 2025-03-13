from sklearn.metrics import jaccard_score
import numpy as np

# batch iou calculation function is taken from https://github.com/abewley/sort/blob/master/sort.py

def iou_batch(bb_test, bb_gt):#bb_test, bb_gt):
  """
  From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]
  """

  bb_gt = np.expand_dims(bb_gt, 0)
  bb_test = np.expand_dims(bb_test, 1)

  xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
  yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
  xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
  yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
  w = np.maximum(0., xx2 - xx1)
  h = np.maximum(0., yy2 - yy1)
  wh = w * h
  o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
    + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)

  return(o)



# function to calculate the mask iou score between detected and tracked objects.
# box iou is calculated first to reduce calculation time for mask iou score 
# if boxes do not overlap, then mask also do not overlap

def iou_mask(bb_test, mask_test, bb_gt, mask_gt):
  s_test = len(mask_test)
  s_gt = len(mask_gt)

  out_array = iou_batch(bb_test, bb_gt)


  for i in range(s_test):
    for j in range(s_gt):
      if out_array[i,j] > 0:
        t_mask = mask_test[i]
        gt_mask = mask_gt[j]
        score = jaccard_score(gt_mask.astype(np.int8), t_mask.astype(np.int8), average = 'micro')
        out_array[i,j] = score

  return out_array