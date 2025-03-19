from sklearn.metrics import jaccard_score
import numpy as np
import matplotlib.colors
import matplotlib.pyplot as plt
import torch
import cv2

from sklearn.metrics.pairwise import cosine_similarity
from skimage.transform import resize
from scipy.optimize import linear_sum_assignment


"""
function to assign prediction and detection according to precomputed cost matrix

Parameters:
- precomputed cost matrix nxm

Returns:
- array describing assigned pairs
"""
def linear_assignment(cost_matrix):
    x, y = linear_sum_assignment(cost_matrix)
    return np.array(list(zip(x, y)))



"""
This function computes the IOU between two bboxes in the form [x1,y1,x2,y2]
(this function is taken from https://github.com/abewley/sort/blob/master/sort.py)

Parameters:
- bb_test (nx4 numpy array)
- bb_gt (mx4 numpy array)

Returns:
- pairwise IoU scores between boxes (nxm numpy array)
"""
def iou_batch(bb_test, bb_gt):#bb_test, bb_gt):
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
  

"""
function to calculate the mask iou score between detected and tracked objects.
box iou is calculated first to reduce calculation time for mask iou score 
if boxes do not overlap, then mask also do not overlap

Parameters:
- bb_test (nx4 numpy array)
- bb_gt (mx4 numpy array)
- mask_test (NxM numpy array)
- mask_gt (NxM numpy array)

Returns:
- pairwise mask IoU scores (nxm numpy array)
"""
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



"""
function to calculate the decision matrix that incoporates information in the form of
feature distance, mask_iou score and a time component

Parameters:
- model (DistNet model object [torch.nn])
- bb_test (nx4 numpy array)
- mask_test (NxM numpy array)
- bb_gt (nx4 numpy array)
- mask_gt (NxM numpy array)
- feature_vect_test (list containing n 1x1536 numpy arrays)
- feature_vect (list containing m 1x1536 numpy arrays)
- n_last_seen (list containing n float values)
- device (torch device)

Returns:
- decision matrix (nxm numpy array)
"""
def decision_matr(model, 
                  bb_test, 
                  mask_test, 
                  bb_gt, 
                  mask_gt, 
                  feature_vect_test, 
                  feature_vect, 
                  n_last_seen, 
                  device, 
                  lambda_val = 0.6):
  iou_array = iou_mask(bb_test, mask_test, bb_gt, mask_gt)
  s_test = len(mask_test)
  s_gt = len(mask_gt)
  combined_dist = np.zeros((s_test, s_gt))
  for i in range(s_test):
    for j in range(s_gt):
      n_seen = n_last_seen[j]
      t_feature = np.asarray(feature_vect_test[i]).reshape((1,-1))
      gt_feature_vect = np.asarray(feature_vect[j]).reshape((1,-1))
      similarity = cosine_similarity(t_feature, gt_feature_vect)[0][0]
      # prepare the input vector
      input_vect = np.asarray([similarity, iou_array[i,j], n_seen], dtype = np.float32)
      input_vect = torch.from_numpy(input_vect)
      input_vect = input_vect.to(device)
      # calculate confidence value for one specific pair
      output = model(input_vect)
      output = output.detach().cpu().numpy()[0,0]
      combined_dist[i,j] = output
  # rescale and transform decision value
  dec_matrix = -(combined_dist-1)
  return dec_matrix




"""
Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
[x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
the aspect ratio
(this function is taken from https://github.com/abewley/sort/blob/master/sort.py)

Parameters:
- bbox (1x4 numpy array)

Returns:
- bbox 1x4 numpy array
"""
def convert_bbox_to_z(bbox):
  w = bbox[2] - bbox[0]
  h = bbox[3] - bbox[1]
  x = bbox[0] + w/2.
  y = bbox[1] + h/2.
  s = w * h    #scale is just area
  r = w / float(h)
  return np.array([x, y, s, r]).reshape((4, 1))


"""
Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
[x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
(this function is taken from https://github.com/abewley/sort/blob/master/sort.py)

Parameters:
- x (1x4 numpy array)
- score (bool)

Returns:
- 1x4 or 1x5 numpy array
"""
def convert_x_to_bbox(x,score=None):
  w = np.sqrt(x[2] * x[3])
  h = x[2] / w
  if(score==None):
    return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.]).reshape((1,4))
  else:
    return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.,score]).reshape((1,5))
  

"""
function to corp a binary mask to a given bounding box

Parameters:
- mask (NxM numpy array)
- bbox (1x4 numpy array)

Returns:
- nxm numpy array
"""
def cutout_mask(mask, bbox):
    x0 = int(bbox[0])
    y0 = int(bbox[1])
    x1 = int(bbox[2])
    y1 = int(bbox[3])

    x_len = np.abs(x1-x0)
    y_len = np.abs(y1-y0)
    ct_mask = np.zeros((y_len,x_len))

    # if box is outside of image return empty mask
    if x0 >= mask.shape[1] or y0 >= mask.shape[0] or x1 <= 0 or y1 <= 0 or x1<x0 or y1<y0:
        return ct_mask

    x0m = y0m = x1m = y1m = 0

    if x0 < 0:
        x0m = -1*x0
        x0 = 0
    if y0 < 0:
        y0m = -1*y0
        y0 = 0
    if x1 >= mask.shape[1]:
        x1m = x1 - mask.shape[1]
        x1 = mask.shape[1]
    if y1 >= mask.shape[0]:
        y1m = y1 - mask.shape[0]
        y1 = mask.shape[0]

    visible_mask = mask[y0:y1, x0:x1]
    ct_mask[y0m:y_len-y1m, x0m:x_len-x1m] = visible_mask

    return ct_mask


"""
this function inserts a binary mask into a numpy array according to coordinates given by a bounding box

Parameters:
- cut_mask (nxm numpy array)
- bbox (1x4 numpy array)
- insert_array (NxM numpy array)

Returns:
- NxM numpy array
"""
def insert_mask(cut_mask, bbox, insert_array):
    x0 = int(bbox[0])
    y0 = int(bbox[1])
    x1 = int(bbox[2])
    y1 = int(bbox[3])

    if x0 >= insert_array.shape[1] or y0 >= insert_array.shape[0] or x1 <= 0 or y1 <= 0 or x1<x0 or y1<y0:
        return insert_array

    height = x1-x0
    width = y1-y0

    resize_mask = resize(cut_mask,(width, height), anti_aliasing=False, order=0)

    x0_marg = 0
    x1_marg = 0
    y0_marg = 0
    y1_marg = 0

    if x0 < 0:
        x0_marg = np.absolute(x0-0)
        x0 = 0
    if x1 >= insert_array.shape[1]:
        x1_marg = np.absolute(x1 - insert_array.shape[1])
        x1 = insert_array.shape[1]
    if y0 < 0:
        y0_marg = np.absolute(y0-0)
        y0 = 0
    if y1 >= insert_array.shape[0]:
        y1_marg = np.absolute(y1 - insert_array.shape[0])
        x1 = insert_array.shape[0]

    cut_resize_mask = resize_mask[y0_marg:width-y1_marg, x0_marg:height-x1_marg]

    insert_array[y0:y0+cut_resize_mask.shape[0], x0:x0+cut_resize_mask.shape[1]] = cut_resize_mask# resize_mask

    return insert_array



"""
function to overlay a mask onto a picture with a given alpha valua

Parameters:
- image (NxM numpy array)
- mask (NxM numpy array)
- color (1x3 numpy array) [RGB format]
- alpha (float in [0,1])
- resize (bool)

Returns:
- NxM numpy array
"""
def overlay(image, mask, color, alpha, resize=None):
    color = color[::-1]
    colored_mask = np.expand_dims(mask, 0).repeat(3, axis=0)
    colored_mask = np.moveaxis(colored_mask, 0, -1)
    masked = np.ma.MaskedArray(image, mask=colored_mask, fill_value=color)
    image_overlay = masked.filled()

    if resize is not None:
        image = cv2.resize(image.transpose(1, 2, 0), resize)
        image_overlay = cv2.resize(image_overlay.transpose(1, 2, 0), resize)

    image_combined = cv2.addWeighted(image, 1 - alpha, image_overlay, alpha, 0)

    return image_combined


def merge_masks_fast(org_mask, merge_mask, n_id):
  ind = np.where(org_mask==0)

  org_mask[ind] = merge_mask[ind]*n_id

  return org_mask




"""
function to visualize the distribution of the training dataset for DistNet, to ensure
that the dataset does not only contain positive/negative instances, and that the data distribution
is 'acceptable' (no obvious errors in the data)

Parameters:
- c_dataset (instance of training dataset for DistNet)

Returns:
-
"""
def visualize_dataset_distribution(c_dataset):    
    pos_elems = []
    neg_elems = []

    for i in range(len(c_dataset)):
        item, label = c_dataset[i]

        if label[0].item() == 1:
            pos_elems.append(item)
        else:
            neg_elems.append(item)


    pos_feature_dist = [item[0] for item in pos_elems]
    pos_IoU = [item[1] for item in pos_elems]
    pos_last = [item[2] for item in pos_elems]

    neg_feature_dist = [item[0] for item in neg_elems]
    neg_IoU = [item[1] for item in neg_elems]
    neg_last = [item[2] for item in neg_elems]

    bins =  np.arange(0, 1, 0.001)
    bins1 = np.arange(0, 1, 0.01)

    fig, axs = plt.subplots(1,3)
    fig.suptitle("GMOT8 - bird")

    axs[0].set_title('feature distance')
    axs[0].hist(pos_feature_dist, bins = bins, density = True, alpha = 0.75)

    axs[1].set_title('IoU')
    axs[1].hist(pos_IoU, bins = bins1, density = True, alpha = 0.75)

    axs[2].set_title('last matched')
    axs[2].hist(pos_last, density = True, alpha = 0.75)

    axs[0].hist(neg_feature_dist, bins = bins, density = True, alpha = 0.75)

    axs[1].hist(neg_IoU, bins = bins1, density = True, alpha = 0.75)

    axs[2].hist(neg_last, density = True, alpha = 0.75)

    plt.show()



"""
class containing code for 63 different colors for visualization
can return colors in RGB format
"""
class col_list():
  def __init__(self):
    hex_colors = ['#0000FF', '#FF0000', '#01FFFE', '#FFA6FE', '#FFDB66','#00FF00', #'#000000',
                  '#006401', '#010067', '#95003A', '#007DB5', '#FF00F6', '#FFEEE8', '#774D00', '#90FB92', '#0076FF', '#D5FF00', '#FF937E', '#6A826C', '#FF029D', '#FE8900', '#7A4782', '#7E2DD2', '#85A900', '#FF0056', '#A42400',
                  '#00AE7E','#683D3B', '#BDC6FF', '#263400', '#BDD393', '#00B917', '#9E008E', '#001544', '#C28C9F', '#FF74A3', '#01D0FF', '#004754', '#E56FFE', '#788231', '#0E4CA1', '#91D0CB', '#BE9970', '#968AE8', '#BB8800',
                  '#43002C', '#DEFF74', '#00FFC6', '#FFE502', '#620E00', '#008F9C', '#98FF52', '#7544B1', '#B500FF', '#00FF78', '#FF6E41', '#005F39', '#6B6882', '#5FAD4E', '#A75740', '#A5FFD2', '#FFB167', '#009BFF', '#E85EBE']

    self.colors = []
    self.counter = 0

    for item in hex_colors:
      col = tuple(np.asarray((np.asarray(matplotlib.colors.to_rgb(item))*255), dtype = np.int32))
      self.colors.append(col)
    return

  def get_color(self, id):
     color = self.colors[int(id%62)]
     return color
