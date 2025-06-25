"""
The following code is taken from https://github.com/VisualComputingInstitute/mots_tools.git.

Copyright (c) 2019 Visual Computing Institute

The seqmap file is customized to work with the wildlife crossings dataset. (further information on the dataset can be found in the readme file)
"""


import PIL.Image as Image
import numpy as np
import pycocotools.mask as rletools
# pycocotools has to be installed. can be installed with 'pip install pycocotools'
import glob
import os


class SegmentedObject:
  """
    Class representing a single segmented object in a frame.

    Attributes:
        mask (dict): COCO-style RLE encoded binary mask.
        class_id (int): The object class ID.
        track_id (int): The unique tracking ID for this object.
  """
  def __init__(self, mask, class_id, track_id):
    self.mask = mask
    self.class_id = class_id
    self.track_id = track_id


def load_sequences(path, seqmap):
  """
    Loads all annotated sequences from a directory using a given sequence map.

    Args:
        path (str): Path to the folder containing sequences or annotation files.
        seqmap (list): List of sequence names to load.

    Returns:
        dict: Nested dictionary for a sequence.
  """
  objects_per_frame_per_sequence = {}
  for seq in seqmap:
    #print("Loading sequence", seq)
    seq_path_folder = os.path.join(path, seq)
    seq_path_txt = os.path.join(path, seq + ".txt")
    print(seq_path_txt)
    if os.path.isdir(seq_path_folder):
      objects_per_frame_per_sequence[seq] = load_images_for_folder(seq_path_folder)
    elif os.path.exists(seq_path_txt):
      objects_per_frame_per_sequence[seq] = load_txt(seq_path_txt)
    else:
      assert False, "Can't find data in directory " + path

  return objects_per_frame_per_sequence


def load_txt(path):
  """
    Loads object annotations from a TXT file.

    Args:
        path (str): Path to the .txt annotation file.

    Returns:
        dict: Dictionary mapping frame numbers to lists of segmented objects.
  """
  objects_per_frame = {}
  track_ids_per_frame = {}  # To check that no frame contains two objects with same id
  combined_mask_per_frame = {}  # To check that no frame contains overlapping masks
  with open(path, "r") as f:
    for line in f:
      line = line.strip()
      fields = line.split(" ")
      #print(fields)
      frame = int(fields[0])
      if frame not in objects_per_frame:
        objects_per_frame[frame] = []
      if frame not in track_ids_per_frame:
        track_ids_per_frame[frame] = set()
      if int(fields[1]) in track_ids_per_frame[frame]:
        #print(int(fields[1]), track_ids_per_frame, objects_per_frame)
        assert False, "Multiple objects with track id " + fields[1] + " in frame " + fields[0]
      else:
        track_ids_per_frame[frame].add(int(fields[1]))

      class_id = int(fields[2])
      if not(class_id == 1 or class_id == 2 or class_id == 10):
        assert False, "Unknown object class " + fields[2]

      mask = {'size': [int(fields[3]), int(fields[4])], 'counts': fields[5].encode(encoding='UTF-8')}
      if frame not in combined_mask_per_frame:
        combined_mask_per_frame[frame] = mask
      elif rletools.area(rletools.merge([combined_mask_per_frame[frame], mask], intersect=True)) > 0.0:
        assert False, "Objects with overlapping masks in frame " + fields[0]
      else:

        #import matplotlib.pyplot as plt
        #print(combined_mask_per_frame[frame].shape)
        #plt.imshow(combined_mask_per_frame[frame])
        #print('h')

        combined_mask_per_frame[frame] = rletools.merge([combined_mask_per_frame[frame], mask], intersect=False)

      objects_per_frame[frame].append(SegmentedObject(
        mask,
        class_id,
        int(fields[1])
      ))

  return objects_per_frame


def load_images_for_folder(path):
  """
    Loads segmented object masks from PNG images in a folder.

    Args:
        path (str): Path to the folder containing segmentation images.

    Returns:
        dict: Dictionary mapping frame numbers to lists of segmented objects.
  """
  files = sorted(glob.glob(os.path.join(path, "*.png")))

  objects_per_frame = {}
  for file in files:
    objects = load_image(file)
    frame = filename_to_frame_nr(os.path.basename(file))
    objects_per_frame[frame] = objects

  return objects_per_frame


def filename_to_frame_nr(filename):
  """
    Extracts the frame number from a filename.

    Args:
        filename (str): Filename in the format '000000.png', '000001.png', etc.

    Returns:
        int: Extracted frame number.
  """
  assert len(filename) == 10, "Expect filenames to have format 000000.png, 000001.png, ..."
  return int(filename.split('.')[0])


def load_image(filename, id_divisor=1000):
  """
    Loads a segmentation mask image and returns object instances.

    Args:
        filename (str): Path to the image file.
        id_divisor (int): Divisor to extract class ID from object ID.

    Returns:
        list: List of segmented objects.
  """
  img = np.array(Image.open(filename))
  obj_ids = np.unique(img)

  objects = []
  mask = np.zeros(img.shape, dtype=np.uint8, order="F")  # Fortran order needed for pycocos RLE tools
  for idx, obj_id in enumerate(obj_ids):
    if obj_id == 0:  # background
      continue
    mask.fill(0)
    pixels_of_elem = np.where(img == obj_id)
    mask[pixels_of_elem] = 1
    objects.append(SegmentedObject(
      rletools.encode(mask),
      obj_id // id_divisor,
      obj_id
    ))

  return objects


def load_seqmap(seqmap_filename):
  """
    Loads a sequence map file specifying sequences and their frame counts.

    Args:
        seqmap_filename (str): Path to the seqmap file.

    Returns:
        tuple:
            - list: List of sequence names.
            - dict: Mapping from sequence names to their maximum frame count.
  """
  print("Loading seqmap...")
  seqmap = []
  max_frames = {}
  with open(seqmap_filename, "r") as fh:
    for i, l in enumerate(fh):
      fields = l.split(" ")
      seq = str(fields[0])
      #seq = "%04d" % int(fields[0])
      seqmap.append(seq)
      max_frames[seq] = int(fields[3])
  return seqmap, max_frames


def write_sequences(gt, output_folder):
  """
    Writes all sequences to the output folder.

    Args:
        gt (dict): Dictionary mapping sequence names to frame data.
        output_folder (str): Destination folder path.
  """
  os.makedirs(output_folder, exist_ok=True)
  for seq, seq_frames in gt.items():
    write_sequence(seq_frames, os.path.join(output_folder, seq + ".txt"))
  return


def write_sequence(frames, path):
  """
    Writes a single sequence's annotations to a text file.

    Args:
        frames (dict): Dictionary mapping frame numbers to SegmentedObject lists.
        path (str): Path to save the .txt file.
  """
  with open(path, "w") as f:
    for t, objects in frames.items():
      for obj in objects:
        print(t, obj.track_id, obj.class_id, obj.mask["size"][0], obj.mask["size"][1],
              obj.mask["counts"].decode(encoding='UTF-8'), file=f)
        

import pycocotools.mask as rletools
import sys
#from mots_common.io import load_seqmap, load_sequences
from mots_eval.MOTS_metrics import compute_MOTS_metrics


IGNORE_CLASS = 10


def mask_iou(a, b, criterion="union"):
  """
    Computes IoU (intersection over union) between two segmented objects.

    Args:
        a (SegmentedObject): First object.
        b (SegmentedObject): Second object.
        criterion (str): "union" (default) or other criterion for evaluation.

    Returns:
        float: IoU score between the two masks.
  """
  is_crowd = criterion != "union"
  return rletools.iou([a.mask], [b.mask], [is_crowd])[0][0]


def evaluate_class(gt, results, max_frames, class_id):
  """
    Computes MOTS metrics for a specific class.

    Args:
        gt (dict): Ground truth annotations.
        results (dict): Predicted annotations.
        max_frames (dict): Maximum frame count per sequence.
        class_id (int): Object class ID to evaluate.

    Returns:
        dict: Computed evaluation metrics for the class.
  """
  _, results_obj = compute_MOTS_metrics(gt, results, max_frames, class_id, IGNORE_CLASS, mask_iou)
  return results_obj


def run_eval(results_folder, gt_folder, seqmap_filename):
  """
    Main function to run the MOTS evaluation.

    Args:
        results_folder (str): Folder containing prediction sequences.
        gt_folder (str): Folder containing ground truth sequences.
        seqmap_filename (str): Path to sequence map file.

    Returns:
        None
  """
  seqmap, max_frames = load_seqmap(seqmap_filename)
  print("Loading ground truth...")
  gt = load_sequences(gt_folder, seqmap)
  print("Loading results...")
  results = load_sequences(results_folder, seqmap)
  
  results = evaluate_class(gt, results, max_frames, 1)
  


