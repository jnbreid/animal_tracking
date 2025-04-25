import motmetrics as mm 
# mot metrics can be installed with 'pip install motmetrics'
import numpy as np

def motMetricsEnhancedCalculator(gtSource, tSource):
  """
    Evaluates tracking performance using MOTChallenge metrics.
    This function is taken from https://github.com/cheind/py-motmetrics.

    This function compares tracking predictions against ground truth annotations 
    using the MOT16 format. It computes various MOT metrics using the 
    `motmetrics` library, such as IDF1, MOTA, MOTP, precision, and recall.

    Args:
        gtSource (str): File path to the ground truth `.txt` file in MOT16 format.
        tSource (str): File path to the predicted tracking `.txt` file in MOT16 format.

    Returns:
        np.ndarray: A 1D array containing metric values in the following order:
            ['num_frames', 'idf1', 'idp', 'idr', 'recall', 'precision', 
             'num_objects', 'mostly_tracked', 'partially_tracked', 'mostly_lost', 
             'num_false_positives', 'num_misses', 'num_switches', 
             'num_fragmentations', 'mota', 'motp']
    """

  # import required packages
  import motmetrics as mm
  import numpy as np

  # load ground truth
  gt = np.loadtxt(gtSource, delimiter=',').reshape((-1,10))
  if gt.shape[0] == 0:
    gt = np.empty((1,10))

  # load tracking output
  t = np.loadtxt(tSource, delimiter=',').reshape((-1,10))
  if t.shape[0] == 0:
    t = np.empty((1,10))

  acc = mm.MOTAccumulator(auto_id=True)

  for frame in range(int(gt[:,0].max())):
    frame += 1 # detection and frame numbers begin at 1

    gt_dets = gt[gt[:,0]==frame,1:6] # select all detections in gt
    t_dets = t[t[:,0]==frame,1:6] # select all detections in t

    C = mm.distances.iou_matrix(gt_dets[:,1:], t_dets[:,1:], \
                                max_iou=0.5) # format: gt, t

    acc.update(gt_dets[:,0].astype('int').tolist(), \
              t_dets[:,0].astype('int').tolist(), C)

  mh = mm.metrics.create()

  summary = mh.compute(acc, metrics=['num_frames', 'idf1', 'idp', 'idr', \
                                     'recall', 'precision', 'num_objects', \
                                     'mostly_tracked', 'partially_tracked', \
                                     'mostly_lost', 'num_false_positives', \
                                     'num_misses', 'num_switches', \
                                     'num_fragmentations', 'mota', 'motp' \
                                    ], \
                      name='acc')

  strsummary = mm.io.render_summary(
      summary,
      #formatters={'mota' : '{:.2%}'.format},
      namemap={'idf1': 'IDF1', 'idp': 'IDP', 'idr': 'IDR', 'recall': 'Rcll', \
               'precision': 'Prcn', 'num_objects': 'GT', \
               'mostly_tracked' : 'MT', 'partially_tracked': 'PT', \
               'mostly_lost' : 'ML', 'num_false_positives': 'FP', \
               'num_misses': 'FN', 'num_switches' : 'IDsw', \
               'num_fragmentations' : 'FM', 'mota': 'MOTA', 'motp' : 'MOTP',  \
              }
  )
  print(strsummary)
  return summary.to_numpy()


import os
import numpy

def eval_set(org_anns_path, prediction_path):
    """
    Evaluates tracking performance over a dataset of multiple sequences.

    For each video or sequence in the dataset, this function computes MOT metrics 
    using the ground truth and predicted files (assumed to be in MOT16 format).
    The final output is the average of all metric values across the dataset.

    Args:
        org_anns_path (str): Path to the directory containing ground truth `.txt` files.
        prediction_path (str): Path to the directory containing prediction `.txt` files.
                               File names must match those in the ground truth directory.

    Returns:
        None. Prints the average of each MOT metric across all sequences.
    """

    pred_files = os.listdir(prediction_path)

    metrics_results = []

    for item in pred_files:
        pred_files_path = os.path.join(prediction_path, item)
        ann_file_path = os.path.join(org_anns_path, item)

        summary = motMetricsEnhancedCalculator(ann_file_path, pred_files_path)

        metrics_results.append(summary)

    mean = np.asarray(metrics_results).reshape(-1,16)#.nanmean(axis = 0)
    mean = np.nanmean(mean, axis = 0)

    names = ['num_frames', 'idf1', 'idp', 'idr', \
                                        'recall', 'precision', 'num_objects', \
                                        'mostly_tracked', 'partially_tracked', \
                                        'mostly_lost', 'num_false_positives', \
                                        'num_misses', 'num_switches', \
                                        'num_fragmentations', 'mota', 'motp' \
                                        ]

    for i in range(16):
        print(names[i], f"{mean[i]:.3f}")