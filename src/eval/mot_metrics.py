import motmetrics as mm 
# mot metrics can be installed with 'pip install motmetrics'
import numpy as np


# function to evaluate results with prediction and ground truth given as .txt files in mot16 format
# this function is taken from https://github.com/cheind/py-motmetrics

def motMetricsEnhancedCalculator(gtSource, tSource):
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


# function to evaluate across multiple videos
# results are accumulated across all predictions and ground truth data for each video. then the average is calculated
# the ground truth annotation files are assumed to be in the 'org_anns_path' directory and predicted tracking files in 'prediction_path'
# the ground truth file and the predicted tracking files are assumed to have identical names
# files mit have mot16 format (<frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>)

import os
import numpy

def eval_set(org_anns_path, prediction_path):
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