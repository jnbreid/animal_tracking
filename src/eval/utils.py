import sys
sys.path.insert(0, '..')

from src.tracker import infer_video
from src.utils_loader import get_detector, get_extractor, get_segmentor, get_DistNet

import cv2

def gen_prediction_files(eval_dataset, 
                         save_dir = 'out_dir', 
                         distnet_weights = 'weights/distnet_t.pth',
                         device = None,
                         dist_mode = 'default'):
    
    """
    Runs inference on a dataset of videos and saves detection and segmentation predictions.

    This function processes each video in the dataset by:
      - Loading frames
      - Running object detection, segmentation, and feature extraction
      - Computing distance-based predictions with DistNet
      - Saving results in specified formats (bounding boxes, segmentation masks)

    Args:
        eval_dataset (torch dataset): dataset where each item contains a list of image paths
                             representing frames of a single video. Each item is expected to be 
                             a tuple/list of (frame_paths, label, metadata).
        save_dir (str, optional): Directory where prediction files are saved. Defaults to 'out_dir'.
        distnet_weights (str, optional): Path to pretrained DistNet weights. Defaults to 'weights/distnet_t.pth'.
        device (torch.device or str, optional): Device to run all models on. Defaults to 'cpu' if not specified.
        dist_mode (str, optional): Distance computation mode used by DistNet. Defaults to 'default'.

    Returns:
        None: Outputs are written to disk.
    """
    
    if device is None:
        device = 'cpu'


    detector = get_detector(device = device)
    extractor = get_extractor(device = device)
    segmentor = get_segmentor(device = device)
    distnet = get_DistNet(device = device, weight_path = distnet_weights)


    for i in range(len(eval_dataset)):
        print("reeee")
        c_img_paths, _, _ = eval_dataset[i]
    
        # load all frames
        frames = []
        for item in c_img_paths:
            c_img = cv2.imread(item)
            frames.append(cv2.cvtColor(frames, cv2.COLOR_BGR2RGB))
    
        eval_file_name = f"{str(i+1).zfill(6)}.txt"
        
        infer_video(frames,
                visualize = False,
                box_vis = False,
                box_file = True,
                box_file_name = eval_file_name,
                seg_file = True,
                seg_file_name = eval_file_name,
                save_dir = save_dir,
                distnet = distnet,
                distnet_weights = distnet_weights,
                detector = detector,
                extractor = extractor,
                segmentor = segmentor,
                device = device,
                refine = True,
                dist_mode = dist_mode,
                suppress_warnings = True,)