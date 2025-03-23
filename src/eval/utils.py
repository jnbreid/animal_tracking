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