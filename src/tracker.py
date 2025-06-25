# SPDX-License-Identifier: GPL-3.0-only
# Copyright (c) 2025 Jon Breid


import numpy as np
import cv2
from pycocotools import mask
import time
import os
from tqdm import tqdm

from src.utils_loader import get_detector, get_segmentor, get_extractor, get_DistNet
from src.utils import overlay, col_list, merge_masks_fast
from src.track import Tracker
from src.utils_data import write_mp4


"""
function to start inference on a video. annotation files, and a annotated video can be generated

Parameters:
- img_seq (list of x NxMxC numpy arrays)
- confidences_p (list of x numpy arrays of size y)
- bbox_p (list of x bounding box arrays)
- visualize (bool) [generate a video]
- box_vis (bool) [vidualize bounding boxes in video]
- box_file (bool) [generate mot file]
- box_file_name (str)
- seg_file (bool) [generate mots file]
- seg_file_name (str)
- save_dir (str)
- distnet (DistNet object)
- distnet_weights (str)
- detector (object that can be called as detector) (see utils_loader.py for further information)
- extractor (object that can be called as extractor) (see utils_loader.py for further information)
- segmentor (SAM object)
- device (torch device)
- refine (bool)
- dist_mode (str)
- fps (int)

Returns:
-
"""
def infer_video(img_seq, 
                confidences_p = None,
                bbox_p = None,
                visualize = True,
                box_vis = True, 
                box_file = True, 
                box_file_name = "mot16.txt",
                seg_file = True, 
                seg_file_name = "mots.txt",
                save_dir = 'out_dir', 
                distnet = None,
                distnet_weights = None,
                detector = None, 
                extractor = None, 
                segmentor = None, 
                device = None, 
                refine = True, 
                dist_mode = 'default',
                fps = 12,
                suppress_warnings = False,):
    
    """
    Function to perform inference on a video sequence. It generates annotation files (MOT format for tracking and segmentation) and an annotated video.
    
    This function takes a sequence of images, processes each frame to detect and track objects, and optionally visualizes the results, 
    saving the annotated frames and videos, as well as tracking and segmentation annotations.

    Args:
        img_seq (list of ndarray): List of x (N,M,C) numpy arrays representing the image frames of the video sequence.
        confidences_p (list of ndarray, optional): List of x pre computed numpy arrays of size y, where each array represents the confidence values for detected objects.
        bbox_p (list of ndarray, optional): List of x pre computed bounding box arrays, where each array contains bounding boxes for objects in the corresponding frame.
        visualize (bool, optional): If True, generates and saves an annotated video. Default is True.
        box_vis (bool, optional): If True, visualizes bounding boxes in the annotated video. Default is True.
        box_file (bool, optional): If True, generates and saves a MOT-style annotation file with bounding boxes. Default is True.
        box_file_name (str, optional): Name of the bounding box annotation file. Default is "mot16.txt".
        seg_file (bool, optional): If True, generates and saves a MOT-style annotation file with segmentation masks. Default is True.
        seg_file_name (str, optional): Name of the segmentation annotation file. Default is "mots.txt".
        save_dir (str, optional): Directory to save the output (images, annotations, and videos). Default is 'out_dir'.
        distnet (DistNet, optional): Pretrained DistNet model for feature vector extraction. Default is None (Default Model is loaded).
        distnet_weights (str, optional): Path to the DistNet model weights file. Default is None (Default pre trained Model weights are loaded).
        detector (callable, optional): Object detector function that takes an image as input and returns bounding boxes and confidence scores. Default is None.
        extractor (callable, optional): Feature extractor function that extracts feature vectors for detected objects. Default is None.
        segmentor (SAM, optional): Segmentation model used to generate segmentation masks for the objects in the image. Default is None.
        device (str, optional): Device to run the inference on ('cpu' or 'cuda'). Default is None, which uses 'cpu'.
        refine (bool, optional): If True, refines bounding boxes based on segmentation mask. Default is True.
        dist_mode (str, optional): The mode for distance calculation ('default', 'mask', 'box'). Default is 'default'.
        fps (int, optional): Frames per second for the generated video. Default is 12.
        suppress_warnings (bool, optional): If True, suppresses warnings about existing directories. Default is False.

    Returns:
        None: This function saves the results (video, annotation files) in the specified directory.
    """
    
    start_time = time.time()

    print(f"Video processing started.")
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    else:
        if suppress_warnings == False:
            print(f"Warning: {save_dir} already exists.")
    if visualize == True:
        image_dir = os.path.join(save_dir, 'images')
        if not os.path.exists(image_dir):
            os.mkdir(image_dir)
        else: 
            if suppress_warnings == False:
                print(f"Warning: {save_dir} already exists.")
    
    if box_file:
        box_path = os.path.join(save_dir, 'mot16')
        if not os.path.exists(box_path):
            os.mkdir(box_path)
    if seg_file:
        seg_path = os.path.join(save_dir, 'mots')
        if not os.path.exists(seg_path):
            os.mkdir(seg_path)

    if device is None:
        device = 'cpu'

    if detector is None:
        print(f"Loading detection model ...")
        detector = get_detector(device = device)
    if extractor is None:
        print(f"Loading feature extraction model ...")
        extractor = get_extractor(device = device)
    if segmentor is None:
        print(f"Loading segmentation model ...")
        segmentor = get_segmentor(device = device)
    if distnet is None:
        print(f"Loading DistNet ...")
        distnet = get_DistNet(device = device, weight_path = distnet_weights)


    mot_tracker = Tracker(distnet, device = device, max_age = 15, min_hits = 3, iou_threshold = 0.1, dist_mode = dist_mode)
    colors = col_list()

    full_tracks = []
    mots_string = ""

    for i in tqdm(range(len(img_seq))):
 
        
        current_image = img_seq[i]
        full_mask = np.zeros((current_image.shape[:2]))

        # detection
        if bbox_p is not None:
            current_boxes = bbox_p[i].copy()
            current_confidences = confidences_p[i]
        else:
            current_boxes, current_confidences = detector(current_image.copy())
        
        # segmentation
        segmentor.set_image(current_image.copy())
        
        c_masks = []
        c_feature_vects = []
        full_detect = []

        for j in range(current_boxes.shape[0]):
            c_box = current_boxes[j,:].copy()
            c_box[2] = c_box[0] + c_box[2]
            c_box[3] = c_box[1] + c_box[3]

            # calculate mask for each detected object for mask IoU
            pred_mask, _, _ = segmentor.predict(box=c_box, multimask_output=False)
            pred_mask = pred_mask[0,:,:].astype(np.int8)
            c_masks.append(pred_mask)

            # calculate feature vector for each detected object
            current_crop = current_image[c_box[1]:c_box[3], c_box[0]:c_box[2],:].copy()
            feature_vect = extractor(current_crop)
            feature_vect = feature_vect.detach().cpu().numpy().reshape(-1)
            c_feature_vects.append(feature_vect)

            # save box and confidence for tracker
            full_detect.append(list(c_box)+[current_confidences[j]])

        # exception handling if no animal is detected on a frame
        if current_boxes.shape[0] == 0:
            full_detect = np.empty((0, 5))
            c_masks = np.asarray([])
            c_feature_vects = np.asarray([])
        
        tracks = mot_tracker.update(np.array(full_detect), c_masks, c_feature_vects)

        masks = []
        save_detections = []

        for j in range(tracks.shape[0]):
            item = tracks[j,:]
            x0 = int(item[0])
            y0 = int(item[1])
            x1 = int(item[2])
            y1 = int(item[3])
            track_label = str(int(item[4]))

            # calculate msak for predicted area
            box = np.asarray([x0,y0,x1,y1])
            pred_mask, _, _ = segmentor.predict(box=box, multimask_output=False)
            pred_mask = pred_mask[0,:,:].astype(np.int8)
            # refine box
            if refine == True and len(np.unique(pred_mask))>1:
                refined_indexes = np.where(pred_mask == 1)

                ref_x0 = int(refined_indexes[0].min())
                ref_x1 = int(refined_indexes[0].max())

                ref_y0 = int(refined_indexes[1].min())
                ref_y1 = int(refined_indexes[1].max())

                box = np.asarray([ref_y0, ref_x0, ref_y1,  ref_x1])

            masks.append(pred_mask)
            save_detections.append([item[4], box[0], box[1], box[2]-box[0], box[3]-box[1]])
            c_score = current_confidences[j]
            full_tracks.append([i, item[4], box[0], box[1], box[2]-box[0], box[3]-box[1], c_score, -1, -1, -1])

            # calculate mask image for mota file
            full_mask = merge_masks_fast(full_mask, pred_mask, item[4])
        
        if visualize == True:
            bgr_img = cv2.cvtColor(current_image.copy(), cv2.COLOR_RGB2BGR)

        
        c_id_list = list(np.unique(full_mask))
        c_id_list.remove(0)

        for k in range(len(c_id_list)):
            wanted_id = c_id_list[k]
            c_mask = np.zeros(full_mask.shape, dtype = np.uint8)
            c_mask[full_mask==wanted_id] = 1

            vis_mask = c_mask.copy()

            c_mask = np.asfortranarray(c_mask)

            rle = mask.encode(c_mask)
            size = rle['size']
            counts = rle['counts'].decode('UTF-8')
            mots_string = mots_string + f"{str(i)} {str(1000 + int(wanted_id))} {str(1)} {str(size[0])} {str(size[1])} {counts}\n"

            if visualize == True:
                col = colors.get_color(int(wanted_id))
                bgr_img = overlay(bgr_img, vis_mask, (int(col[2]),int(col[1]),int(col[0])), alpha = 0.5)
                
                if box_vis == True:
                    box_indexes = np.where(vis_mask == 1)
                    ref_x0 = int(box_indexes[0].min())
                    ref_x1 = int(box_indexes[0].max())
                    ref_y0 = int(box_indexes[1].min())
                    ref_y1 = int(box_indexes[1].max())
                    box = np.asarray([ref_y0, ref_x0, ref_y1,  ref_x1])
                    cv2.rectangle(bgr_img, (box[0], box[1]), (box[2], box[3]), (int(col[0]),int(col[1]),int(col[2])), 4)
                    cv2.putText(bgr_img, '#'+str(int(wanted_id)), (box[0]+5, box[1]-10), 0,0.6,(int(col[0]),int(col[1]),int(col[2])),thickness=4)

        # save generated annotated frames    
        if visualize == True:
            img_name = f"{str(i).zfill(6)}.jpg"
            img_path = os.path.join(image_dir, img_name)
            cv2.imwrite(img_path, bgr_img)

    #save mot16 style annotation file
    if box_file == True:
        print(f"Saving segmentation masks in mots txt file.")
        pred_array = np.asarray(full_tracks)
        pred_path = os.path.join(box_path, box_file_name) 
        np.savetxt(pred_path, pred_array, delimiter=',')

    #save mots annotation file
    if seg_file == True:
        print(f"Saving segmentation masks in mots txt file.")
        mots_save_path = os.path.join(seg_path, seg_file_name)
        with open(mots_save_path, "w") as text_file:
            text_file.write(mots_string)

    #save annotated video file
    if visualize == True:
        print(f"Generating and saving mp4 video file ...")
        video_path = os.path.join(save_dir, 'annotated.mp4')
        image_list = os.listdir(image_dir)
        image_list.sort()
        image_list = [os.path.join(image_dir, item) for item in image_list]
        write_mp4(video_path, image_list, fps = fps) 
        print(f"Annotated video saved.")


    end_time = time.time()
    print(f"Task finished. Total time needed (in seconds): {end_time-start_time}.")
