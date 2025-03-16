
import numpy as np
import cv2

from utils_loader import get_detector, get_segmentor, get_extractor, get_DistNet
from utils import overlay, col_list
from track import Tracker


def infer_video(img_seq, 
                confidences = None,
                bbox = None,
                c_id = None,
                visualize = True, 
                box_file = True, 
                seg_file = True, 
                save_dir = 'out_dir', 
                distnet = None,
                detector = None, 
                extractor = None, 
                segmentor = None, 
                device = None, 
                refine = True, 
                dist_mode = 'default'):
    
    if device is None:
        device = 'cpu'

    if detector is None:
        detector = get_detector()
    if extractor is None:
        extractor = get_extractor()
    if segmentor is None:
        segmentor = get_segmentor()
    if distnet is None:
        distnet = get_DistNet()
    

    mot_tracker = Tracker(distnet, max_age = 15, min_hits = 3, iou_threshold = 0.1)
    colors = col_list()


    if box_file == True:
        full_tracks = []

    for i in range(len(img_seq)):





img_seq, confidences, bbox, c_id = dset.__getitem__(elem)


for i in range(len(confidences)):
full_detect = []
c_masks = []
c_feature_vects = []

current_img_path = img_seq[i]
curr_img = cv2.imread(current_img_path)
megadesc_img = Image.open(current_img_path)
print(f"{i+1}/{len(confidences)}  - ", current_img_path)
image_rgb = cv2.cvtColor(curr_img, cv2.COLOR_BGR2RGB)

mask_predictor.set_image(image_rgb)

img_name = current_img_path.split('/')[-1].split('.')[0] + '.npy'

mask_file_path = os.path.join(save_path, img_name)
detec_file_path = os.path.join(save_path, ('tracks_' + img_name))
current_boxes = bbox[i].copy()

#print(current_boxes)

scores = confidences[i]#np.ones(current_boxes.shape[0])
#if current_boxes.shape[0] > 0:
for j in range(current_boxes.shape[0]):
    c_box = current_boxes[j,:].copy()
    c_box[2] = c_box[0] + c_box[2]
    c_box[3] = c_box[1] + c_box[3]

    pred_mask, _, _ = mask_predictor.predict(box=c_box, multimask_output=False)
    pred_mask = pred_mask[0,:,:].astype(np.int8)
    c_masks.append(pred_mask)

    current_crop = megadesc_img.crop(c_box)
    feature_vect = apply_megadescriptor(megadescriptor, current_crop)
    feature_vect = feature_vect.detach().cpu().numpy().reshape(-1)
    c_feature_vects.append(feature_vect)

    # fit box to calculated mask
    #refined_indexes = np.where(pred_mask == 1)
    #ref_x0 = int(refined_indexes[0].min())
    #ref_x1 = int(refined_indexes[0].max())
    #ref_y0 = int(refined_indexes[1].min())
    #ref_y1 = int(refined_indexes[1].max())
    #box = np.asarray([ref_y0, ref_x0, ref_y1,  ref_x1])

    full_detect.append(list(c_box)+[scores[j]])


if current_boxes.shape[0] == 0:
    full_detect = np.empty((0, 5))
    c_masks = np.asarray([])
    c_feature_vects = np.asarray([])

tracks = mot_tracker.update(np.array(full_detect), c_masks, c_feature_vects)
#print(np.array(tracks[:, 0:4], dtype = np.int8))

#print(np.array(tracks, dtype = np.int16))
masks = []
save_detections = []

for j in range(tracks.shape[0]):
    item = tracks[j,:]
    x0 = int(item[0])
    y0 = int(item[1])
    x1 = int(item[2])
    y1 = int(item[3])
    track_label = str(int(item[4]))

    #  --------------frame, id, b_left, b_top, b_width, b_height, conf, x, y, z
    # with x = y = z = -1 as there are no 3d coordinates (here conf is always 1 beceause we use original annotations)

    box = np.asarray([x0,y0,x1,y1])
    pred_mask, _, _ = mask_predictor.predict(box=box, multimask_output=False)
    pred_mask = pred_mask[0,:,:].astype(np.int8)
    # refine box
    if refine == True:
    refined_indexes = np.where(pred_mask == 1)

    ref_x0 = int(refined_indexes[0].min())
    ref_x1 = int(refined_indexes[0].max())

    ref_y0 = int(refined_indexes[1].min())
    ref_y1 = int(refined_indexes[1].max())

    box = np.asarray([ref_y0, ref_x0, ref_y1,  ref_x1])

    masks.append(pred_mask)
    save_detections.append([item[4], box[0], box[1], box[2]-box[0], box[3]-box[1]])

    c_score = scores[j]

    full_tracks.append([i, item[4], box[0], box[1], box[2]-box[0], box[3]-box[1], c_score, -1, -1, -1])

    #curr_img = overlay(img_bgr, pred_mask, (0,255,0), alpha = 0.5)
    #cv2.rectangle(curr_img, (x0, y0), (x1, y1), (0,255,255), 4)
    #cv2.putText(curr_img, '#'+track_label, (x0+5, y0-10), 0,0.6,(0,255,255),thickness=4)
masks = np.asarray(masks)
#np.save(mask_file_path, masks)
save_detecs = np.asarray(save_detections)
#np.save(detec_file_path, save_detecs)
#cv2.imwrite(new_img_path, curr_img)

pred_array = np.asarray(full_tracks)

pred_path = save_path + str(c_id) + '.txt'#str(elem) + '.txt'
print(pred_path)#, img_seq[0])
np.savetxt(pred_path, pred_array, delimiter=',')