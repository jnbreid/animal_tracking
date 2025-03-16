
import os
import numpy as np
from PIL import Image
import torch
import timm
import torchvision.transforms as T
from torch.utils.data import DataLoader
from PytorchWildlife.models import detection as pw_detection
from PytorchWildlife.data import transforms as pw_trans
from PytorchWildlife.data import datasets as pw_data
from PytorchWildlife import utils as pw_utils
from PIL import Image
from urllib.request import urlopen

from model import DistNet_t


class megadetector_wrapper(object):
  
    def __init__(self, megadetector):
        self.megadetector = megadetector

    # image must bei in rgb (c,h,w) format
    def __call__(self, image):
        an_boxes = []
        out = self.megadetector.single_image_detection(image)
        detections = out['detections']
        boxes = np.asarray(detections.xyxy, dtype = np.int32)
        labels = np.asarray(detections.class_id, dtype = np.int32)
        for i in range(boxes.shape[0]):
            if labels[i] == 0: # remove all other found classes
                box = boxes[i,:]
                an_boxes.append(box)
        
        return np.asarray(an_boxes, dtype=np.int32)



"""
function to load model and pretrained weights. here the megadetector is used
"""
def get_detector(device = None):
    if device is None:
        device = 'cpu'

    detection_model = pw_detection.MegaDetectorV5(device=device, pretrained=True)
    
    detector = megadetector_wrapper(detection_model)

    return detector
    

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
"""
function to load segmentation model
"""
def get_segmentor(device = None):
    if device is None:
        device = 'cpu'

    # get model files
    HOME = os.getcwd()
    print("HOME:", HOME)

    os.system(f'mkdir - {HOME}/weights')
    os.system(f'wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -P {HOME}/weights')
    
    CHECKPOINT_PATH = os.path.join(HOME, "weights", "sam_vit_h_4b8939.pth")
    MODEL_TYPE = "vit_h"

    sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH).to(device=device)
    mask_predictor = SamPredictor(sam)  
    return mask_predictor




def megadetector_wrapper(object):
    def __init__(self, megadescriptor, device):
        self.megadescriptor = megadescriptor
        self.device = device

    def __call__(self, image):
        transform = T.Compose([T.Resize([224, 224]),
                            T.ToTensor(),
                            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
        input = transform(image).unsqueeze(0).to(self.device)
        output = self.megadescriptor(input)
        return output


"""
function to load feature extractor model
"""
def get_extractor(device = None):
    if device is None:
        device = 'cpu'

    model_name = 'hf-hub:BVRA/MegaDescriptor-T-224' # other existing model names can also be used

    megadescriptor = timm.create_model(model_name, num_classes=0, pretrained=True)#timm.create_model("hf-hub:BVRA/MegaDescriptor-L-384", pretrained=True)
    megadescriptor = megadescriptor.eval()
    megadescriptor.to(device)
    
    extractor = megadetector_wrapper(megadescriptor, device)

    return extractor

"""
function to load distnet model
"""
def get_DistNet(device = None, weight_path = None):
    if device is None:
        device = 'cpu'
    
    model = DistNet_t()
    model.eval()
    model.to(device)

    if weight_path is None:
        return model
    
    model.load_state_dict(torch.load(weight_path, weights_only=True))
    model.eval()    

    return model

    
