# SPDX-License-Identifier: GPL-3.0-only
# Copyright (c) 2025 Jon Breid

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
import wget

from src.model import DistNet_t



class megadetector_wrapper(object):
    """
    Wrapper class for the MegaDetector object detection model.

    This wrapper simplifies calling MegaDetector on RGB images by handling the output 
    format and filtering detections to include only the relevant class (class_id == 0).
    It can be called directly on an image using the __call__ method.

    Attributes:
        megadetector: An instance of the MegaDetector model.
    """
  
    def __init__(self, megadetector):
        self.megadetector = megadetector

    def __call__(self, image):
        """
        Runs the detector on a single RGB image and returns filtered bounding boxes.

        Args:
            image (np.ndarray): An RGB image (H x W x C format).

        Returns:
            tuple:
                - boxes (np.ndarray): Filtered bounding boxes in [x, y, w, h] format.
                - confidences (np.ndarray): Confidence scores for each detected box.
        """
        an_boxes = []
        an_confidences = []
        out = self.megadetector.single_image_detection(image)
        detections = out['detections']
        boxes = np.asarray(detections.xyxy, dtype = np.int32)
        labels = np.asarray(detections.class_id, dtype = np.int32)
        confidences = np.asarray(detections.confidence)
        for i in range(boxes.shape[0]):
            if labels[i] == 0: # remove all other found classes
                box = boxes[i,:]
                conf = confidences[i]
                an_boxes.append(box)
                an_confidences.append(conf)
        
        boxes = np.asarray(an_boxes, dtype=np.int32)
        boxes[:,2] = boxes[:,2]-boxes[:,0]
        boxes[:,3] = boxes[:,3]-boxes[:,1]

        return boxes, np.asarray(an_confidences)




def get_detector(device = None):
    """
    Loads a pretrained MegaDetector model and wraps it.

    Args:
        device (torch.device or str, optional): The device to load the model on.
                                                Defaults to 'cpu' if not specified.

    Returns:
        megadetector_wrapper: A wrapped MegaDetector instance ready for inference.
    """
    if device is None:
        device = 'cpu'

    detection_model = pw_detection.MegaDetectorV5(device=device, pretrained=True)
    
    detector = megadetector_wrapper(detection_model)

    return detector
    


from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

def get_segmentor(device = None):
    """
    Loads and initializes the Segment Anything Model (SAM) for segmentation tasks.

    Downloads the required checkpoint if not already present and returns a predictor object.

    Args:
        device (torch.device or str, optional): The device to load the model on.
                                                Defaults to 'cpu'.

    Returns:
        SamPredictor: An instance of the SAM predictor for segmentation.
    """
    if device is None:
        device = 'cpu'

    # get model files
    HOME = os.getcwd()
    if not os.path.exists(os.path.join(HOME, 'weights')):
        os.mkdir(os.path.join(HOME, 'weights'))
    if not os.path.isfile(os.path.join(HOME, 'weights', 'sam_vit_h_4b8939.pth')):
        os.system(f'wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -P {HOME}/weights')
    
    CHECKPOINT_PATH = os.path.join(HOME, "weights", "sam_vit_h_4b8939.pth")
    MODEL_TYPE = "vit_h"
    print(f"Initializing sam")
    sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH).to(device=device)
    mask_predictor = SamPredictor(sam)  
    return mask_predictor


class megadescriptor_wrapper(object):
    """
    Wrapper class for feature extraction using MegaDescriptor.

    Applies preprocessing transforms and returns model output on input image.

    Attributes:
        megadescriptor: A timm model used for feature extraction.
        device: The torch device where the model is loaded.
    """
    def __init__(self, megadescriptor, device):
        self.megadescriptor = megadescriptor
        self.device = device

    def __call__(self, image):
        """
        Extracts feature vector from a given RGB image.

        Args:
            image (np.ndarray): An RGB image (H x W x C).

        Returns:
            torch.Tensor: The feature vector of the image.
        """
        transform = T.Compose([T.ToTensor(),
                               T.Resize([224, 224]),
                               T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
        input = transform(image).unsqueeze(0).to(self.device)
        output = self.megadescriptor(input)
        return output


def get_extractor(device = None):
    """
    Loads a pretrained MegaDescriptor model for feature extraction.

    Args:
        device (torch.device or str, optional): The device to load the model on.
                                                Defaults to 'cpu'.

    Returns:
        megadescriptor_wrapper: A wrapped feature extractor model.
    """
    if device is None:
        device = 'cpu'

    model_name = 'hf-hub:BVRA/MegaDescriptor-T-224' # other existing model names can also be used

    megadescriptor = timm.create_model(model_name, num_classes=0, pretrained=True)#timm.create_model("hf-hub:BVRA/MegaDescriptor-L-384", pretrained=True)
    megadescriptor = megadescriptor.eval()
    megadescriptor.to(device)
    
    extractor = megadescriptor_wrapper(megadescriptor, device)

    return extractor



def get_DistNet(device = None, weight_path = None):
    """
    Loads the DistNet model and optionally loads pretrained weights.

    Args:
        device (torch.device or str, optional): The device to load the model on.
                                                Defaults to 'cpu'.
        weight_path (str, optional): Path to the pretrained weights file.

    Returns:
        DistNet_t: An instance of the DistNet model ready for inference.
    """
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

    
