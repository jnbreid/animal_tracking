# Animal Tracker

## Overview
This project provides a pipeline for tracking and segmenting animals in video footage. 

It uses pre-trained models and basic models to facilitate the automatic analysis of animal videos. This pipeline is specifically designed for tracking in camera trap videos.
The pipeline requires minimal training and does not require data annotated with segmentation masks for training. This enables automated analysis of camera trap video in domains where little training data is available.

### Example videos

|Example video 1               | Example video 2               |
| ---------------------------- | ----------------------------- |
|![Example 1](assets/vid1.gif) | ![Example 2](assets/vid2.gif) |

The animals are annotated by applying the pipeline. The raw videos used as input for the pipeline can be found [here](demo_data). 

The videos are part of the [**GMOT-40 Benchmark**](https://github.com/Spritea/GMOT40).
The video data is licensed under a [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-nc-sa/4.0/).
Further information about the data can be found  [here](licenses.md) under point 7.

### Demo

The notebook [*demo_inference.ipynb*](demo_inference.ipynb) can be used to track animals in two [example videos](demo_data).

## Dependencies

To install all required packages run
~~~text 
pip install -r requirements.txt
~~~

## License

This project is licensed under a [GPL-3.0 license](LICENSE).
