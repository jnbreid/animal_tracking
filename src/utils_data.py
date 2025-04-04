
import cv2
import moviepy.video.io.ImageSequenceClip

"""
function to read a mp4 file and return a list of all frames in the video

Parameters:
- file_path (str)

Returns:
- list (containing f NxM numpy arrays)
"""
def read_mp4(file_path):
  frames = []

  video = cv2.VideoCapture(file_path)
  vid_len = video.get(cv2.CAP_PROP_FRAME_COUNT)
  for i in range(int(vid_len)):
    ret, fr = video.read()
    frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))

  return frames

"""
function to generate a mp4 file from image files 

Parameters:
- imgpaths_list (list containing x strings)
- fps (int)

Returns:
- bool
"""
def write_mp4(file_path, imgpaths_list, fps = 16):
    clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(imgpaths_list, fps=fps)
    clip.write_videofile(file_path)
    return True