
import cv2

def read_mp4(file_path):
  frames = []

  video = cv2.VideoCapture(file_path)
  vid_len = video.get(cv2.CAP_PROP_FRAME_COUNT)
  for i in range(vid_len):
    ret, fr = video.read()
    frames.append(fr)

  return frames