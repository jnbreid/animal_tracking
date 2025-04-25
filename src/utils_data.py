
import cv2
import moviepy.video.io.ImageSequenceClip


def read_mp4(file_path):
  """
    Reads an MP4 video file and returns a list of frames as numpy arrays.

    This function opens the provided MP4 video file, reads all the frames, and stores them as 
    numpy arrays. Each frame is converted to RGB format.

    Args:
        file_path (str): The path to the MP4 video file to be read.

    Returns:
        list: A list of numpy arrays where each element is an NxMx3 array representing a frame.
  """
  frames = []

  video = cv2.VideoCapture(file_path)
  vid_len = video.get(cv2.CAP_PROP_FRAME_COUNT)
  for i in range(int(vid_len)):
    ret, fr = video.read()
    frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))

  return frames


def write_mp4(file_path, imgpaths_list, fps = 16):
    """
    Generates an MP4 video file from a sequence of image files.

    This function takes a list of image file paths, compiles them into a video sequence.

    Args:
        file_path (str): The path where the generated MP4 video will be saved.
        imgpaths_list (list of str): A list of image file paths to be included in the video.
        fps (int, optional): Frames per second for the generated video. Default is 16.

    Returns:
        bool: Returns True when the video is successfully written to the file.
    """
    clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(imgpaths_list, fps=fps)
    clip.write_videofile(file_path)
    return True