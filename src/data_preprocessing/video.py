from os.path import dirname, abspath
import sys
import numpy as np
from scipy.ndimage import rotate
from tqdm import tqdm
### Move to parent directory
parent_dir = dirname(dirname(abspath(__file__)))
sys.path.append(parent_dir)

def rotate_video(video, angle=0, reshape=True):
    if angle==0: return video
    else:
        video_rotated = []
        for frame in tqdm(video, desc="Rotating video frame", leave=False):
            video_rotated.append(rotate(frame, angle, (1, 0), reshape))
        return np.array(video_rotated)

def crop_video(video, height, width):
    return video[:, height[0]:height[1], width[0]:width[1]]

def crop_video_depend_scanposition(video, linescan_params):
    position = linescan_params['position']
    range = linescan_params['width']
    height = [position[0]-int(range/2), position[0]+int(range/2)]
    width = [position[1]-int(range/2), position[1]+int(range/2)]
    return crop_video(video, height, width)

def downsample_video(video, factor=2):
    new_shape = (-1, video.shape[1] // factor, factor, video.shape[2] // factor, factor)
    return video.reshape(new_shape).mean(axis=(2, 4))

def find_factor_for_downsample_video(video):
    _, a, b = video.shape
    gcd = np.gcd(a, b)
    factors = [i for i in range(1, gcd + 1) if gcd % i == 0]
    return factors