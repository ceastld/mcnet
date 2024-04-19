import face_alignment
import skimage.io
import numpy
from argparse import ArgumentParser
from skimage.util import img_as_ubyte
from skimage.transform import resize
from tqdm import tqdm
import os
import imageio
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import cv2

def extract_bbox(frame, fa):
    if max(frame.shape[0], frame.shape[1]) > 640:
        scale_factor =  max(frame.shape[0], frame.shape[1]) / 640.0
        frame = resize(frame, (int(frame.shape[0] / scale_factor), int(frame.shape[1] / scale_factor)))
        frame = img_as_ubyte(frame)
    else:
        scale_factor = 1
    frame = frame[..., :3]
    bboxes = fa.face_detector.detect_from_image(frame[..., ::-1])
    if len(bboxes) == 0:
        return []
    return np.array(bboxes)[:, :-1] * scale_factor

def compute_bbox(tube_bbox, frame_shape, increase_area=0.1):
    left, top, right, bot = tube_bbox
    width = right - left
    height = bot - top
    #Computing aspect preserving bbox
    width_increase = max(increase_area, ((1 + 2 * increase_area) * height - width) / (2 * width))
    height_increase = max(increase_area, ((1 + 2 * increase_area) * width - height) / (2 * height))
    left = int(left - width_increase * width)
    top = int(top - height_increase * height)
    right = int(right + width_increase * width)
    bot = int(bot + height_increase * height)
    top, bot, left, right = max(0, top), min(bot, frame_shape[0]), max(0, left), min(right, frame_shape[1])
    return np.array((left, top, right, bot))

device = 'cpu'
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device=device)
img = cv2.imread('../../../data/chenglai/ori_imgs/000199.jpg')
bbox = extract_bbox(img[:,:,::-1], fa)
crop_rect = compute_bbox(bbox[0], img.shape[:2], increase_area=0.2)
crop_img = cv2.resize(img[crop_rect[1]:crop_rect[3], crop_rect[0]:crop_rect[2], :], (256, 256))
cv2.imwrite('source_crop.jpg', crop_img)