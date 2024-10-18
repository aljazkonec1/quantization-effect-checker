import numpy as np 
import cv2

def resize_and_pad(img: np.array, target_size = (512, 288)) -> np.array:
    img_h, img_w, _ = img.shape
    
    scale = target_size[0] / img_w
    new_w, new_h = target_size[0], min(int(img_h * scale), target_size[1])
    
    resized_image = cv2.resize(img, (new_w, new_h))
    pad_h = 0
    if new_h < target_size[1]:
        pad_h = target_size[1] - new_h
    
    top = pad_h // 2
    bottom = pad_h - top
    left = 0
    right = 0
    
    padded_image = cv2.copyMakeBorder(resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    
    return padded_image