# -*- coding: utf-8 -*-
"""
Created on Sat Oct 16 23:24:28 2021

@author: serdarhelli
"""

import pytesseract
import cv2
import numpy as np
from scipy.ndimage import interpolation as inter
import string
import math
from typing import Tuple, Union
from deskew import determine_skew

def delete_shadow(img):
    rgb_planes = cv2.split(img)

    result_planes = []
    result_norm_planes = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        norm_img = cv2.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        result_planes.append(diff_img)
        result_norm_planes.append(norm_img)
    #result = cv2.merge(result_planes)
    result_norm = cv2.merge(result_norm_planes)
    return result_norm



def clear_background(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    mask = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)[1]
    
    # negate mask
    mask = 255 - mask
    
    # apply morphology to remove isolated extraneous noise
    # use borderconstant of black since foreground touches the edges
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # anti-alias the mask -- blur then stretch
    # blur alpha channel
    mask = cv2.GaussianBlur(mask, (0,0), sigmaX=2, sigmaY=2, borderType = cv2.BORDER_DEFAULT)
    
    # linear stretch so that 127.5 goes to 0, but 255 stays 255
    mask = (2*(mask.astype(np.float32))-255.0).clip(0,255).astype(np.uint8)
    
    # put mask into alpha channel
    result = img.copy()
    result = cv2.cvtColor(result, cv2.COLOR_BGR2BGRA)
    result[:, :, 3] = mask
    return result

def rotate(
        image: np.ndarray, angle: float, background: Union[int, Tuple[int, int, int]]
) -> np.ndarray:
    old_width, old_height = image.shape[:2]
    angle_radian = math.radians(angle)
    width = abs(np.sin(angle_radian) * old_height) + abs(np.cos(angle_radian) * old_width)
    height = abs(np.sin(angle_radian) * old_width) + abs(np.cos(angle_radian) * old_height)

    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    rot_mat[1, 2] += (width - old_width) / 2
    rot_mat[0, 2] += (height - old_height) / 2
    return cv2.warpAffine(image, rot_mat, (int(round(height)), int(round(width))), borderValue=background)


def correct_skew(image, delta=1, limit=5):
    def determine_score(arr, angle):
        data = inter.rotate(arr, angle, reshape=False, order=0)
        histogram = np.sum(data, axis=1)
        score = np.sum((histogram[1:] - histogram[:-1]) ** 2)
        return histogram, score

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1] 

    scores = []
    angles = np.arange(-limit, limit + delta, delta)
    for angle in angles:
        histogram, score = determine_score(thresh, angle)
        scores.append(score)

    best_angle = angles[scores.index(max(scores))]

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, best_angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, \
              borderMode=cv2.BORDER_REPLICATE)

    return best_angle, rotated



def resize(image):
    if image.shape[1]>1500:
        return image
    else:
        image=cv2.resize(image, None, fx=1.2, fy=1.2, interpolation=cv2.INTER_CUBIC)
        return image








class TextRecognition:
    def __init__(self,lang_tesseract,):
        self.lang_tesseract=lang_tesseract
 

    def recognize(self,image,skewness,delete_shadows,clear_background):
        if isinstance(image, str):
            img=cv2.imread(image,cv2.IMREAD_COLOR)
        else:
            img=image
        img = resize(image)
        if skewness==True:
            try: 
                rotation = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                angle = determine_skew(rotation)
                rotated = rotate(img, angle, (0, 0, 0))
                img=rotated
            except:
                img=img
        if delete_shadows==True:
            try:
                img=delete_shadow(img)
            except:
                img=img
        if clear_background==True:  
            try:
                img=clear_background(img)
            except:
                img=img           
        kernel= np.ones((2,2), np.uint8)
        img = cv2.dilate(img, kernel, iterations=1)
        img = cv2.erode(img, kernel, iterations=1) 
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v = np.uint8(cv2.normalize(v, v, 50, 255, cv2.NORM_MINMAX))
        ret, thresh0 = cv2.threshold(v, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)   
        thresh1 = cv2.adaptiveThreshold(v, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 33, 2)
        thresh = cv2.bitwise_or(thresh0, thresh1)
        text = pytesseract.image_to_string(thresh, lang=self.lang_tesseract)
        text=text.translate({ord(c): None for c in string.whitespace})
        if len(text)==0:
            return None,None
        return text,thresh


# pytesseract.pytesseract.tesseract_cmd =r"C:/Users/sserd/Desktop/OCRPassport/Tesseract-OCR/tesseract.exe"
# text_recognition=TextRecognition("mrz+OCRB")

# mrz=text_recognition.recognize("C:/Users/sserd/Desktop/OCRPassport/ex.jpg")



