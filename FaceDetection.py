# -*- coding: utf-8 -*-
"""
Created on Sat Oct 23 22:04:46 2021

@author: serdarhelli
"""

import numpy as np
import cv2




def readFaceNet(prototxt,caffemodel):
    faceNet = cv2.dnn.readNet(prototxt, caffemodel)
    return faceNet

def detect(image,FaceNet,confidence_input):
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    FaceNet.setInput(blob)
    detections = FaceNet.forward()
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the
        # prediction
        confidence = detections[0, 0, i, 2]
        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if confidence > confidence_input:
            # compute the (x, y)-coordinates of the bounding box for the
            # object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int") 
            roi=image[startY:endY,startX:endX].copy()
            return roi,confidence
    return None,None


class FaceDetection:
    def __init__(self,protxt_path,caffemodel_path):
        self.protxt=protxt_path
        self.caffemodel=caffemodel_path
        self.faceNet=readFaceNet(self.protxt,self.caffemodel)
    def detect_face(self,image,confidence):
        if isinstance(image, str):
            img=cv2.imread(image,cv2.IMREAD_COLOR)
        else:
            img=image
        detected_face,confidence=detect(img,self.faceNet,confidence)
        return detected_face,confidence



