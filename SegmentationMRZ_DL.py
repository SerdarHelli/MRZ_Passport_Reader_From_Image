# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 18:17:57 2021

@author: serdarhelli
"""

import numpy as np
import cv2
# try:
#   # Import TFLite interpreter from tflite_runtime package if it's available.
#   from tflite_runtime.interpreter import Interpreter
# except ImportError:
#   # If not, fallback to use the TFLite interpreter from the full TF package.
import tensorflow as tf
Interpreter = tf.lite.Interpreter


class SegmentationMRZ_DL:
    def __init__(self,model_path):
        self.interpreter = Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
    def process(self,image):
        if isinstance(image, str):
            img=cv2.imread(image,cv2.IMREAD_COLOR)
        else:
            img=image
        img=cv2.resize(img,(256,256),interpolation=cv2.INTER_NEAREST)
        img=np.asarray(np.float32(img/255))
        if len(img.shape)>3:
            img=img[:,:,:3]
        img=np.reshape(img,(1,256,256,3))
        return img
    
    def output(self,output_data,image):
        if isinstance(image, str):
            img=cv2.imread(image,cv2.IMREAD_COLOR)
        else:
            img=image
        shape=img.shape
        kernel =( np.ones((5,5), dtype=np.float32))
        output_data=(output_data[0,:,:,0]>0.35)*1
        output_data=np.uint8(output_data*255)
        img2=cv2.resize(output_data,(shape[1],shape[0]))
        img2=cv2.erode(img2,kernel,iterations =3)
        contours, hierarchy = cv2.findContours(img2.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if len(contours)==0:
            return None
        c_area=np.zeros([len(contours)])
        for j in range(len(contours)):
            c_area[j]= cv2.contourArea(contours[j]) 
        cnts=contours[np.argmax(c_area)]
        x, y, w, h = cv2.boundingRect(cnts)
        roi = img[y:y + h, x:x + w].copy()
        return roi
    
    def predict(self,image): 
        image_array=self.process(image)
        self.interpreter.set_tensor(self.input_details[0]['index'], image_array)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        output_data=self.output(output_data,image)
        return output_data

# AI=SegmentationMRZ_DL("C:/Users/sserd/Desktop/OCRPassport/ModelsAI/mrz_seg.tflite")            
# output=AI.predict("C:/Users/sserd/Desktop/OCRPassport/example1.jpg")        




