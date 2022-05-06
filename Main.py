# -*- coding: utf-8 -*-
"""
Created on Sat Oct 23 22:16:32 2021

@author: serdarhelli
"""
import sys
sys.path.append('./')
from SegmentationMRZ_OLD import SegmentationMRZ_OLD
from FaceDetection import FaceDetection
from SegmentationMRZ_DL import SegmentationMRZ_DL
from TextRecognition import TextRecognition
import pytesseract


class mrz_reader:
    
    def __init__(self, facedetection_protxt="./face_detector/deploy.prototxt",
                 facedetection_caffemodel="./face_detector/res10_300x300_ssd_iter_140000.caffemodel",
                 mrzdetection_model="./mrz_detector/mrz_seg.tflite",
                 tesseract_exe=r"./Tesseract-OCR/tesseract.exe"):
        self.facedetection_protxt=facedetection_protxt
        self.facedetection_caffemodel=facedetection_caffemodel
        self.mrzdetection_model=mrzdetection_model
        self.tesseract_exe=tesseract_exe
        self.facedetect=True
        self.skewness=False
        self.delete_shadows=True
        self.clear_background=True
        self.IsLoad=False
        
    def load(self,tesseract_models="mrz+OCRB"):
        pytesseract.pytesseract.tesseract_cmd =self.tesseract_exe
        self.facedetection=FaceDetection(self.facedetection_protxt,
                                    self.facedetection_caffemodel)
        self.mrzdetection=SegmentationMRZ_DL(self.mrzdetection_model)
        self.text_recognition=TextRecognition(tesseract_models)
        self.IsLoad=True

        
    def predict(self,img):
        if self.IsLoad:
            self.img_dl=self.mrzdetection.predict(img)
            self.mrz_dl,self.threshdl=self.text_recognition.recognize(self.img_dl,
                                                                      self.skewness,
                                                                      self.delete_shadows,
                                                                      self.clear_background)
            if self.facedetect==True:      
                self.face=self.facedetection.detect_face(img,.1)
                return self.mrz_dl,self.face
            return self.mrz_dl
        else:
            print("Firstly, You must compile models")

mrz_reader=mrz_reader()
mrz_reader.load()
mrz_dl,face=mrz_reader.predict("./example1.jpg")