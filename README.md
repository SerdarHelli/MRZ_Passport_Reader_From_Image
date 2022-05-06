# MRZ_Passport_Reader_From_Image

The of this study are to detect and recognize MRZ ID from one-shot passport images. It includes our own model to segment mrz area. Then, we are using Tesseract OCR to recognize text.


 
 ## Setup Tesseract 

- Setup Tesseract into Tesseract-OCR directory - [You should download tesseract-ocr-w64-setup-v5.0.0-alpha.20210811 version](https://digi.bib.uni-mannheim.de/tesseract/)
- Then , Copy all files which are at ./tesseract_trained  to ./Tesseract-OCR/tessdata .These files are trained models.
- Finally , run this command  ```pip install -r ./requirements.txt ``` to install necessary libraries


Now, You are ready
 
 
 
  ```python

from mrz_reader import mrz_reader
 
mrz_reader=mrz_reader()
mrz_reader.load()

#If you want detect face 
mrz_reader.facedetect=True

#Befero tesseract , check skewness
mrz_reader.skewness=False

#Befero tesseract , delete shadows
mrz_reader.delete_shadows=True

#Befero tesseract , clear background
mrz_reader.clear_background=True

mrz_dl,face=mrz_reader.predict("./example.jpg")

 ```
 ## Example Result 
 
 
<img align="left" width="33%" src="https://github.com/SerdarHelli/MRZ_Passport_Reader_From_Image/blob/main/example.jpg">

<br/><br/>

After you give this image to the models , you will take this result ***P<GBRUNITED<KINGDOM<FIVE<<JODIE<PIPPA<<<<<<<1071857032GBR8501178F1601312<<<<<<<<<<<<<<02*** as a string. This study is a basic solution . Your image which you give  should be clear, and it should cover whole area.
<br/><br/>
<br/><br/>
<br/><br/>


 ## Libraries - Refs
 
 - [Tesseract OCR Engine ](https://github.com/tesseract-ocr/tesseract)
 - [Skewness ](https://github.com/sbrunner/deskew)
 
