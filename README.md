# MRZ_Passport_Reader_From_Image

This study detect and recognize MRZ ID from passport images. It includes our own model to segment mrz area then we are using Tesseract OCR to recognize text.


 
 
 ```python
 #For An Example
from mrz_reader import mrz_reader
 
mrz_reader=mrz_reader()
mrz_reader.load()
mrz_dl,face=mrz_reader.predict("./example.jpg")
 ```
