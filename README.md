# MRZ Passport Reader from Image

This project is an implementation of a Machine-Readable Zone (MRZ) reader from images using segmentation, face detection, and Optical Character Recognition (OCR). The implementation leverages TensorFlow Lite models for segmentation, a Caffe model for face detection, and EasyOCR for text recognition.

## Features

- **MRZ Detection**: Automatically detects and segments the MRZ region in passport images.
- **Face Detection**: Identifies and crops the face from the passport image.
- **OCR with EasyOCR**: Extracts text from the segmented MRZ region using EasyOCR.
- **Preprocessing**: Includes optional preprocessing steps such as skew correction, shadow removal, and background clearing to improve OCR accuracy.

## Installation

### Prerequisites

- Python 3.10 or higher
- Install the required Python packages from the `requirements.txt` file:

```bash
git clone https://github.com/SerdarHelli/MRZ_Passport_Reader_From_Image.git
cd MRZ_Passport_Reader_From_Image
pip install -e . -q
```


## Example Usage
```python

import json
import cv2
from mrz_reader.reader import MRZReader


# Initialize the MRZReader
reader = mrz_reader.reader.MRZReader( 
    facedetection_protxt = "./weights/face_detector/deploy.prototxt",
    facedetection_caffemodel = "./weights/face_detector/res10_300x300_ssd_iter_140000.caffemodel",
    segmentation_model = "./weights/mrz_detector/mrz_seg.tflite",
    easy_ocr_params = { "lang_list": ["en"], "gpu": False }

)
# Load an image
image_path = 'path_to_your_image.jpg'

# Perform MRZ reading with preprocessing and face detection

text_results,segmented_image ,detected_face = reader.predict(
    image_path,
    do_facedetect = True,
    preprocess_config = {
            "do_preprocess": False,
            "skewness": False,
            "delete_shadow": False,
            "clear_background": False
        } # or {} send empty

)
# Display results
print("Recognized Text:")
for result in text_results:
    bbox, text, confidence = result
    print(f"Bounding Box on segmented_image: {bbox}")
    print(f"Recognized Text: {text}")
    print(f"Confidence: {confidence:.2f}")
    print("-" * 50)

if detected_face is not None:
    print("Face detected in the image.")

# Display the images
cv2.imshow("Segmented Image", segmented_image)
if detected_face is not None:
    cv2.imshow("Detected Face", detected_face)
cv2.waitKey(0)
cv2.destroyAllWindows()

```

 ## Example Result 
 
 
<img align="left" width="33%" src="https://github.com/SerdarHelli/MRZ_Passport_Reader_From_Image/blob/main/example.jpg">

<br/><br/>

After you give this image to the models , you will take this result ***P<GBRUNITED<KINGDOM<FIVE<<JODIE<PIPPA<<<<<<<1071857032GBR8501178F1601312<<<<<<<<<<<<<<02*** as a string. This study is a basic solution . Your image which you give  should be clear, and it should cover whole area.
<br/><br/>
<br/><br/>


