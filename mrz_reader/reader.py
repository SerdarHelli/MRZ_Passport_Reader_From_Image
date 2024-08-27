import cv2
import numpy as np
import easyocr

from mrz_reader.segmentation import SegmentationNetwork, FaceDetection
from mrz_reader.utils import *

def instantiate_from_config_easyocr(config, reload=False):
    """
    Instantiates an EasyOCR Reader object using a configuration dictionary.

    Parameters:
    -----------
    config : dict
        Configuration dictionary containing parameters for easyocr.Reader.
    reload : bool, optional
        If True, reload the module before instantiation (default is False).

    Returns:
    --------
    easyocr.Reader
        An instance of easyocr.Reader configured based on the provided parameters.
    """
    print("Initializing EasyOCR...")
    return get_obj_from_str("easyocr.Reader", reload)(**config)

def get_obj_from_str(string, reload=False):
    """
    Dynamically loads and returns a class or function from a string.

    Parameters:
    -----------
    string : str
        The fully qualified name of the class or function (e.g., 'module.ClassName').
    reload : bool, optional
        If True, reload the module before returning the object (default is False).

    Returns:
    --------
    object
        The class or function specified by the string.
    """
    import importlib

    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

class MRZReader:
    """
    A class for reading Machine-Readable Zone (MRZ) data from images using segmentation, 
    face detection, and Optical Character Recognition (OCR).

    Attributes:
    -----------
    segmentation : SegmentationNetwork
        The segmentation model used to detect and segment MRZ in the image.
    face_detection : FaceDetection
        The face detection model used to identify and locate faces in the image.
    ocr_reader : easyocr.Reader
        The OCR reader used to extract text from the segmented MRZ regions.
    
    Methods:
    --------
    predict(image, do_facedetect=False, facedetect_coef=0.1, preprocess_config=None)
        Predicts MRZ text from the given image with optional face detection and preprocessing.
    
    recognize_text(image, preprocess_config)
        Recognizes text from the preprocessed image using OCR.
    
    _preprocess_image(img, preprocess_config)
        Applies preprocessing steps like skew correction, shadow deletion, and background clearing.
    
    _correct_skew(img)
        Corrects the skewness of the image if detected.
    
    _delete_shadow(img)
        Removes shadows from the image if detected.
    
    _clear_background(img)
        Clears the background of the image if detected.
    
    _apply_morphological_operations(img)
        Applies dilation and erosion to the image to enhance features.
    
    _apply_threshold(img)
        Applies binary thresholding to the image to prepare it for OCR.
    """

    def __init__(self, 
                 easy_ocr_params: dict,
                 facedetection_protxt: str = "./weights/face_detector/deploy.prototxt",
                 facedetection_caffemodel: str = "./weights/face_detector/res10_300x300_ssd_iter_140000.caffemodel",
                 segmentation_model: str = "./weights/mrz_detector/mrz_seg.tflite"):
        """
        Initializes the MRZReader with segmentation, face detection, and OCR models.

        Parameters:
        -----------
        easy_ocr_params : dict
            Keyword arguments to configure the EasyOCR reader.
        facedetection_protxt : str
            Path to the face detection model's deploy.prototxt file.
        facedetection_caffemodel : str
            Path to the face detection model's .caffemodel file.
        segmentation_model : str
            Path to the segmentation model file in .tflite format.
        """
        self.segmentation = SegmentationNetwork(segmentation_model)
        self.face_detection = FaceDetection(facedetection_protxt, facedetection_caffemodel)
        self.ocr_reader = instantiate_from_config_easyocr(easy_ocr_params)

    def predict(self, image, do_facedetect=False, facedetect_coef=0.1, preprocess_config=None):
        """
        Predicts MRZ text from the given image with optional face detection and preprocessing.

        Parameters:
        -----------
        image : str or numpy.ndarray
            Path to the image file or an image array.
        do_facedetect : bool, optional
            Whether to perform face detection (default is False).
        facedetect_coef : float, optional
            Confidence coefficient for face detection (default is 0.1).
        preprocess_config : dict, optional
            Configuration dictionary for preprocessing steps (default is None).

        Returns:
        --------
        tuple
            A tuple containing the recognized text, segmented image, and detected face (if any).
        """
        if isinstance(image, str):
            img = cv2.imread(image, cv2.IMREAD_COLOR)
        else:
            img = image

        face = None
        # Segmentation prediction
        segmented_image = self.segmentation.predict(img)

        # Optional face detection
        if do_facedetect:
            face, face_coef = self.face_detection.detect(img, facedetect_coef)

        # Text recognition
        text_results = self.recognize_text(segmented_image, preprocess_config or {})
        return text_results, segmented_image, face

    def recognize_text(self, image, preprocess_config):
        """
        Recognizes text from the preprocessed image using OCR.

        Parameters:
        -----------
        image : str or numpy.ndarray
            Path to the image file or an image array.
        preprocess_config : dict
            Configuration dictionary for preprocessing steps.

        Returns:
        --------
        list
            A list of tuples containing the recognized text and bounding box information.
        """
        if isinstance(image, str):
            img = cv2.imread(image, cv2.IMREAD_COLOR)
        else:
            img = image

        # Preprocessing steps
        if preprocess_config.get("do_preprocess", False):
            img = self._preprocess_image(img, preprocess_config)

        return self.ocr_reader.readtext(img)

    def _preprocess_image(self, img, preprocess_config):
        """
        Applies preprocessing steps like skew correction, shadow deletion, and background clearing.

        Parameters:
        -----------
        img : numpy.ndarray
            The image array to preprocess.
        preprocess_config : dict
            Configuration dictionary for preprocessing steps.

        Returns:
        --------
        numpy.ndarray
            The preprocessed image array.
        """
        img = resize(img)

        if preprocess_config.get("skewness", False):
            img = self._correct_skew(img)

        if preprocess_config.get("delete_shadow", False):
            img = self._delete_shadow(img)

        if preprocess_config.get("clear_background", False):
            img = self._clear_background(img)

        # Further image processing
        img = self._apply_morphological_operations(img)
        img = self._apply_threshold(img)

        return img

    def _correct_skew(self, img):
        """
        Corrects the skewness of the image if detected.

        Parameters:
        -----------
        img : numpy.ndarray
            The image array to correct skewness.

        Returns:
        --------
        numpy.ndarray
            The skew-corrected image array.
        """
        try:
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            angle = determine_skew(gray_img)
            rotated = rotate(img, angle, (0, 0, 0))
            return rotated
        except Exception as e:
            print(f"Skew correction failed: {e}")
            return img

    def _delete_shadow(self, img):
        """
        Removes shadows from the image if detected.

        Parameters:
        -----------
        img : numpy.ndarray
            The image array to remove shadows.

        Returns:
        --------
        numpy.ndarray
            The shadow-removed image array.
        """
        try:
            return delete_shadow(img)
        except Exception as e:
            print(f"Shadow deletion failed: {e}")
            return img

    def _clear_background(self, img):
        """
        Clears the background of the image if detected.

        Parameters:
        -----------
        img : numpy.ndarray
            The image array to clear the background.

        Returns:
        --------
        numpy.ndarray
            The background-cleared image array.
        """
        try:
            return clear_background(img)
        except Exception as e:
            print(f"Background clearing failed: {e}")
            return img

    def _apply_morphological_operations(self, img):
        """
        Applies dilation and erosion to the image to enhance features.

        Parameters:
        -----------
        img : numpy.ndarray
            The image array to apply morphological operations.

        Returns:
        --------
        numpy.ndarray
            The morphologically processed image array.
        """
        kernel = np.ones((2, 2), np.uint8)
        img = cv2.dilate(img, kernel, iterations=1)
        img = cv2.erode(img, kernel, iterations=1)
        return img

    def _apply_threshold(self, img):
        """
        Applies binary thresholding to the image to prepare it for OCR.

        Parameters:
        -----------
        img : numpy.ndarray
            The image array to apply thresholding.

        Returns:
        --------
        numpy.ndarray
            The thresholded image array.
        """
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        _, _, v = cv2.split(hsv)
        v = np.uint8(cv2.normalize(v, v, 50, 255, cv2.NORM_MINMAX))
        _, thresh0 = cv2.threshold(v, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thresh1 = cv2.adaptiveThreshold(v, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 33, 2)
        return cv2.bitwise_or(thresh0, thresh1)
