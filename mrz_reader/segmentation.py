import numpy as np
import cv2
# Import TFLite interpreter from tflite_runtime package
import tensorflow as tf
Interpreter = tf.lite.Interpreter


class SegmentationNetwork:
    """
    A class to perform segmentation using a TFLite model.

    Attributes:
    -----------
    interpreter : tflite.Interpreter
        The TFLite interpreter for the segmentation model.
    input_details : list
        Details about the input tensor for the model.
    output_details : list
        Details about the output tensor for the model.

    Methods:
    --------
    process(image)
        Preprocesses the input image to the required format.
    output(output_data, image)
        Processes the model's output to extract the region of interest (ROI).
    predict(image)
        Runs the segmentation model on the input image and returns the ROI.
    """

    def __init__(self, model_path):
        """
        Initializes the SegmentationNetwork with the given TFLite model.

        Parameters:
        -----------
        model_path : str
            Path to the TFLite model file.
        """
        self.interpreter = Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def process(self, image):
        """
        Preprocesses the input image to the required format for the model.

        Parameters:
        -----------
        image : str or numpy.ndarray
            Path to the image file or an image array.

        Returns:
        --------
        numpy.ndarray
            The preprocessed image array.
        """
        if isinstance(image, str):
            img = cv2.imread(image, cv2.IMREAD_COLOR)
        else:
            img = image
        img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_NEAREST)
        img = np.asarray(np.float32(img / 255))
        if len(img.shape) > 3:
            img = img[:, :, :3]
        img = np.reshape(img, (1, 256, 256, 3))
        return img

    def output(self, output_data, image):
        """
        Processes the model's output to extract the region of interest (ROI).

        Parameters:
        -----------
        output_data : numpy.ndarray
            The output data from the segmentation model.
        image : str or numpy.ndarray
            Path to the original image file or an image array.

        Returns:
        --------
        numpy.ndarray or None
            The extracted ROI or None if no valid ROI is found.
        """
        if isinstance(image, str):
            img = cv2.imread(image, cv2.IMREAD_COLOR)
        else:
            img = image
        shape = img.shape
        kernel = np.ones((5, 5), dtype=np.float32)
        output_data = (output_data[0, :, :, 0] > 0.35) * 1
        output_data = np.uint8(output_data * 255)
        img2 = cv2.resize(output_data, (shape[1], shape[0]))
        img2 = cv2.erode(img2, kernel, iterations=3)
        contours, _ = cv2.findContours(img2.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if len(contours) == 0:
            return None
        c_area = np.zeros([len(contours)])
        for j in range(len(contours)):
            c_area[j] = cv2.contourArea(contours[j]) 
        cnts = contours[np.argmax(c_area)]
        x, y, w, h = cv2.boundingRect(cnts)
        roi = img[y:y + h, x:x + w].copy()
        return roi

    def predict(self, image):
        """
        Runs the segmentation model on the input image and returns the ROI.

        Parameters:
        -----------
        image : str or numpy.ndarray
            Path to the image file or an image array.

        Returns:
        --------
        numpy.ndarray or None
            The extracted ROI or None if no valid ROI is found.
        """
        image_array = self.process(image)
        self.interpreter.set_tensor(self.input_details[0]['index'], image_array)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        output_data = self.output(output_data, image)
        return output_data


class FaceDetection:
    """
    A class to perform face detection using a Caffe model.

    Attributes:
    -----------
    faceNet : cv2.dnn_Net
        The loaded Caffe model for face detection.

    Methods:
    --------
    detect(image, confidence_input)
        Detects a face in the image and returns the region of interest (ROI).
    """

    def __init__(self, prototxt_path, caffemodel_path):
        """
        Initializes the FaceDetection with the given Caffe model files.

        Parameters:
        -----------
        prototxt_path : str
            Path to the Caffe model's deploy.prototxt file.
        caffemodel_path : str
            Path to the Caffe model's .caffemodel file.
        """
        self.faceNet = cv2.dnn.readNet(prototxt_path, caffemodel_path)

    def detect(self, image, confidence_input):
        """
        Detects a face in the image and returns the region of interest (ROI).

        Parameters:
        -----------
        image : str or numpy.ndarray
            Path to the image file or an image array.
        confidence_input : float
            The minimum confidence threshold for detecting a face.

        Returns:
        --------
        tuple
            A tuple containing the ROI (numpy.ndarray) and the confidence score (float).
            Returns (None, None) if no face is detected with sufficient confidence.
        """
        if isinstance(image, str):
            img = cv2.imread(image, cv2.IMREAD_COLOR)
        else:
            img = image
        (h, w) = img.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        self.faceNet.setInput(blob)
        detections = self.faceNet.forward()
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > confidence_input:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                roi = img[startY:endY, startX:endX].copy()
                return roi, confidence
        return None, None
