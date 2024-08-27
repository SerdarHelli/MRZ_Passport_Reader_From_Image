

import cv2
import numpy as np
from scipy.ndimage import interpolation as inter
import string
import math
from typing import Tuple, Union
from deskew import determine_skew

def delete_shadow(img: np.ndarray) -> np.ndarray:
    """
    Removes shadows from an image.

    Parameters:
    -----------
    img : numpy.ndarray
        Input image in which shadows are to be removed.

    Returns:
    --------
    numpy.ndarray
        Image with shadows removed.
    """
    rgb_planes = cv2.split(img)

    result_planes = []
    result_norm_planes = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((7, 7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        result_planes.append(diff_img)
        result_norm_planes.append(norm_img)

    result_norm = cv2.merge(result_norm_planes)
    return result_norm

def clear_background(img: np.ndarray) -> np.ndarray:
    """
    Clears the background of an image and enhances the foreground.

    Parameters:
    -----------
    img : numpy.ndarray
        Input image whose background is to be cleared.

    Returns:
    --------
    numpy.ndarray
        Image with background cleared and enhanced foreground.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)[1]

    # Negate mask
    mask = 255 - mask

    # Apply morphology to remove isolated extraneous noise
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Anti-alias the mask - blur then stretch
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=2, sigmaY=2, borderType=cv2.BORDER_DEFAULT)
    
    # Linear stretch so that 127.5 goes to 0, but 255 stays 255
    mask = (2 * (mask.astype(np.float32)) - 255.0).clip(0, 255).astype(np.uint8)

    # Put mask into alpha channel
    result = img.copy()
    result = cv2.cvtColor(result, cv2.COLOR_BGR2BGRA)
    result[:, :, 3] = mask
    return result

def rotate(image: np.ndarray, angle: float, background: Union[int, Tuple[int, int, int]]) -> np.ndarray:
    """
    Rotates an image around its center.

    Parameters:
    -----------
    image : numpy.ndarray
        Input image to be rotated.
    angle : float
        Angle by which the image is to be rotated.
    background : int or Tuple[int, int, int]
        Background color to be used in the empty regions after rotation.

    Returns:
    --------
    numpy.ndarray
        Rotated image.
    """
    old_width, old_height = image.shape[:2]
    angle_radian = math.radians(angle)
    width = abs(np.sin(angle_radian) * old_height) + abs(np.cos(angle_radian) * old_width)
    height = abs(np.sin(angle_radian) * old_width) + abs(np.cos(angle_radian) * old_height)

    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    rot_mat[1, 2] += (width - old_width) / 2
    rot_mat[0, 2] += (height - old_height) / 2
    return cv2.warpAffine(image, rot_mat, (int(round(height)), int(round(width))), borderValue=background)

def correct_skew(image: np.ndarray, delta: int = 1, limit: int = 5) -> Tuple[float, np.ndarray]:
    """
    Corrects skewness in an image.

    Parameters:
    -----------
    image : numpy.ndarray
        Input image in which skewness is to be corrected.
    delta : int, optional
        Incremental step size for angle testing (default is 1).
    limit : int, optional
        Maximum angle to test for skewness correction (default is 5).

    Returns:
    --------
    Tuple[float, numpy.ndarray]
        Tuple containing the best angle for correction and the rotated image.
    """
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
        _, score = determine_score(thresh, angle)
        scores.append(score)

    best_angle = angles[scores.index(max(scores))]

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, best_angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    return best_angle, rotated

def resize(image: np.ndarray) -> np.ndarray:
    """
    Resizes an image if its width is less than 1500 pixels.

    Parameters:
    -----------
    image : numpy.ndarray
        Input image to be resized.

    Returns:
    --------
    numpy.ndarray
        Resized image.
    """
    if image.shape[1] > 1500:
        return image
    else:
        image = cv2.resize(image, None, fx=1.2, fy=1.2, interpolation=cv2.INTER_CUBIC)
        return image

