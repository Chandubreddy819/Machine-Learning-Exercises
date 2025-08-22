'''
Created on 05.10.2016
Modified on 23.12.2020

@author: Daniel
@modified by Charly, Christian, Max (23.12.2020)
'''

import numpy as np
import cv2
import matplotlib.pyplot as plt


# do not import more modules!


def drawCircle(img, x, y):
    '''
    Draw a circle at circle of radius 5px at (x, y) stroke 2 px
    This helps you to visually check your methods.
    :param img: a 2d nd-array
    :param y:
    :param x:
    :return: img with circle at desired position
    '''
    cv2.circle(img, (x, y), 5, 255, 2)
    return img


def binarizeAndSmooth(img) -> np.ndarray:
    '''
    First Binarize using threshold of 115, then smooth with gauss kernel (5, 5)
    :param img: greyscale image in range [0, 255]
    :return: preprocessed image
    '''
    _, threshold_image = cv2.threshold(img, 115, 255, cv2.THRESH_BINARY)
    convolve_img = cv2.GaussianBlur(threshold_image, (5, 5), 0)
    return convolve_img


def drawLargestContour(img) -> np.ndarray:
    '''
    find the largest contour and return a new image showing this contour drawn with cv2 (stroke 2)
    :param img: preprocessed image (mostly b&w)
    :return: contour image
    '''
    contour_image = np.zeros_like(img)
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    cv2.drawContours(contour_image, [largest_contour], -1, (255, 255, 255), 2)
    return contour_image


def getFingerContourIntersections(contour_img, x) -> np.ndarray:
    '''
    Run along a column at position x, and return the 6 intersecting y-values with the finger contours.
    (For help check Palmprint_Algnment_Helper.pdf section 2b)
    :param contour_img:
    :param x: position of the image column to run along
    :return: y-values in np.ndarray in shape (6,)
    '''
    column = contour_img[:, x]
    transitions = np.where(np.diff(column) == 255)[0] + 1
    y_values = transitions[:6]
    return y_values


def findKPoints(img, y1, x1, y2, x2) -> tuple:
    '''
    given two points and the contour image, find the intersection point k
    :param img: binarized contour image (255 == contour)
    :param y1: y-coordinate of point (float)
    :param x1: x-coordinate of point (float)
    :param y2: y-coordinate of point (float)
    :param x2: x-coordinate of point (float)
    :return: intersection point k as a tuple (ky, kx)
    '''
    m = (y2 - y1) / (x2 - x1)
    c = y1 - m * x1

    height, width = img.shape

    mask = np.zeros_like(img)
    start_point = (0, int(c)) if c is not None else (0, y1)
    end_point = (width, int(m * width + c)) if c is not None else (width, y2)
    cv2.line(mask, start_point, end_point, (190, 200, 255), 1)

    # Find intersection points
    intersection = cv2.bitwise_and(img, mask)
    intersections = np.argwhere(intersection > 0)

    distances = np.sqrt((intersections[:, 0] - y1) ** 2 + (intersections[:, 1] - x1) ** 2)
    closest_idx = np.argmin(distances)
    ky, kx = intersections[closest_idx]

    return ky, kx


def getCoordinateTransform(k1, k2, k3) -> np.ndarray:
    '''
    Get a transform matrix to map points from old to new coordinate system defined by k1-3
    Hint: Use cv2 for this.
    :param k1: point in (y, x) order
    :param k2: point in (y, x) order
    :param k3: point in (y, x) order
    :return: 2x3 matrix rotation around origin by angle
    '''
    m = (k3[0] - k1[0]) / (k3[1] - k1[1])
    c = k3[0] - m * k3[1]
    perpendicular_slope = -1 / m
    c1 = k2[0] - perpendicular_slope * k2[1]

    x_intersect = (c1 - c) / (m - perpendicular_slope)
    y_intersect = m * x_intersect + c

    angle = np.arctan(perpendicular_slope)

    return cv2.getRotationMatrix2D((y_intersect, x_intersect), np.degrees(angle), scale=1.0)


def palmPrintAlignment(img):
    '''
    Transform a given image like in the paper using the helper functions above when possible
    :param img: greyscale image
    :return: transformed image
    '''

    # TODO threshold and blur
    smooth_and_binary_image = binarizeAndSmooth(img)

    # TODO find and draw largest contour in image
    largest_counter = drawLargestContour(smooth_and_binary_image)

    # TODO choose two suitable columns and find 6 intersections with the finger's contour
    x1 = 10
    x2 = 15
    y1 = getFingerContourIntersections(largest_counter, x1)
    y2 = getFingerContourIntersections(largest_counter, x2)

    # TODO compute middle points from these contour intersections
    midpoints_y1 = y1.reshape(-1, 2).mean(axis=1)
    midpoints_y2 = y2.reshape(-1, 2).mean(axis=1)

    k1 = findKPoints(largest_counter, int(midpoints_y1[0]), x1, int(midpoints_y2[0]), x2)
    k2 = findKPoints(largest_counter, int(midpoints_y1[1]), x1, int(midpoints_y2[1]), x2)
    k3 = findKPoints(largest_counter, int(midpoints_y1[2]), x1, int(midpoints_y2[2]), x2)

    # TODO extrapolate line to find k1-3
    transform_matrix = getCoordinateTransform(k1, k2, k3)

    # TODO calculate Rotation matrix from coordinate system spanned by k1-3
    transformed_img = cv2.warpAffine(img, transform_matrix, (img.shape[1], img.shape[0]))
    # TODO rotate the image around new origin
    return transformed_img
