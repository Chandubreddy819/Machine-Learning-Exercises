'''
Created on 04.10.2016

@author: Daniel Stromer
@modified by Charly, Christian, Max (23.12.2020)
'''
import cv2
import numpy as np
import matplotlib.pyplot as plt


# do not import more modules!


def polarToKart(shape, r, theta):
    '''
    convert polar coordinates with origin in image center to kartesian
    :param shape: shape of the image
    :param r: radius from image center
    :param theta: angle
    :return: y, x
    '''
    height, width = shape
    center_x = width // 2
    center_y = height // 2
    # Convert polar to Cartesian coordinates
    x_prime = r * np.cos(theta)
    y_prime = r * np.sin(theta)

    # Adjust coordinates relative to the image center
    x = center_x + x_prime
    y = center_y + y_prime
    return y, x


def calculateMagnitudeSpectrum(img) -> np.ndarray:
    '''
    use the fft to generate a magnitude spectrum and shift it to the image center.
    Hint: This can be done with numpy :)
    Return Magnitude in Decibel
    :param img:
    :return:
    '''
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log10(np.abs(fshift))
    return magnitude_spectrum


def extractRingFeatures(magnitude_spectrum, k, sampling_steps) -> np.ndarray:
    '''
    Follow the approach to extract ring features
    :param magnitude_spectrum:
    :param k: number of rings to extract = #features
    :param sampling_steps: times to sample one ring --> theta/sampling rate
    :return: feature vector of k features
    '''

    shape = magnitude_spectrum.shape
    h, w = shape

    feature_vector = np.zeros(k)

    for i in range(1, k+1):
        theta = 0
        while theta <= np.pi:
            for r in range(k * (i-1), k * i + 1):
                y, x = polarToKart(shape, r, theta)
                if 0 <= x < w and 0 <= y < h:
                    feature_vector[i-1] += magnitude_spectrum[int(y), int(x)]
            theta += np.pi / (sampling_steps - 1)

    return feature_vector


def extractFanFeatures(magnitude_spectrum, k, sampling_steps) -> np.ndarray:
    """
    Follow the approach to extract Fan features
    Assume all rays have same length regardless of angle.
    Their length should be set by the smallest feasible ray.
    :param magnitude_spectrum:
    :param k: number of fans-like features to extract
    :param sampling_steps: number of rays to sample from in one fan-like area --> theta/sampling rate
    :return: feature vector of length k
    """
    shape = magnitude_spectrum.shape
    feature_vector = np.zeros(k)

    for i in range(k):
        theta_start = i * np.pi / k
        theta_end = (i + 1) * np.pi / k
        theta_values = np.linspace(theta_start, theta_end, sampling_steps)

        for theta in theta_values:
            for r in range(0, 36):
                y, x = polarToKart(shape, r, theta)
                y = int(np.clip(y, 0, shape[0] - 1))
                x = int(np.clip(x, 0, shape[1] - 1))
                feature_vector[i] += magnitude_spectrum[y, x]

    return feature_vector


def calcuateFourierParameters(img, k, sampling_steps) -> (np.ndarray, np.ndarray):
    '''
    Extract Features in Fourier space following the paper.
    :param img: input image
    :param k: number of features to extract from each method
    :param sampling_steps: number of samples to accumulate for each feature
    :return: R, T feature vectors of length k
    '''
    magnitude_spectrum = calculateMagnitudeSpectrum(img)
    R = extractRingFeatures(magnitude_spectrum, k, sampling_steps)
    T = extractFanFeatures(magnitude_spectrum, k, sampling_steps)
    return R, T
