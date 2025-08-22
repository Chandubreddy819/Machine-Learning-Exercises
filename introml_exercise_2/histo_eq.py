# Implement the histogram equalization in this file
import cv2
import numpy as np


def calculate_histogram(image):
    flatten_image = image.flatten()
    histogram = [0] * 256
    for pixel in flatten_image:
        histogram[pixel] += 1
    return histogram


def sum_of_elements(histogram, end_limit):
    total_sum = 0
    for element in histogram[:end_limit]:
        total_sum += element
    return total_sum


def normalize(histogram):
    normalized_histogram = [0] * len(histogram)
    total_sum_of_elements = sum_of_elements(histogram, len(histogram))
    for element in range(len(histogram)):
        normalized_histogram[element] = histogram[element] / total_sum_of_elements
    return normalized_histogram


def calculate_cumulative_distribution_function(histogram):
    cumulative_sum = 0
    cdf = [0] * len(histogram)
    normalized_histogram = normalize(histogram)
    for i in range(len(histogram)):
        cumulative_sum += normalized_histogram[i]
        cdf[i] = cumulative_sum
    return cdf


def histogram_equalization(histogram, cdf):
    pixel_new = [0] * len(histogram)
    c_min = next(x for x in cdf if x > 0)

    for i in range(len(histogram)):
        new_pixel_value = int(((cdf[i] - c_min) / (1 - c_min)) * 255)
        pixel_new[i] = new_pixel_value

    return pixel_new


if __name__ == '__main__':
    img = cv2.imread('hello.png', cv2.IMREAD_GRAYSCALE)

    hist = calculate_histogram(img)
    sum_of_first_90_elements = sum_of_elements(hist, 90)
    if sum_of_first_90_elements == 249:
        print('The sum of first 90 elements is 249')

    cumulative_distribution_function = calculate_cumulative_distribution_function(hist)
    sum_of_first_90_elements_in_cdf = sum_of_elements(cumulative_distribution_function, 90)
    if sum_of_first_90_elements_in_cdf == 0.001974977:
        print('The sum of first 90 elements in cumulative distribution function is 0.001974977')

    equalized_pixels = histogram_equalization(hist, cumulative_distribution_function)
    equalized_image = np.interp(img, range(256), equalized_pixels).astype(np.uint8)
    cv2.imwrite('kitty.png', equalized_image)
    cv2.imshow('original', img)
    cv2.imshow('Equalised kitty image', equalized_image)
    cv2.waitKey(0)
