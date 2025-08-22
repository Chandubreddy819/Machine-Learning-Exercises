import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
from convo import make_kernel
from PIL import Image


#
# NO MORE MODULES ALLOWED
#


def gaussFilter(img_in, ksize, sigma):
    """
    filter the image with a gauss kernel
    :param img_in: 2D greyscale image (np.ndarray)
    :param ksize: kernel size (int)
    :param sigma: sigma (float)
    :return: (kernel, filtered) kernel and gaussian filtered image (both np.ndarray)
    """
    # TODO
    kernel = make_kernel(ksize, sigma)
    convolved_image = convolve(img_in, kernel)
    gaussian_filtered_image = convolved_image.astype(np.int32)
    return kernel, gaussian_filtered_image


def sobel(img_in):
    """
    applies the sobel filters to the input image
    Watch out! scipy.ndimage.convolve flips the kernel...

    :param img_in: input image (np.ndarray)
    :return: gx, gy - sobel filtered images in x- and y-direction (np.ndarray, np.ndarray)
    """
    # TODO
    sobel_x_kernel = np.array([[-1, 0, 1],
                               [-2, 0, 2],
                               [-1, 0, 1]])

    sobel_y_kernel = np.array([[-1, -2, -1],
                               [0, 0, 0],
                               [1, 2, 1]])

    gx = convolve(img_in, sobel_x_kernel).astype(np.int32)
    gy = convolve(img_in, np.flipud(sobel_y_kernel)).astype(np.int32)
    return gx, gy


def gradientAndDirection(gx, gy):
    """
    calculates the gradient magnitude and direction images
    :param gx: sobel filtered image in x direction (np.ndarray)
    :param gy: sobel filtered image in x direction (np.ndarray)
    :return: g, theta (np.ndarray, np.ndarray)
    """
    # TODO
    g = np.sqrt(gx ** 2 + gy ** 2).astype(np.int32)
    theta = np.arctan2(gy, gx)
    return g, theta


def convertAngle(angle):
    """
    Compute nearest matching angle
    :param angle: in radians
    :return: nearest match of {0, 45, 90, 135}
    """
    theta_degrees = np.degrees(angle)

    # Normalize angle to be in the range [0, 180)
    theta_normalized = np.mod(theta_degrees, 180)

    if 157.5 <= theta_normalized < 180:
        return 0
    elif 22.5 <= theta_normalized < 67.5:
        return 45
    elif 67.5 <= theta_normalized < 112.5:
        return 90
    elif 112.5 <= theta_normalized < 157.5:
        return 135
    else:
        return 0


def maxSuppress(g, theta):
    """
    calculate maximum suppression
    :param g: gradient magnitude image (np.ndarray)
    :param theta: gradient direction image (np.ndarray)
    :return: max_sup: maximum suppression result (np.ndarray)
    """
    max_sup = np.zeros_like(g, dtype=np.int32)

    # Get image shape
    height, width = g.shape

    # Iterate through each pixel in the image
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            # Determine gradient direction

            angle = convertAngle(theta[y, x])

            # Check local maximum based on gradient direction
            if angle == 0:
                if g[y, x] >= g[y, x + 1] and g[y, x] >= g[y, x - 1]:
                    max_sup[y, x] = g[y, x]
            elif angle == 45:
                if g[y, x] >= g[y + 1, x - 1] and g[y, x] >= g[y - 1, x + 1]:
                    max_sup[y, x] = g[y, x]
            elif angle == 90:
                if g[y, x] >= g[y + 1, x] and g[y, x] >= g[y - 1, x]:
                    max_sup[y, x] = g[y, x]
            elif angle == 135:
                if g[y, x] >= g[y + 1, x + 1] and g[y, x] >= g[y - 1, x - 1]:
                    max_sup[y, x] = g[y, x]

    return max_sup


def hysteris(max_sup, t_low, t_high):
    """
    Calculate hysteresis thresholding.

    :param max_sup: 2D image (np.ndarray)
    :param t_low: Lower threshold value (int)
    :param t_high: Upper threshold value (int)
    :return: Hysteresis thresholded image (np.ndarray)
    """
    # Classify each pixel based on thresholds
    thresh_img = np.zeros_like(max_sup, dtype=np.uint8)
    thresh_img[max_sup > t_low] = 1
    thresh_img[max_sup > t_high] = 2

    # Walk through the classified image
    height, width = thresh_img.shape
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            if thresh_img[y, x] == 2:  # If pixel is greater than t_high
                # Set the pixel to 255
                max_sup[y, x] = 255
                # Set neighboring pixels greater than t_low to 255
                for ny in range(y - 1, y + 2):
                    for nx in range(x - 1, x + 2):
                        if thresh_img[ny, nx] == 1:
                            max_sup[ny, nx] = 255

    return max_sup


def canny(img):
    # gaussian
    kernel, gauss = gaussFilter(img, 3, 1)

    # sobel
    gx, gy = sobel(gauss)

    # plotting
    plt.subplot(1, 2, 1)
    plt.imshow(gx, 'gray')
    plt.title('gx')
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.imshow(gy, 'gray')
    plt.title('gy')
    plt.colorbar()
    plt.show()

    # gradient directions
    g, theta = gradientAndDirection(gx, gy)

    # plotting
    plt.subplot(1, 2, 1)
    plt.imshow(g, 'gray')
    plt.title('gradient magnitude')
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.imshow(theta)
    plt.title('theta')
    plt.colorbar()
    plt.show()

    # maximum suppression
    maxS_img = maxSuppress(g, theta)

    # plotting
    plt.imshow(maxS_img, 'gray')
    plt.show()

    result = hysteris(maxS_img, 50, 75)

    return result


if __name__ == '__main__':
    img = cv2.imread('contrast.jpg', cv2.IMREAD_GRAYSCALE)
    canny(img)
