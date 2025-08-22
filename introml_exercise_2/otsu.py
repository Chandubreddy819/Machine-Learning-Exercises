import numpy as np
#
# NO OTHER IMPORTS ALLOWED
#

def create_greyscale_histogram(img):
    '''
    returns a histogram of the given image
    :param img: 2D image in greyscale [0, 255]
    :return: np.ndarray (256,) with absolute counts for each possible pixel value
    '''
    # TODO
    flatten_image = img.flatten()
    hist, bin_edges = np.histogram(flatten_image, 256, (0, 255))
    return hist


def binarize_threshold(img, t):
    '''
    binarize an image with a given threshold
    :param img: 2D image as ndarray
    :param t: int threshold value
    :return: np.ndarray binarized image with values in {0, 255}
    '''
    # TODO
    binary_image = np.zeros_like(img, dtype=np.uint8)
    binary_image[img > t] = 255
    return binary_image


def p_helper(hist, theta: int):
    '''
    Compute p0 and p1 using the histogram and the current theta,
    do not take care of border cases in here
    :param hist:
    :param theta: current theta
    :return: p0, p1
    '''
    cdf = np.cumsum(hist)
    p0 = cdf[theta]
    p1 = 1 - p0
    return p0, p1


def mu_helper(hist, theta, p0, p1):
    '''
    Compute mu0 and m1
    :param hist: histogram
    :param theta: current theta
    :param p0:
    :param p1:
    :return: mu0, mu1
    '''
    cumulative_mean = np.cumsum(hist * np.arange(hist.shape[0]))
    global_mean = cumulative_mean[-1]
    mu0 = cumulative_mean[theta] / p0
    mu1 = (global_mean - cumulative_mean[theta]) / p1
    return mu0, mu1


def calculate_otsu_threshold(hist):
    '''
    calculates theta according to otsus method

    :param hist: 1D array
    :return: threshold (int)
    '''
    # TODO initialize all needed variables
    optimum_theta = 0
    max_variance = 0

    # TODO change the histogram, so that it visualizes the probability distribution of the pixels
    # --> sum(hist) = 1
    normalized_hist = hist.astype(float) / np.sum(hist)

    # TODO loop through all possible thetas

    for i in range(256):
        # TODO compute p0 and p1 using the helper function
        p0, p1 = p_helper(normalized_hist, i)
        if p0 == 0 or p1 == 0:
            continue

        # TODO compute mu and m1 using the helper function
        mu0, mu1 = mu_helper(normalized_hist, i, p0, p1)

        global_mean = p0*mu0+p1*mu1

        # TODO compute variance

        inter_class_variance = p0*((mu0-global_mean)**2) + p1*((mu1-global_mean)**2)

        # TODO update the threshold
        if inter_class_variance > max_variance:
            max_variance = inter_class_variance
            optimum_theta = i

    return optimum_theta


def otsu(img):
    '''
    calculates a binarized image using the otsu method.
    Hint: reuse the other methods
    :param image: grayscale image values in range [0, 255]
    :return: np.ndarray binarized image with values {0, 255}
    '''
    # TODO
    histogram = create_greyscale_histogram(img)
    threshold = calculate_otsu_threshold(histogram)
    return binarize_threshold(img, threshold)
