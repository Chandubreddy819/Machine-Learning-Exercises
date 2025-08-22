'''
Created on 04.10.2016

@author: Daniel Stromer
@modified by Charly, Christian, Max (23.12.2020)
'''
import numpy as np
import matplotlib.pyplot as plt
# do not import more modules!


def calculate_R_Distance(Rx, Ry):
    '''
    Calculate similarity of Ring features using Euclidean distance.
    :param Rx: Ring features of Person X (numpy array)
    :param Ry: Ring features of Person Y (numpy array)
    :return: Similarity index (float)
    '''
    n = len(Rx)
    distance = (1 / n) * np.sum(np.abs(Rx - Ry))
    return distance


def calculate_Theta_Distance(Thetax, Thetay):
    '''
    Calculate similarity of Fan features using Cosine similarity.
    :param Thetax: Fan features of Person X (numpy array)
    :param Thetay: Fan features of Person Y (numpy array)
    :return: Similarity index (float)
    '''
    mean_theta_x = np.mean(Thetax)
    mean_theta_y = np.mean(Thetay)

    lxx = np.sum((Thetax - mean_theta_x) ** 2)
    lyy = np.sum((Thetay - mean_theta_y) ** 2)
    lxy = np.sum((Thetax - mean_theta_x) * (Thetay - mean_theta_y))

    d_theta_xy = (1 - (lxy ** 2) / (lxx * lyy)) * 100
    return d_theta_xy
