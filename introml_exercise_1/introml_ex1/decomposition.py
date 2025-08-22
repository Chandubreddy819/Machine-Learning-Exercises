import numpy as np


def createTriangleSignal(samples: int, frequency: int, k_max: int):
    t = np.linspace(0, 1, samples)
    y = np.zeros_like(t)

    for k in range(0, k_max):
        sign = (-1) ** k
        coefficient = (8 / (np.pi ** 2))
        phase = ((2 * np.pi) * ((2 * k) + 1) * frequency * t)
        y += (np.sin(phase) / (((2 * k) + 1) ** 2)) * coefficient * sign

    return y


def createSquareSignal(samples: int, frequency: int, k_max: int):
    t = np.linspace(0, 1, samples)
    y = np.zeros_like(t)

    for k in range(1, k_max):
        coefficient = 4 / np.pi
        phase = ((2 * np.pi) * ((2 * k) - 1) * frequency * t)
        y += coefficient * (np.sin(phase) / ((2 * k) - 1))

    return y


def createSawtoothSignal(samples: int, frequency: int, k_max: int, amplitude: int):
    # returns the signal as 1D-array (np.ndarray)
    t = np.linspace(0, 1, samples)
    y = np.zeros_like(t)
    a = amplitude

    for k in range(1, k_max):
        coefficient = (a / np.pi)
        phase = (2 * np.pi * k * frequency * t)
        y += coefficient * (np.sin(phase) / k)

    y = a / 2 - y
    return y
