import matplotlib.pyplot as plt
import numpy as np

from chirp import createChirpSignal
from decomposition import createTriangleSignal, createSquareSignal, createSawtoothSignal

# TODO: Test the functions imported in lines 1 and 2 of this file.

if __name__ == '__main__':
    # Exercise-1
    sampling_rate = 200
    duration = 1
    freq_from = 1
    freq_to = 10
    t = np.linspace(freq_from, freq_to, int(sampling_rate * duration))
    linear_signal = createChirpSignal(sampling_rate, duration, freq_from, freq_to, True)
    plt.plot(t, linear_signal)
    plt.show()

    exponential_signal = createChirpSignal(sampling_rate, duration, freq_from, freq_to, False)
    plt.plot(t, exponential_signal)
    plt.show()
    #
    # Exercise-2
    time = np.linspace(0, 1, 200)

    triangular_signal = createTriangleSignal(200, 2, 1000)
    square_signal = createSquareSignal(200, 2, 10000)
    sawtooth_signal = createSawtoothSignal(200, 2, 10000, 1)
    figure, axis = plt.subplots(3)
    axis[0].plot(time, triangular_signal)
    axis[0].set_title('Triangular signal')
    axis[1].plot(time, square_signal)
    axis[1].set_title('Square signal')
    axis[2].plot(time, sawtooth_signal)
    axis[2].set_title('Sawtooth signal')
    plt.show()


