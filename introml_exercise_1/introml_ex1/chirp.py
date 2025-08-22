import numpy as np


def createChirpSignal(samplingrate: int, duration: int, freqfrom: int, freqto: int, linear: bool):
    # returns the chirp signal as list or 1D-array
    num_of_samples = int(samplingrate * duration)
    t = np.linspace(0, duration, num_of_samples)

    if linear:
        return linear_chirp_signal(duration, freqfrom, freqto, t)
    return exponential_chirp_signal(duration, freqfrom, freqto, t)


def linear_chirp_signal(duration: int, freqfrom: int, freqto: int, t: [int]):
    chirp_rate = (freqto - freqfrom) / duration
    phase = (chirp_rate * t * t) / 2 + (freqfrom * t)
    signal = np.sin(2 * np.pi * phase)
    return signal


def exponential_chirp_signal(duration: int, freqfrom: int, freqto: int, t: [int]):
    chirp_rate = (freqto / freqfrom) ** (1 / duration)
    phase = (2 * np.pi * freqfrom) * (chirp_rate ** t - 1) / np.log(chirp_rate)
    signal = np.sin(phase)
    return signal
