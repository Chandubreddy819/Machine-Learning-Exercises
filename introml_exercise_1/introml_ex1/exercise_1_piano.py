import os

import numpy as np


def load_sample(filename, duration=4 * 44100, offset=44100 // 10):
    array = np.load(filename)
    abs_data = np.abs(array)
    max_peak = np.argmax(abs_data)
    starting_value = max_peak + offset
    end_value = min(starting_value + duration, len(array))
    return array[starting_value:end_value]


def compute_frequency(signal, min_freq=20):
    dft_signal = np.fft.fft(signal)
    magnitude = np.abs(dft_signal)
    frequencies = np.fft.fftfreq(len(signal), d=1 / 44100)
    valid_indexes = np.where(frequencies > min_freq)[0][0]
    max_index = np.argmax(magnitude[valid_indexes:]) + valid_indexes
    return frequencies[max_index]


if __name__ == '__main__':
    sounds_dir = 'sounds/'
    files = ['Piano.ff.A2.npy', 'Piano.ff.A3.npy', 'Piano.ff.A4.npy',
             'Piano.ff.A5.npy', 'Piano.ff.A6.npy', 'Piano.ff.A7.npy',
             'Piano.ff.XX.npy']

    source = [os.path.join(sounds_dir, x) for x in files]

    # Expected frequencies for the notes A2, A3, A4, A5, A6, A7 (in Hz)
    test_frequencies = [110, 220, 440, 880, 1760, 3520]

    # Compute the frequencies of all notes and compare them to expected values
    computed_frequencies = []
    for sound_file in source:
        sample = load_sample(sound_file)
        peak_frequency = compute_frequency(sample)
        computed_frequencies.append(peak_frequency)

    # Find the mysterious note
    mysterious_note_frequency = computed_frequencies[-1]
    print(f"file XX Note Frequency: {mysterious_note_frequency:.2f} Hz")

    # Compare computed frequencies to expected frequencies
    for i, expected_frequency in enumerate(test_frequencies):
        print(
            f"Expected Frequency for A{i + 2}: {expected_frequency} Hz, Computed Frequency: {computed_frequencies[i]:.2f} Hz")

    # Identify the mysterious note
    notes = ['A2', 'A3', 'A4', 'A5', 'A6', 'A7']
    closest_note_index = np.argmin([abs(mysterious_note_frequency - freq) for freq in test_frequencies])
    closest_note = notes[closest_note_index]
    print(f"The mysterious note is closest to: {closest_note}")

# This will be helpful:
# https://en.wikipedia.org/wiki/Piano_key_frequencies
