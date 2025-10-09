# -*- coding: utf-8 -*-
# %% Preamble
"""
shmReferenceSignals.py
----------------------

Generates reference signals as for calibrating and testing the validity of
ECMA-418-2 implementation (Sottek Hearing Model)

Requirements
------------
numpy
scipy
matplotlib

Ownership and Quality Assurance
-------------------------------
Author: Mike JB Lotinga (m.j.lotinga@edu.salford.ac.uk)
Institution: University of Salford

Date created: 29/05/2023
Date last modified: 02/10/2025
Python version: 3.11

Copyright statement: This code has been devloped during work undertaken within
the RefMap project (www.refmap.eu), based on the RefMap code repository
(https://github.com/acoustics-code-salford/refmap-psychoacoustics),
and as such is subject to copyleft licensing as detailed in the code repository
(https://github.com/acoustics-code-salford/sottek-hearing-model).

The code has been modified to amend imports or omit unnecessary lines.

As per the licensing information, please be aware that this code is WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.

"""

# %% Import block
import numpy as np
from sottek_hearing_model.shmSubs import shmRMS


# %% shmGenerateRefSignals
def shmGenerateRefSignals(signalDuration):

    # %% Input check
    try:
        if signalDuration <= 0.3:
            raise ValueError("Input signalDuration must be longer than 0.3 s.")
    except TypeError:
        raise TypeError("Input signalDuration must be a single numerical value.")

    # %% Input parameters
    fs = 48_000  # Hz
    dt = 1/fs  # s
    T = signalDuration  # s
    n = int(T*fs)  # number of samples
    t = np.linspace(0, T - dt, n)  # time vector
    f_tone = 1000  # sinusoid frequency Hz
    f_mod70 = 70  # modulation frequency for roughness Hz
    f_mod4 = 4  # modulation frequency for fluctuation strength Hz
    # Amplitude of 1 kHz tone at 40 dB Lp
    A_tone = np.sqrt(2)*2e-5*10**(40/20)

    # %% Generate signals
    # reference sinusoid for loudness and tonality

    sine_1kHz_40dB = A_tone*np.sin(2*np.pi*f_tone*t)

    # reference modulated sinusoid for roughness

    sine_70Hz_mod = np.sin(2*np.pi*f_mod70*t - np.pi/2)
    sine_1kHz_70Hz_60dB = (1 + sine_70Hz_mod)*np.sin(2*np.pi*f_tone*t)
    A_adjust = shmRMS(sine_1kHz_70Hz_60dB)/0.02
    sine_1kHz_70Hz_60dB = sine_1kHz_70Hz_60dB/A_adjust

    # reference modulated sinusoid for fluctuation strength

    sine_4Hz_mod = np.sin(2*np.pi*f_mod4*t - np.pi/2)
    sine_1kHz_4Hz_60dB = (1 + sine_4Hz_mod)*np.sin(2*np.pi*f_tone*t)
    A_adjust = shmRMS(sine_1kHz_4Hz_60dB)/0.02
    sine_1kHz_4Hz_60dB = sine_1kHz_4Hz_60dB/A_adjust

    return (sine_1kHz_40dB, sine_1kHz_70Hz_60dB, sine_1kHz_4Hz_60dB)

# end of function
