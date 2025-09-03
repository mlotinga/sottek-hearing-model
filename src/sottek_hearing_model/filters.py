# -*- coding: utf-8 -*-
"""
filterFuncs.py
------------

Filter functions:

- Frequency weightings in the time domain

Requirements
------------
numpy
scipy

Ownership and Quality Assurance
-------------------------------
Author: Mike JB Lotinga (m.j.lotinga@edu.salford.ac.uk)
Institution: University of Salford

Date created: 03/09/2025
Date last modified: 03/09/2025
Python version: 3.11

Copyright statements: This file is based on code developed within the refmap-psychoacoustics
repository (https://github.com/acoustics-code-salford/refmap-psychoacoustics),
and as such is subject to copyleft licensing as detailed in the code repository
(https://github.com/acoustics-code-salford/refmap-psychoacoustics).

The code has been modified to omit unnecessary lines.

"""

import numpy as np
from scipy.signal import (bilinear, freqz, lfilter, lfilter_zi,
                          resample_poly)
from src.py.dsp.noct import noctf
from math import gcd


def A_weight_T(x, fs, axis=0, check=False):
    """
    Return time-domain-filtered signal according to standard sound frequency
    weighting 'A'.

    Implements IIR filter via bilinear transform. Includes pre-warping of
    analogue design frequencies to compensate for bilinear transform frequency
    distortion.

    Upsamples signals to 36kHz (if necessary)
    before processing, to ensure compliance with IEC 61672-1 class 1 acceptance
    limits.
    
    Resampling frequency and pre-warping defined according to [1].

    Parameters
    ----------
    x : 1D or 2D array
        contains the time signals to be weighted (filtered)
    fs : number
         the sampling frequency of the signals to be processed
    axis : integer
           the signal array axis along which to apply the filter
    check : boolean
            flag to check the filter against IEC acceptance limits

    Returns
    -------
    y : 1D or 2D array
        contains the weighted (filtered) time signals
    f : 1D array
        contains the frequencies (Hz) of the filter frequency response function
    H : 1D array
        contains the complex frequency response function values for each f

    Requirements
    ------------
    numpy
    scipy

    Assumptions
    -----------

    References
    ----------
    [1] Rimell, AN et al, 2015 - Design of digital filters for frequency
        weightings (A and C) required for risk assessments of workers exposed
        to noise. Industrial Health, 53, 21-27.

    """

    if fs < 36000:
        # upsampled sampling frequency
        fsu = 36000
        up = int(fsu/gcd(fsu, fs))
        down = int(fs/gcd(fsu, fs))
    else:
        fsu = fs
    dtu = 1/fsu

    G_Aw = 10**(2/20)
    
    f1 = np.sqrt((-1/(1 - np.sqrt(0.5))*(1e3**2 + 1e3*10**7.8/1e3**2
                                         - np.sqrt(0.5)*(1e3 + 10**7.8))
                  - np.sqrt((1/(1 - np.sqrt(0.5))*(1e3**2
                                                   + 1e3*10**7.8/1e3**2
                                                   - np.sqrt(0.5)*(1e3 + 10**7.8)))**2
                            - 4*1e3*10**7.8)) / 2)
    
    f4 = np.sqrt((-1/(1 - np.sqrt(0.5))*(1e3**2 + 1e3*10**7.8/1e3**2
                                         - np.sqrt(0.5)*(1e3 + 10**7.8))
                  + np.sqrt((1/(1 - np.sqrt(0.5))*(1e3**2
                                                   + 1e3*10**7.8/1e3**2
                                                   - np.sqrt(0.5)*(1e3 + 10**7.8)))**2
                            - 4*1e3*10**7.8)) / 2)
    
    f2 = 10**2.45*((3 - np.sqrt(5))/2)
    f3 = 10**2.45*((3 + np.sqrt(5))/2)

    w1 = 2*np.pi*f1
    w1w = 2/dtu*np.tan(w1*dtu/2)  # pre-warped frequency
    w4 = 2*np.pi*f4
    w4w = 2/dtu*np.tan(w4*dtu/2)  # pre-warped frequency
    w2 = 2*np.pi*f2
    w2w = 2/dtu*np.tan(w2*dtu/2)  # pre-warped frequency
    w3 = 2*np.pi*f3
    w3w = 2/dtu*np.tan(w3*dtu/2)  # pre-warped frequency

    B = np.array([G_Aw*w4w**2, 0, 0, 0, 0])
    A1 = [1.0, 2*w4w, (w4w)**2]
    A2 = [1.0, 2*w1w, (w1w)**2]
    A3 = [1.0, w3w]
    A4 = [1.0, w2w]
    A = np.convolve(np.convolve(np.convolve(A1, A2), A3), A4)

    b, a = bilinear(B, A, fsu)

    # determine filter initial conditions
    if len(x.shape) == 1 or axis == 1:
        zi = lfilter_zi(b, a)
    else:
        zi = lfilter_zi(b, a)[:, None]

    if check:
        # Check filter spec against acceptance limits
        f, H = freqz(b, a, worN=15000, fs=fsu)
        f = f[1:]  # discards zero-frequency
        H = H[1:]  # discards zero-frequency

        # IEC 61672-1:2013 acceptance limits (class 1)
        fm, _, _ = noctf(10, 20000, 3)
        Lm = np.array([-70.4, -63.4, -56.7, -50.5, -44.7, -39.4, -34.6, -30.2,
                      -26.2, -22.5, -19.1, -16.1, -13.4, -10.9, -8.6, -6.6, -4.8,
                      -3.2, -1.9, -0.8, 0.0, 0.6, 1.0, 1.2, 1.3, 1.2, 1.0, 0.5,
                      -0.1, -1.1, -2.5, -4.3, -6.6, -9.3])
        Ll = np.array([-9999.0, -9999.0, -4.0, -2.0, -1.5, -1.5, -1, -1, -1, -1,
                      -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -0.7, -1, -1, -1,
                      -1, -1, -1, -1.5, -2, -2.5, -3, -5, -16, -9999]) + Lm
        Lu = np.array([3.0, 2.5, 2.0, 2.0, 2.0, 1.5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                      1, 1, 1, 1, 0.7, 1, 1, 1, 1, 1, 1, 1.5, 1.5, 1.5, 2, 2, 2.5,
                      3]) + Lm
        Lmf = np.interp(f, fm, Lm)  # Interpolate Lm onto f axis
        Luf = np.interp(f, fm, Lu)  # Interpolate Lu onto f axis
        Llf = np.interp(f, fm, Ll)  # Interpolate Lu onto f axis

    # Filter data on upsampled version

    if len(x.shape) == 1:
        # upsample signal (if necessary)
        if fsu > fs:
            x = resample_poly(x, up, down, padtype='line')
        # filter signal
        y, _ = lfilter(b, a, x, zi=zi*x[0])
        # if upsampled, downsample to original fs
        if fsu > fs:
            y = resample_poly(y, down, up, padtype='line')

    elif len(x.shape) == 2:
        # upsample signal (if necessary)
        if fsu > fs:
            x = resample_poly(x, up, down, axis=axis, padtype='line')
        # filter signal
        y, _ = lfilter(b, a, x, axis=axis, zi=zi*np.take(x, [0], axis=axis))
        # if upsampled, downsample to original fs
        if fsu > fs:
            y = resample_poly(y, down, up, axis, padtype='line')

    else:
        raise TypeError("\nInput must be 1d or 2d array")

    return y