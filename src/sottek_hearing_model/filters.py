# -*- coding: utf-8 -*-
"""
filters.py
----------

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

Date created: 22/01/2024
Date last modified: 20/10/2025
Python version: 3.11

Copyright statements: This file is based on code developed within the refmap-psychoacoustics
repository (https://github.com/acoustics-code-salford/refmap-psychoacoustics),
and as such is subject to copyleft licensing as detailed in the code repository
(https://github.com/acoustics-code-salford/sottek-hearing-model).

The code has been modified to omit unnecessary lines.

As per the licensing information, please be aware that this code is WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.

"""

import numpy as np
from scipy.signal import (bilinear, lfilter, lfilter_zi, resample_poly)
from math import gcd


def weight_A_t(x, fs, axis=0):
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

    Returns
    -------
    y : 1D or 2D array
        contains the weighted (filtered) time signals

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