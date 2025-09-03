# -*- coding: utf-8 -*-
# %% Preamble
"""
acousticSHMSubs.py
--------------

Acoustic signal analysis subfunctions for implementing the Sottek Hearing
Model, as defined in the ECMA-418-2 standard (currently 2025).

Requirements
------------
numpy
scipy
matplotlib

Ownership and Quality Assurance
-------------------------------
Author: Mike JB Lotinga (m.j.lotinga@edu.salford.ac.uk)
Institution: University of Salford

Date created: 27/10/2023
Date last modified: 23/07/2025
Python version: 3.11

Copyright statement: This file and code is part of work undertaken within
the RefMap project (www.refmap.eu), and is subject to licence as detailed
in the code repository
(https://github.com/acoustics-code-salford/refmap-psychoacoustics)

As per the licensing information, please be aware that this code is WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.

Parts of this code were developed from an original MATLAB file
'SottekTonality.m' authored by Matt Torjussen (14/02/2022), based on
implementing ECMA-418-2:2020. The original code has been reused and translated
here with permission.

Checked by:
Date last checked:

"""

# %% Imports and parameter setup
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
from scipy.signal import (freqz, lfilter, resample_poly, sosfilt, sosfreqz)
from scipy.special import (comb)
from math import gcd

# set plot parameters
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = 'Arial'
mpl.rcParams.update({'font.size': 14})
mpl.rcParams['mathtext.fontset'] = 'stixsans'
mpl.rcParams['figure.autolayout'] = True


# %% shmAuditoryFiltBank
def shmAuditoryFiltBank(signal, outPlot=False):
    """Function signalFiltered = shmAuditoryFiltBank(signal, outplot)

    Returns a set of signals, bandpass filtered for the inner ear response
    in each half-Bark critical band rate scale width, according to
    ECMA-418-2:2025 (the Sottek Hearing Model) for an input calibrated
    single (sound pressure) time-series signal.

    Inputs
    ------
    signal : 1D or 2D array
             the input signal as single audio (sound pressure) signal/s,
             arranged as [time(, chans)]

    outPlot : Boolean true/false (default: false)
              flag indicating whether to generate a figure a frequency and
              phase response figure for the filter bank

    Returns
    -------
    signalFiltered : 2D of 3D array
                     the filtered signals, arranged as [time, bands(, chans)]

    Assumptions
    -----------
    The input signal is a numpy array oriented with time on axis 0, ie, the
    filter operation is applied along axis 0.
    The input signal must be sampled at 48 kHz.

    """

    # %% Define constants

    sampleRate48k = 48e3  # Signal sample rate prescribed to be 48kHz (to be used for resampling), Section 5.1.1 ECMA-418-2:2025
    deltaFreq0 = 81.9289  # defined in Section 5.1.4.1 ECMA-418-2:2025
    c = 0.1618  # Half-Bark band centre-frequency demoninator constant defined in Section 5.1.4.1 ECMA-418-2:2025

    dz = 0.5  # critical band resolution
    halfBark = np.arange(0.5, 27, dz)  # half-overlapping critical band rate scale
    # Section 5.1.4.1 Equation 9 ECMA-418-2:2025
    bandCentreFreqs = (deltaFreq0/c)*np.sinh(c*halfBark)
    # Section 5.1.4.1 Equation 10 ECMA-418-2:2025
    dfz = np.sqrt(deltaFreq0**2 + (c*bandCentreFreqs)**2)

    # %% Signal processing

    # Apply auditory filter bank
    # --------------------------
    # Filter equalised signal using 53 1/2Bark ERB filters according to
    # Section 5.1.4.2 ECMA-418-2:2025

    k = 5  # filter order = 5, footnote 5 ECMA-418-2:2025
    # filter coefficients for Section 5.1.4.2 Equation 15 ECMA-418-2:2025
    e_i = [0, 1, 11, 11, 1]

    if signal.ndim == 2:
        signalFiltered = np.zeros((signal.shape[0], len(halfBark),
                                   signal.shape[1]))
    elif signal.ndim == 1:
        signalFiltered = np.zeros((signal.size, len(halfBark)))
    else:
        raise ValueError("Input signal must have no more than two dimensions.")

    for zBand in range(53):
        # Section 5.1.4.1 Equation 8 ECMA-418-2:2025
        tau = (1/(2**(2*k - 1)))*comb(2*k - 2, k - 1)*(1/dfz[zBand])

        d = np.exp(-1./(sampleRate48k*tau))  # Section 5.1.4.1 ECMA-418-2:2025

        # Band-pass modifier Section 5.1.4.2 Equation 16/17 ECMA-418-2:2025
        bp = np.exp((1j*2*np.pi*bandCentreFreqs[zBand]*np.arange(0,
                                                                 k + 2))/sampleRate48k)

        # Feed-backward coefficients, Section 5.1.4.2 Equation 14 ECMA-418-2:2025
        m_a = range(1, k + 1)
        a_m = np.append(1, ((-d)**m_a)*comb(k, m_a))*bp[0:k + 1]

        # Feed-forward coefficients, Section 5.1.4.2 Equation 15 ECMA-418-2:2025
        m_b = range(0, k)
        i = range(1, k)
        b_m = ((((1 - d)**k)/np.sum(e_i[1:]*(d**i)))*(d**m_b)*e_i)*bp[0:k]

        # Recursive filter Section 5.1.4.2 Equation 13 ECMA-418-2:2025
        # Note, the results are complex so 2x the real-valued band-pass signal
        # is required.
        if signalFiltered.ndim == 3:
            signalFiltered[:, zBand, :] = 2*np.real(lfilter(b_m, a_m, signal,
                                                            axis=0))
        else:
            signalFiltered[:, zBand] = 2*np.real(lfilter(b_m, a_m, signal,
                                                         axis=0))

        # Plot figures

        if outPlot:
            f, H = freqz(b_m, a=a_m, worN=int(10e3), whole=True,
                         fs=sampleRate48k)
            phir = np.angle(H)
            phirUnwrap = np.unwrap(p=phir, discont=np.pi, axis=0)
            phiUnwrap = phirUnwrap/np.pi*180
            # Plot frequency and phase response for filter
            if zBand == 0:
                fig, axs = plt.subplots(nrows=2, ncols=1)
                ax1 = axs[0]
                ax2 = axs[1]

            ax1.semilogx(f, 20*np.log10(abs(H)))
            ax2.semilogx(f, phiUnwrap)

            if zBand == 52:
                ax1.set(xlim=[20, 20e3], ylim=[-100, 0],
                        xticks=[31.5, 63, 125, 250, 500, 1e3, 2e3, 4e3, 8e3,
                                16e3],
                        xticklabels=["31.5", "63", "125", "250", "500", "1k",
                                     "2k", "4k", "8k", "16k"],
                        xlabel="Frequency, Hz", ylabel="$H$, dB")
                ax1.minorticks_off()
                ax1.grid(alpha=0.15, linestyle='--')

                ax2.set(xlim=[20, 20e3],
                        xticks=[31.5, 63, 125, 250, 500, 1e3, 2e3, 4e3, 8e3,
                                16e3],
                        xticklabels=["31.5", "63", "125", "250", "500", "1k",
                                     "2k", "4k", "8k", "16k"],
                        xlabel="Frequency, Hz",
                        ylabel=r"Phase angle$^\degree$")
                ax2.minorticks_off()
                ax2.grid(alpha=0.15, linestyle='--')

        # end of if branch for outPlot
    # end of for loop over bands

    return signalFiltered  # end of shmAuditoryFiltBank function


# %% shmBasisLoudness
def shmBasisLoudness(signalSegmented, bandCentreFreq=None):
    """Function shmBandBasisLoudness(signalSegmented, bandCentreFreq)

    Returns rectified input and basis loudness in specified half-Bark
    critical band according to ECMA-418-2:2025 (the Sottek Hearing Model)
    for an input band-limited signal, segmented into processing blocks

    Inputs
    ------
    signalSegmented : 2D or 3D array
                      input band-limited segmented signal(s)

    bandCentreFreq : double (optional, default: None)
                     half-Bark critical band centre frequency - if None, all
                     bands are assumed to be present in the input segmented
                     signal matrix

    Returns
    -------
    signalRectSeg : 2D or 3D array
                    rectified band-limited segmented signal

    basisLoudness : 2D or 3D array
                    basis loudness in each block

    blockRMS : 1D or 2D array
               RMS for each block

    Assumptions
    -----------
    The input signal is a band-limited segmented signal obtained using
    shmAuditoryFiltBank.m and shmSignalSegment.m

    """

    # %% Define constants

    deltaFreq0 = 81.9289  # defined in Section 5.1.4.1 ECMA-418-2:2025
    # Half-Bark band centre-frequency denominator constant defined in Section
    # 5.1.4.1 ECMA-418-2:2025
    c = 0.1618

    halfBark = np.arange(0.5, 27, 0.5)  # half-critical band rate scale
    # Section 5.1.4.1 Equation 9 ECMA-418-2:2025
    bandCentreFreqs = (deltaFreq0/c)*np.sinh(c*halfBark)

    # Calibration factor from Section 5.1.8 Equation 23 ECMA-418-2:2025
    cal_N = 0.0211668
    cal_Nx = 1.00132  # Calibration multiplier (Footnote 8 ECMA-418-2:2025)

    a = 1.5  # Constant (alpha) from Section 5.1.8 Equation 23 ECMA-418-2:2025

    # Values from Section 5.1.8 Table 2 ECMA-418-2:2025
    p_threshold = 2e-5*10**(np.arange(15, 95, 10)/20)
    v = [1, 0.6602, 0.0864, 0.6384, 0.0328, 0.4068, 0.2082, 0.3994, 0.6434]

    # Loudness threshold in quiet Section 5.1.9 Table 3 ECMA-418-2:2025
    LTQz = np.array([0.3310, 0.1625, 0.1051, 0.0757, 0.0576, 0.0453, 0.0365,
                     0.0298, 0.0247, 0.0207, 0.0176, 0.0151, 0.0131, 0.0115,
                     0.0103, 0.0093, 0.0086, 0.0081, 0.0077, 0.0074, 0.0073,
                     0.0072, 0.0071, 0.0072, 0.0073, 0.0074, 0.0076, 0.0079,
                     0.0082, 0.0086, 0.0092, 0.0100, 0.0109, 0.0122, 0.0138,
                     0.0157, 0.0172, 0.0180, 0.0180, 0.0177, 0.0176, 0.0177,
                     0.0182, 0.0190, 0.0202, 0.0217, 0.0237, 0.0263, 0.0296,
                     0.0339, 0.0398, 0.0485, 0.0622])

    # %% Input check

    if bandCentreFreq is not None and ~np.all(np.isin(bandCentreFreq,
                                                      bandCentreFreqs)):
        raise ValueError("Input critical band centre frequency input does not match ECMA-418-2:2025 values.")
    # end of if branch to check bandCentreFreq is valid

    # %% Signal processing

    # Half Wave Rectification
    # -----------------------
    # Section 5.1.6 Equation 21 ECMA-418-2:2020
    signalRectSeg = signalSegmented
    signalRectSeg[signalSegmented <= 0] = 0

    # Calculation of RMS
    # ------------------
    # Section 5.1.7 Equation 22 ECMA-418-2:2025
    blockRMS = np.sqrt((2/signalRectSeg.shape[0])*np.sum(signalRectSeg**2,
                                                         axis=0))

    # Transformation into Loudness
    # ----------------------------
    # Section 5.1.8 Equations 23 & 24 ECMA-418-2:2025
    bandLoudness = cal_N*cal_Nx*(blockRMS/2e-5)*np.prod((1
                                                         + (np.moveaxis(np.broadcast_to(blockRMS,
                                                                                        [p_threshold.size]
                                                                                        + list(blockRMS.shape)),
                                                                        0, -1)/p_threshold)**a)**(np.diff(v)/a),
                                                        axis=-1)

    # Section 5.1.9 Equation 25 ECMA-418-2:2025
    if bandCentreFreq is not None and signalSegmented.ndim < 3:
        # half-Bark critical band basis loudness
        basisLoudness = bandLoudness - LTQz[bandCentreFreq == bandCentreFreqs]
        basisLoudness[basisLoudness < 0] = 0
    else:
        # basis loudness for all bands
        basisLoudness = bandLoudness - np.reshape(LTQz, (1, 53, 1))
        basisLoudness[basisLoudness < 0] = 0
    # end of if branch for determining basis loudness
    return (signalRectSeg, basisLoudness, blockRMS)

    # end of function


# %% shmNoiseRedLowPass
def shmNoiseRedLowPass(signal, sampleRateIn):
    """
    signalFiltered = shmNoiseRedLowPass(signal, sampleRateIn)

    Returns signal low pass filtered for noise reduction according to
    ECMA-418-2:2025 (the Sottek Hearing Model) for an input signal.

    Inputs
    ------
    signal : 1D or 2D matrix
             the input signal as single mono or stereo audio (sound
             pressure) signals

    sampleRateIn : double
                   the sample rate (frequency) of the input signal(s)

    Returns
    -------
    signalFiltered : 1D or 2D matrix
                     the filtered signal/s

    Assumptions
    -----------
    The input signal is oriented with time on axis 0 (and channel # on axis 1),
    ie, the filtering operation is applied along axis 0.

    Checked by:
    Date last checked:
    """
    k = 3  # Footnote 21 ECMA-418-2:2025
    e_i = [0, 1, 1]  # Footnote 21 ECMA-418-2:2025

    # Footnote 20 ECMA-418-2:2025
    tau = 1/32*6/7

    d = np.exp(-1/(sampleRateIn*tau))  # Section 5.1.4.2 ECMA-418-2:2025

    # Feed-backward coefficients, Equation 14 ECMA-418-2:2025
    m_a = range(1, k + 1)
    a = np.append(1, ((-d)**m_a)*comb(k, m_a))

    # Feed-forward coefficients, Equation 15 ECMA-418-2:2025
    m_b = range(k)
    i = range(1, k)
    b = (((1 - d)**k)/sum(e_i[1:]*(d**i)))*(d**m_b)*e_i

    # Recursive filter Equation 13 ECMA-418-2:2025
    signalFiltered = lfilter(b, a, signal, axis=0)

    return signalFiltered


# %% shmOutMidEarFilter
def shmOutMidEarFilter(signal, soundField='freeFrontal', outPlot=False):
    """shmOutMidEarFilter_(signal, soundField, outPlot)

    Returns signal filtered for outer and middle ear response according to
    ECMA-418-2:2025 (the Sottek Hearing Model) for an input calibrated
    audio (sound pressure) time-series signal. Optional plot output shows
    frequency and phase response of the filter.

    Inputs
    ------
    signal : 1D or 2D array
             the input signal (sound pressure)

    soundField: keyword string (default: 'freeFrontal')
                determines whether the 'freeFrontal' or 'diffuse' field
                stages are applied in the outer-middle ear filter;
                alternatively, the 'noOuter' option omits the outer ear stage
                entirely, ie, only the middle ear stage is applied.
                (this may be useful for reliance on a recording made with an
                artificial head + outer ear, when compensation equalisation
                filtering is unavailable or is not desired. Note: omitting the
                outer ear stage is beyond the current scope of the standard,
                but is consistent with the description of the implementation in
                section 5.1.3.2 and the frequency response shown in Figure 3).

    outPlot : Boolean true/false (default: false)
              flag indicating whether to generate a frequency and phase
              response figure for the filter

    Returns
    -------
    signalFiltered : 1D or 2D array
                     the output filtered signal

    Assumptions
    -----------
    The input signal is oriented with time on axis 0 (and channel # on axis
    1), ie, the filtering operation is applied along axis 0.
    The input signal must be sampled at 48 kHz.

    Checked by:
    Date last checked:

    """
    # Arguments validation
    if not isinstance(outPlot, bool):
        raise ValueError("\nInput argument 'outplot' must be logical True/False")

    # Signal processing

    # Apply outer & middle ear filter bank
    # ------------------------------------
    #
    # Filter coefficients from Section 5.1.3.2 Table 1 ECMA-418-2:2022
    # b_0k = [1.015896, 0.958943, 0.961372, 2.225804, 0.471735, 0.115267,
    #         0.988029, 1.952238]
    # b_1k = [-1.925299, -1.806088, -1.763632, -1.43465, -0.366092, 0.0,
    #         -1.912434, 0.16232]
    # b_2k = [0.922118, 0.876439, 0.821788, -0.498204, 0.244145, -0.115267,
    #         0.926132, -0.667994]
    # a_0k = ones(size(b_0k))
    # a_1k = [-1.925299, -1.806088, -1.763632, -1.43465, -0.366092, -1.796003,
    #         -1.912434, 0.16232]
    # a_2k = [0.938014, 0.835382, 0.78316, 0.727599, -0.28412, 0.805838,
    #         0.914161, 0.284244]
    #
    # Accurate coefficient values
    b_0k = [1.01589602025559, 0.958943219304445, 0.961371976333197,
            2.22580350360974, 0.471735128494163, 0.115267139824401,
            0.988029297230954, 1.95223768730136]
    b_1k = [-1.92529887777608, -1.80608801184949, -1.76363215433825,
            -1.43465048479216, -0.366091796830044, 0.0, -1.91243380293387,
            0.162319983017519]
    b_2k = [0.922118060364679, 0.876438777856084, 0.821787991845146,
            -0.498204282194628, 0.244144703885020, -0.115267139824401,
            0.926131550180785, -0.667994113035186]
    a_0k = np.ones(len(b_0k))
    a_1k = [-1.92529887777608, -1.80608801184949, -1.76363215433825,
            -1.43465048479216, -0.366091796830044, -1.79600256669201,
            -1.91243380293387, 0.162319983017519]
    a_2k = [0.938014080620272, 0.835381997160530, 0.783159968178343,
            0.727599221415107, -0.284120167620817, 0.805837815618546,
            0.914160847411739, 0.284243574266175]

    # form 2nd-order section for transfer function (copy is needed to enforce
    # C-contiguity in the tranposed array)
    if soundField == "freeFrontal":
        sos = np.array([b_0k, b_1k, b_2k, a_0k, a_1k, a_2k]).T.copy(order='C')
    elif soundField == "diffuse":
        sos = np.array([b_0k[2:], b_1k[2:], b_2k[2:],
                        a_0k[2:], a_1k[2:], a_2k[2:]]).T.copy(order='C')
    elif soundField == "noOuter":
        sos = np.array([b_0k[5:], b_1k[5:], b_2k[5:],
                        a_0k[5:], a_1k[5:], a_2k[5:]]).T.copy(order='C')
    else:
        raise ValueError("\nInput argument 'fieldtype' must be one of 'freeFrontal', 'diffuse', or 'noOuter'.")

    # Section 5.1.3.2 ECMA-418-2:2022 Outer and middle/inner ear signal filtering
    try:
        signalFiltered = sosfilt(sos, signal, axis=0)
    except TypeError as err:
        raise TypeError("\nInput signal does not appear to be an array:", err)

    # Plot figures

    if outPlot:
        f, H = sosfreqz(sos, worN=10000, whole=True, fs=48e3)
        phir = np.angle(H, deg=False)
        phirUnwrap = np.unwrap(phir, discont=np.pi)
        phiUnwrap = phirUnwrap/np.pi*180
        # Plot frequency and phase response for filter
        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(10, 7))
        ax1.semilogx(f[1:], 20*np.log10(np.abs(H[1:])), color=[0.0, 0.2, 0.8])
        ax1.set_xticks(ticks=[31.5, 63, 125, 250, 500, 1e3, 2e3, 4e3, 8e3,
                              16e3],
                       labels=["31.5", "63", "125", "250", "500", "1k", "2k",
                               "4k", "8k", "16k"])
        ax1.axis(xmin=20, xmax=20e3, ymin=-30, ymax=10)
        ax1.tick_params(axis='x',          # changes apply to the x-axis
                        which='minor',      # minor ticks are affected
                        bottom=False,     # ticks along the bottom edge are off
                        top=False,         # ticks along the top edge are off
                        labelbottom=False)  # labels along the bottom edge are off

        ax1.set_xlabel("Frequency, Hz", fontname='Arial', fontsize=12)
        ax1.set_ylabel("$H$, dB", fontname='Arial', fontsize=12)
        ax1.grid(True, which='major', linestyle='--', alpha=0.5)
        ax1.set_title(soundField)

        ax2.semilogx(f[1:], phiUnwrap[1:], color=[0.8, 0.1, 0.8])
        ax2.set_xticks(ticks=[31.5, 63, 125, 250, 500, 1e3, 2e3, 4e3, 8e3,
                              16e3],
                       labels=["31.5", "63", "125", "250", "500", "1k", "2k",
                               "4k", "8k", "16k"])
        ax2.set_yticks(ticks=np.arange(-180, 210, 30))
        ax2.axis(xmin=20, xmax=20e3, ymin=-180, ymax=90)
        ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.0f'))
        ax2.tick_params(axis='x',          # changes apply to the x-axis
                        which='minor',      # minor ticks are affected
                        bottom=False,     # ticks along the bottom edge are off
                        top=False,         # ticks along the top edge are off
                        labelbottom=False)  # labels along the bottom edge are off
        ax2.set_xlabel("Frequency, Hz", fontname='Arial', fontsize=12)
        ax2.set_ylabel("Phase angle, degrees", fontname='Arial', fontsize=12)
        ax2.grid(True, which='major', linestyle='--', alpha=0.5)

    return signalFiltered  # end of shmOutMidEarFilter function


# %% shmPreProc
def shmPreProc(signal, blockSize, hopSize, padStart=True, padEnd=True):
    """Function signalFadePad = shmPreProc(signal, blockSize, hopSize,
                                           padStart, padEnd)

    Returns signal with fade-in and zero-padding pre-processing according to
    ECMA-418-2:2025 (the Sottek Hearing Model) for an input signal.

    Inputs
    ------
    signal : 1D or 2D array
             the input signal/s

    blockSize : integer
                the maximum signal segmentation block size

    hopSize : integer
              the maximum signal segmentation hop size
              = (1 - overlap)*blockSize

    padStart : Boolean (default: True)
               flag to indicate whether to pad the start of the signal

    padEnd : Boolean (default: True)
             flag to indicate whether to pad the end of the signal


    Returns
    -------
    signalFadePad : 1D or 2D array
                    the output faded, padded signal

    Assumptions
    -----------
    The input signal is oriented with time on axis 1 (and channel # on axis
    0), ie, the fade and padding operation is applied along axis 1.
    The input signal must be sampled at 48 kHz.

    Copyright statement: This file and code is part of work undertaken within
    the RefMap project (www.refmap.eu), and is subject to licence as detailed
    in the code repository
    (https://github.com/acoustics-code-salford/refmap-psychoacoustics)

    Checked by:
    Date last checked:

    """

    # %% Signal processing

    # Input pre-processing
    # --------------------
    #
    # Check signal dimensions and add axis if 1D input
    #
    if signal.ndim == 1:
        numChans = 1
        signal = shmDimensional(signal)
    else:
        numChans = signal.shape[1]

    if numChans > 2:
        raise ValueError("Input signal must be 1- or 2-channel")

    # Fade in weighting function Section 5.1.2 ECMA-418-2:2025

    fadeWeight = 0.5 - 0.5*np.cos(np.pi*np.arange(0, 240)/240)[:, np.newaxis]
    # Apply fade in
    signalFade = np.concatenate((fadeWeight*signal[0:240, :],
                                 signal[240:, :]), axis=0)

    # Zero-padding Section 5.1.2 ECMA-418-2:2025
    if padStart:
        n_zeross = int(blockSize)  # start zero-padding
    else:
        n_zeross = 0

    if padEnd:
        n_samples = signal.shape[0]
        n_new = int(hopSize*(np.ceil((n_samples
                                      + hopSize
                                      + n_zeross)/hopSize) - 1))
        n_zerose = n_new - n_samples  # end zero-padding
    else:
        n_zerose = 0

    # Apply zero-padding
    signalFadePad = np.concatenate([np.zeros((n_zeross, numChans)),
                                    signalFade,
                                    np.zeros((n_zerose, numChans))], axis=0)

    return signalFadePad  # end of shmPreProc function


# %% shmResample
def shmResample(signal, sampleRateIn):
    """
    Returns signal resampled to 48 kHz, according to ECMA-418-2:2025
    (the Sottek Hearing Model) for an input signal.

    Inputs
    ------
    signal : 1D or 2D array
             the input signal

    sampleRateIn : integer
                   the sample rate (frequency) of the input signal(s)

    Returns
    -------
    For each channel in the input signal:

    resampledSignal : number or 1D array
                  average (overall) tonality value

    resampledRate : integer
                    the resampled signal sample rate, ie, 48 kHz

    Assumptions
    -----------
    The input signal is oriented with time on axis 1 (and channel # on axis
    0), ie, the resample operation is applied along axis 1.
    The sampleRatein is an integer value (this is to guarantee that the target
                                          rate is exactly met).

    Checked by:
    Date last checked:

    """

    # %% Define constants

    # Section 5.1.1 ECMA-418-2:2025
    resampledRate = int(48e3)  # Signal sample rate prescribed to be 48 kHz

    # %% Signal processing

    # Input pre-processing
    # --------------------
    if sampleRateIn != resampledRate:  # Resample signal
        try:
            # upsampling factor
            up = resampledRate/gcd(resampledRate, sampleRateIn)
            # downsampling factor
            down = sampleRateIn/gcd(resampledRate, sampleRateIn)

        except TypeError as err:
            raise TypeError("The input sample rate must be a positive integer to enable resampling to " + str(resampledRate) + " Hz:", err)
        try:
            # apply resampling
            resampledSignal = resample_poly(signal, up, down, axis=1)
        except TypeError as err:
            raise TypeError("TypeError: The input signal must be a numerical array:", err)
    else:  # don't resample
        resampledSignal = signal

    return resampledSignal, resampledRate  # end of shmResample function


# %% shmSignalSegmentBlocks
def shmSignalSegmentBlocks(signal, blockSize, overlap=0, axisN=0, i_start=0,
                           endShrink=False):
    """
    Returns a truncated signal and the number of blocks to use for segmentation
    processing. The signal is truncated to ensure a whole number of blocks will
    be obtained, accounting for overlap.

    Inputs
    ------
    signal : 1D or 2D array
             the input signal/s

    blockSize : integer
                the block size in samples

    overlap : double (>=0, < 1, default: 0)
              the proportion of overlap for each successive block

    axisN : integer (0 or 1, default: 0)
            the (time) axis along which to apply block segmentation

    i_start : integer (optional, default: 0)
              the sample index from which to start the segmented signal

    endShrink : Boolean (optional, default: false)
                option to include the end of the signal data in a block using
                increased overlap with the preceding block

    Returns
    -------
    For each channel in the input signal:

    signalTrunc :  1D or 2D array
                   the truncated signal/s

    nBlocksTotal : integer
                   the total number of blocks to use for signal segmentation

    excessSignal : Boolean
                   flag indicating whether there is sufficient signal to
                   accommodate an extra block with increased overlap

    Assumptions
    -----------
    None

    """

    # %% Signal pre-processing

    # Orient input
    if axisN == 1:
        signal = signal.T

    # ensure i_start is positive integer
    try:
        i_start = int(abs(i_start))
    except TypeError:
        raise TypeError("Input argument i_start must be a (positive real) number.")

    # Check signal dimensions and add axis if 1D input
    if signal.ndim == 1:
        signal = shmDimensional(signal)

    # Check sample index start will allow segmentation to proceed
    if signal[i_start:, :].shape[0] <= blockSize:
        raise ValueError("Signal is too short to apply segmentation using the selected parameters.")
    # end

    # Hop size
    hopSize = int((1 - overlap)*blockSize)

    # Truncate the signal to start from i_start and to end at an index
    # corresponding with the truncated signal length that will fill an
    # integer number of overlapped blocks
    signalTrunc = signal[i_start:, :]
    nBlocks = int(np.floor((signalTrunc.shape[0]
                            - overlap*blockSize)/hopSize))
    i_end = int(nBlocks*hopSize + overlap*blockSize - 1)
    signalTrunc = signalTrunc[0:i_end + 1, :]

    # Determine total number of blocks including an extra block if adding a
    # frame with an increased overlap
    nBlocksTotal = nBlocks
    if signal[i_start:, :].shape[0] > signalTrunc.shape[0]:
        excessSignal = True
        if endShrink:
            nBlocksTotal = nBlocks + 1
    else:
        excessSignal = False

    return signalTrunc, nBlocksTotal, excessSignal


# %% shmSignalSegment
def shmSignalSegment(signal, blockSize, overlap=0, axisN=0, i_start=0,
                     endShrink=False):
    """
    Returns input signal segmented into blocks for processing.

    Inputs
    ------
    signal : 1D or 2D array
             the input signal/s

    blockSize : integer
                the block size in samples

    overlap : double (>=0, < 1, default: 0)
              the proportion of overlap for each successive block

    axisN : integer (0 or 1, default: 0)
            the (time) axis along which to apply block segmentation

    i_start : integer (optional, default: 0)
              the sample index from which to start the segmented signal

    endShrink : Boolean (optional, default: false)
                option to include the end of the signal data in a block using
                increased overlap with the preceding block

    Returns
    -------
    For each channel in the input signal:

    signalSegmented : 2D or 3D array
                      the segmented signal, arranged by channels over the
                      last axis, samples (within each block) along the axis
                      corresponding with axisN, and block number along the
                      other remaining axis

    Also:

    iBlocksOut : 1D array
                 the indices corresponding with each output block starting
                 index (NOTE: the indices corresponding with the input
                 indexing can be recovered by adding i_start to iBlocksOut)

    Assumptions
    -----------
    None

    """

    # %% Signal pre-processing

    # Orient input
    if axisN == 1:
        signal = signal.T
        axisFlip = True
    elif axisN == 0:
        axisFlip = False
    else:
        raise ValueError("Input argument 'axisN' can take values 0 (default) or 1.")

    # ensure i_start is positive integer
    try:
        i_start = int(abs(i_start))
    except TypeError:
        raise TypeError("Input argument i_start must be a (positive real) number.")

    # Check signal dimensions and add axis if 1D input
    #
    if signal.ndim == 1:
        numChans = 1
        signal = shmDimensional(signal)
    else:
        numChans = signal.shape[1]

    # Check sample index start will allow segmentation to proceed
    if signal[i_start:, :].shape[0] <= blockSize:
        raise ValueError("Signal is too short to apply segmentation using the selected parameters.")
    # end

    # Hop size
    hopSize = int((1 - overlap)*blockSize)

    # Truncate the signal to start from i_start and to end at an index
    # corresponding with the truncated signal length that will fill an
    # integer number of overlapped blocks
    signalTrunc, nBlocksTotal, excessSignal = shmSignalSegmentBlocks(signal,
                                                                     blockSize,
                                                                     overlap=overlap,
                                                                     axisN=0,
                                                                     i_start=i_start,
                                                                     endShrink=endShrink)

    # %% Signal segmentation

    # Arrange the signal into overlapped blocks - each block reads
    # along first axis, and each column is the succeeding overlapped
    # block. 3 columns of zeros are appended to the left side of the
    # matrix and the column shifted copies of this matrix are
    # concatenated. The first 6 columns are then discarded as these all
    # contain zeros from the appended zero columns.
    signalSegmented = np.zeros((blockSize, nBlocksTotal, numChans))

    for chan in range(0, numChans):
        signalSegmentedChan = np.concatenate((np.zeros((hopSize, 3)),
                                              np.reshape(signalTrunc[:, chan],
                                                         (hopSize, -1),
                                                         order='F')),
                                             axis=1)

        signalSegmentedChan = np.concatenate((np.roll(signalSegmentedChan,
                                                      shift=3, axis=1),
                                              np.roll(signalSegmentedChan,
                                                      shift=2, axis=1),
                                              np.roll(signalSegmentedChan,
                                                      shift=1, axis=1),
                                              np.roll(signalSegmentedChan,
                                                      shift=0, axis=1)),
                                             axis=0)

        signalSegmentedChan = signalSegmentedChan[:, 6:]

        # if branch to include block of end data with increased overlap
        if endShrink and excessSignal:
            signalChan = signal[-blockSize:, chan]
            signalSegmentedChanOut = np.concatenate((signalSegmentedChan,
                                                     signalChan[:,
                                                                np.newaxis]),
                                                    axis=1)
            iBlocksOut = np.append(np.arange(0, (nBlocksTotal - 1)*hopSize,
                                             hopSize),
                                   signal[i_start:, chan].size - blockSize)
        else:
            signalSegmentedChanOut = signalSegmentedChan
            iBlocksOut = np.arange(0, nBlocksTotal*hopSize, hopSize)
        # end of if branch for end data with increased overlap

        signalSegmented[:, :, chan] = signalSegmentedChanOut

    # end of for loop over channels

    # squeeze singleton dimensions
    if numChans == 1:
        signalSegmented = np.squeeze(signalSegmented)
        # re-orient segmented signal to match input
        if axisFlip:
            signalSegmented = np.swapaxes(signalSegmented, 0, 1)
    elif axisFlip:
        signalSegmented = np.transpose(signalSegmented, [2, 0, 1])

    # end of if branch to revert shape of input

    return (signalSegmented, iBlocksOut)  # end of shmSignalSegment function


# %% shmDownsample
def shmDownsample(ndArray, axisN=0, downSample=32):
    """
    Return array downsampled by keeping every value at downSample indices
    after the first.

    Parameters
    ----------
    ndArray : array_like
              Input array

    axisN : integer (0 or 1, default: 0)
            the (time) axis along which to apply the downsampling

    downSample : integer
                 The downsampleing factor

    Returns
    -------
    targArray : nD array
                Output numpy array downsampled.

    Assumptions
    -----------
    Input ndArray is 1D or 2D

    """

    # check input type and try to convert to numpy array
    if type(ndArray) is np.ndarray:
        pass
    else:
        try:
            ndArray = np.array(ndArray)
        except TypeError:
            raise TypeError("Input ndArray must be an array-like object.")

    # %% Input array pre-processing

    # Orient input
    if axisN == 1:
        ndArray = ndArray.T
        axisFlip = True
    elif axisN == 0:
        axisFlip = False
    else:
        raise ValueError("Input argument 'axisN' can take values 0 (default) or 1.")

    # %% Downsampling
    # remove every value in between the values to be retained
    if ndArray.ndim == 2:
        targArray = ndArray[0:-1:downSample, :]
        if axisFlip:
            targArray = np.swapaxes(targArray, 0, 1)
    elif ndArray.ndim == 1:
        targArray = ndArray[0:-1:downSample]
    else:
        raise ValueError("Input ndArray must be 1D or 2D")

    return targArray


# %% shmDimensional
def shmDimensional(ndArray, targetDim=2, where='last'):
    """
    Return array increased by dimensions depending on difference between
    targetDim and len(ndarray.shape), placed either 'first' or 'last' according
    to where keyword argument

    Parameters
    ----------
    ndArray : array_like
              Input array

    targetDim : TYPE, optional
        The target number of dimensions. The default is 2.

    where : keyword string or corresponding integer (0, -1), optional
            Where the added dimensions are to be placed.
            The default is 'last' (-1). The altgernative is 'first' (0).

    Returns
    -------
    targArray : nD array
                Output numpy array increased by dimensions.

    """

    # check input type and try to convert to numpy array
    if type(ndArray) is np.ndarray:
        pass
    else:
        try:
            ndArray = np.array(ndArray)
        except TypeError:
            raise TypeError("Input ndArray must be an array-like object.")

    # assign dimensions to add
    dimsToAdd = targetDim - ndArray.ndim

    # add number of dimensions
    targArray = ndArray
    if dimsToAdd <= 0:
        return targArray
    else:
        if where == 'first' or where == 0:
            for ii in range(dimsToAdd):
                targArray = np.expand_dims(targArray, 0)
        elif where == 'last' or where == -1:
            for ii in range(dimsToAdd):
                targArray = np.expand_dims(targArray, -1)
        else:
            raise ValueError("Input argument 'where' must either have string values 'first' or 'last', or integer values 0 or -1")

    return targArray  # end of shmDimensional function


# %% shmRoughWeight
def shmRoughWeight(modRate, modfreqMaxWeight, roughWeightParams):
    """
    Returns roughness weighting for high- and low-frequency (modulation
    rates) according to ECMA-418-2:2025 (the Sottek Hearing Model) for a set
    of modulation rates and parameters.

    Inputs
    ------

    modRate : 3D array
              the estimated modulation rates used to determine the weighting
              factors

    modfreqMaxWeight : 1D array
                       the modulation rate at which the weighting reaches its
                       maximum value (one)

    roughWeightParams : array
                        the parameters for the each of the weightings (high
                        or low)

    Returns
    -------
    roughWeight : array
                  the weighting values for the input parameters

    Assumptions
    -----------
    Inputs are in compatible parallelised (broadcastable) forms

    Checked by:
    Date last checked:
    """
    # Equation 85 [G_l,z,i(f_p,i(l,z))]
    roughWeight = np.divide(1, (1 + ((modRate/modfreqMaxWeight
                                      - np.divide(modfreqMaxWeight, modRate,
                                                  out=np.zeros_like(modRate),
                                                  where=modRate != 0))
                                     * roughWeightParams[0, :, :])**2)**roughWeightParams[1, :, :],
                            out=np.zeros_like(modRate),
                            where=modRate != 0)

    return roughWeight  # end of shmRoughWeight function


# %% shmRoughLowPass
def shmRoughLowPass(specRoughEstTform, sampleRate, riseTime, fallTime):
    """
    specRoughness = shmRoughLowPass(specRoughEstTform, sampleRate, riseTime,
                                    fallTime)

    Returns specific roughness low pass filtered for smoothing according to
    ECMA-418-2:2025 (the Sottek Hearing Model) for an input transformed
    estimate of the specific roughnesss.

    Inputs
    ------
    specRoughEstTform : 2D array
                        the input specific roughness estimate (from
                        Equation 104)

    sampleRate : double
                 the sample rate (frequency) of the input specific
                 roughness (NB: this is not the original signal sample
                 rate; currently it should be set to 50 Hz)

    Returns
    -------
    specRoughness : 2D array
                    the filtered specific roughness

    Assumptions
    -----------
    The input specific roughness estimate is orientated with time on axis 0,
    and critical bands on axis 1.

    Checked by:
    Date last checked:
    """
    riseExponent = np.exp(-1/(sampleRate*riseTime))*np.ones([specRoughEstTform.shape[1]])
    fallExponent = np.exp(-1/(sampleRate*fallTime))*np.ones([specRoughEstTform.shape[1]])

    specRoughness = specRoughEstTform.copy()

    for llBlock in range(1, specRoughEstTform.shape[0]):

        riseMask = (specRoughEstTform[llBlock, :]
                    >= specRoughness[llBlock - 1, :])
        fallMask = ~riseMask

        if specRoughEstTform[llBlock, riseMask].size != 0:
            specRoughness[llBlock, riseMask] = (specRoughEstTform[llBlock,
                                                                  riseMask]*(1 - riseExponent[riseMask])
                                                + specRoughness[llBlock - 1,
                                                                riseMask]*riseExponent[riseMask])
        # end of rise branch
        if specRoughEstTform[llBlock, fallMask].size != 0:
            specRoughness[llBlock, fallMask] = (specRoughEstTform[llBlock,
                                                                  fallMask]*(1 - fallExponent[fallMask])
                                                + specRoughness[llBlock - 1,
                                                                fallMask]*fallExponent[fallMask])
        # end of fall branch

    return specRoughness  # end of shmRoughLowPass function


# %% shmRound
def shmRound(vals, decimals=0):
    """
    Returns a set of rounded values to the specified number of decimal places
    using the traditional approach.

    Inputs
    ------

    vals : float or array of floats
           the values to be rounded.

    decimals : int
               the number of decimal places to round to (default=0).

    Returns
    -------

     : array of floats
          The rounded values.

    """

    try:
        vals = np.asarray(vals)
    except TypeError:
        raise TypeError

    return np.round(vals + 10**(-vals.astype(str).size - 1), decimals)


# %% shmRMS
def shmRMS(vals, axis=0, keepdims=False):
    """
    Returns the root-mean-square for values contained in an array.

    Inputs
    ------

    vals : float or array of floats
           the values to be rounded.

    axis : integer
           the axis along which to calculate the RMS.

    keepdims : Boolean
               flag to indicate whether to retain dimensions along the mean
               axis (with size of one), for broadcasting compatibility.

    Returns
    -------

     : array of floats
          The root-mean-squared values.

    """

    try:
        vals = np.asarray(vals)
    except TypeError:
        raise TypeError

    return np.sqrt(np.mean(np.square(vals), axis=axis, keepdims=keepdims))
