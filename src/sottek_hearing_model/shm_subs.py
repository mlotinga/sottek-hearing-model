# -*- coding: utf-8 -*-
# %% Preamble
"""
shm_subs.py
-----------

Acoustic signal analysis subfunctions for implementing the Sottek Hearing
Model, as defined in the ECMA-418-2 standard (currently 2025).

Requirements
------------
numpy
scipy
matplotlib
bottleneck

Functions
---------

shm_auditory_filtbank : Implements the auditory filter bank based on half-overlapping
                        critical bands.
shm_basis_loudness : Computes the basis loudness.
shm_outmid_ear_filter : Applies the outer-middle ear filter according to the selected
                        sound field.
shm_pre_proc : Prepares the input signal with truncation and zero-padding
shm_resample : Resamples the input signal to the prescribed rate.
shm_signal_segment_blocks : Segments the input signal into blocks.
shm_signal_segment : Segments the input signal into non-overlapping frames.
shm_downsample : Downsamples the input signal using a simple throw-away
                 decimation method.
shm_dimensional : Adds or removes array dimensions.
shm_mod_weight : Computes modulation weighting for the input signal.
shm_mod_lowpass : Applies a low-pass smoothing filter to the specific modulation.
shm_round : Rounds the input signal to the nearest integer using a 'traditional'
            rounding method.
shm_rms : Computes the root mean square of the input array.
shm_in_check : Checks the input arguments for validity.
shm_mov_median : Computes a centred moving median with shrinking endpoint
                 windows (equivalent to MATLAB movmedian).
shm_loud_nonlin : Applies the loudness nonlinearity to a root-mean-square
                  sound pressure.
shm_fluct_weight : Computes the fluctuation strength band-pass modulation-rate
                   weighting.
shm_hsa_window_response : Computes the analysis-window frequency response used
                          by the High-resolution Spectral Analysis (HSA).
shm_hsa : Solves the High-resolution Spectral Analysis (HSA) linear system for
          a set of candidate modulation rates.

Ownership and Quality Assurance
-------------------------------
Author: Mike JB Lotinga (m.j.lotinga@edu.salford.ac.uk)
Institution: University of Salford

Date created: 27/10/2023
Date last modified: 18/09/2026
Python version: 3.11

Copyright statement: This code has been developed during work undertaken within
the RefMap project (www.refmap.eu), based on the RefMap code repository
(https://github.com/acoustics-code-salford/refmap-psychoacoustics),
and as such is subject to copyleft licensing as detailed in the code repository
(https://github.com/acoustics-code-salford/sottek-hearing-model).

As per the licensing information, please be aware that this code is WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.

Parts of this code were developed from an original MATLAB file
'SottekTonality.m' authored by Matt Torjussen (14/02/2022), based on
implementing ECMA-418-2:2020. The original code has been reused and translated
here with permission.

"""

# %% Imports and parameter setup
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
from scipy.signal import (freqz, lfilter, resample_poly, sosfilt, sosfreqz)
from scipy.special import comb
from math import gcd
import bottleneck as bn
from sottek_hearing_model.plotting_tools import create_figure, show_plot

# set plot parameters
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = 'Arial'
mpl.rcParams.update({'font.size': 14})
mpl.rcParams['mathtext.fontset'] = 'stixsans'
mpl.rcParams['figure.autolayout'] = True


# %% shm_auditory_filtbank
def shm_auditory_filtbank(signal, out_plot=False):
    """shm_auditory_filtbank(signal, out_plot=False)

    Returns a set of signals, bandpass filtered for the inner ear response
    in each half-Bark critical band rate scale width, according to
    ECMA-418-2:2025 (the Sottek Hearing Model) for an input calibrated
    single (sound pressure) time-series signal.

    Parameters
    ----------
    signal : 1D or 2D array
        Input signal as single audio (sound pressure) signal/s,
        arranged as [time(, chans)].

    out_plot : Boolean true/false (default: false)
        Flag indicating whether to generate a figure a frequency and
        phase response figure for the filter bank.

    Returns
    -------
    signal_filtered : 2D of 3D array
        Filtered signals, arranged as [time, bands(, chans)].

    Assumptions
    -----------
    The input signal is a numpy array oriented with time on axis 0, ie, the
    filter operation is applied along axis 0.
    The input signal must be sampled at 48 kHz.

    """

    # %% Define constants

    samp_rate48k = 48e3  # Signal sample rate prescribed to be 48kHz (to be used for resampling), Section 5.1.1 ECMA-418-2:2025
    delta_freq0 = 81.9289  # defined in Section 5.1.4.1 ECMA-418-2:2025
    c = 0.1618  # Half-Bark band centre-frequency demoninator constant defined in Section 5.1.4.1 ECMA-418-2:2025

    dz = 0.5  # critical band resolution
    half_bark = np.arange(0.5, 27, dz)  # half-overlapping critical band rate scale
    # Section 5.1.4.1 Equation 9 ECMA-418-2:2025
    band_centre_freqs = (delta_freq0/c)*np.sinh(c*half_bark)
    # Section 5.1.4.1 Equation 10 ECMA-418-2:2025
    dfz = np.sqrt(delta_freq0**2 + (c*band_centre_freqs)**2)

    # %% Signal processing

    # Apply auditory filter bank
    # --------------------------
    # Filter equalised signal using 53 1/2Bark ERB filters according to
    # Section 5.1.4.2 ECMA-418-2:2025

    k = 5  # filter order = 5, footnote 5 ECMA-418-2:2025
    # filter coefficients for Section 5.1.4.2 Equation 15 ECMA-418-2:2025
    e_i = [0, 1, 11, 11, 1]

    if signal.ndim == 2:
        signal_filtered = np.zeros((signal.shape[0], len(half_bark),
                                    signal.shape[1]), order='F')
    elif signal.ndim == 1:
        signal_filtered = np.zeros((signal.size, len(half_bark)), order='F')
    else:
        raise ValueError("Input signal must have no more than two dimensions.")

    for z_band in range(53):
        # Section 5.1.4.1 Equation 8 ECMA-418-2:2025
        tau = (1/(2**(2*k - 1)))*comb(2*k - 2, k - 1)*(1/dfz[z_band])

        d = np.exp(-1./(samp_rate48k*tau))  # Section 5.1.4.1 ECMA-418-2:2025

        # Band-pass modifier Section 5.1.4.2 Equation 16/17 ECMA-418-2:2025
        bp = np.exp((1j*2*np.pi*band_centre_freqs[z_band]*np.arange(0, k + 2))
                    / samp_rate48k)

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
        if signal_filtered.ndim == 3:
            signal_filtered[:, z_band, :] = 2*np.real(lfilter(b_m, a_m, signal,
                                                              axis=0))
        else:
            signal_filtered[:, z_band] = 2*np.real(lfilter(b_m, a_m, signal,
                                                           axis=0))

        # Plot figures

        if out_plot:
            f, H = freqz(b_m, a=a_m, worN=int(10e3), whole=True,
                         fs=samp_rate48k)
            phir = np.angle(H)
            phirUnwrap = np.unwrap(p=phir, discont=np.pi, axis=0)
            phiUnwrap = phirUnwrap/np.pi*180
            # Plot frequency and phase response for filter
            if z_band == 0:
                fig, axs = create_figure(nrows=2, ncols=1)
                ax1 = axs[0]
                ax2 = axs[1]

            ax1.semilogx(f, 20*np.log10(abs(H)))
            ax2.semilogx(f, phiUnwrap)

            if z_band == 52:
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
            
            show_plot(fig)

        # end of if branch for outPlot
    # end of for loop over bands

    return signal_filtered
# end of shm_auditory_filtbank function


# %% shm_basis_loudness
def shm_basis_loudness(signal_segmented, band_centre_freq=None):
    """shm_basis_loudness(signal_segmented, band_centre_freq=None)

    Returns rectified input and basis loudness in specified half-Bark
    critical band according to ECMA-418-2:2025 (the Sottek Hearing Model)
    for an input band-limited signal, segmented into processing blocks

    Parameters
    ----------
    signal_segmented : 2D or 3D array
        Input band-limited segmented signal(s).

    band_centre_freq : double (optional, default: None)
        Critical band centre frequency - if None, all
        bands are assumed to be present in the input segmented
        signal matrix.

    Returns
    -------
    signal_rect_seg : 2D or 3D array
        Rectified band-limited segmented signal.
    
    basis_loudness : 2D or 3D array
        Basis loudness in each block.

    block_rms : 1D or 2D array
        RMS for each time block.

    Assumptions
    -----------
    The input signal is a band-limited segmented signal obtained using
    shm_auditory_filtbank.m and shm_signal_segment.m

    """

    # %% Define constants

    delta_freq0 = 81.9289  # defined in Section 5.1.4.1 ECMA-418-2:2025
    # Half-Bark band centre-frequency denominator constant defined in Section
    # 5.1.4.1 ECMA-418-2:2025
    c = 0.1618

    half_bark = np.arange(0.5, 27, 0.5)  # half-critical band rate scale
    # Section 5.1.4.1 Equation 9 ECMA-418-2:2025
    band_centre_freqs = (delta_freq0/c)*np.sinh(c*half_bark)

    # Calibration factor from Section 5.1.8 Equation 23 ECMA-418-2:2025
    cal_N = 0.0211668
    cal_Nx = 1.00132  # Calibration multiplier (Footnote 8 ECMA-418-2:2025)

    a = 1.5  # Constant (alpha) from Section 5.1.8 Equation 23 ECMA-418-2:2025

    # Values from Section 5.1.8 Table 2 ECMA-418-2:2025
    p_threshold = 2e-5*10**(np.arange(15, 95, 10)/20)
    v = [1, 0.6602, 0.0864, 0.6384, 0.0328, 0.4068, 0.2082, 0.3994, 0.6434]

    # Loudness threshold in quiet Section 5.1.9 Table 3 ECMA-418-2:2025
    loud_thresh = np.array([0.3310, 0.1625, 0.1051, 0.0757, 0.0576, 0.0453,
                            0.0365, 0.0298, 0.0247, 0.0207, 0.0176, 0.0151,
                            0.0131, 0.0115, 0.0103, 0.0093, 0.0086, 0.0081,
                            0.0077, 0.0074, 0.0073, 0.0072, 0.0071, 0.0072,
                            0.0073, 0.0074, 0.0076, 0.0079, 0.0082, 0.0086,
                            0.0092, 0.0100, 0.0109, 0.0122, 0.0138, 0.0157,
                            0.0172, 0.0180, 0.0180, 0.0177, 0.0176, 0.0177,
                            0.0182, 0.0190, 0.0202, 0.0217, 0.0237, 0.0263,
                            0.0296, 0.0339, 0.0398, 0.0485, 0.0622])

    # %% Input check

    if band_centre_freq is not None and ~np.all(np.isin(band_centre_freq,
                                                        band_centre_freqs)):
        raise ValueError("Input critical band centre frequency input does not match ECMA-418-2:2025 values.")
    # end of if branch to check bandCentreFreq is valid

    # %% Signal processing

    # Half Wave Rectification
    # -----------------------
    # Section 5.1.6 Equation 21 ECMA-418-2:2020
    signal_rect_seg = signal_segmented
    signal_rect_seg[signal_segmented <= 0] = 0

    # Calculation of RMS
    # ------------------
    # Section 5.1.7 Equation 22 ECMA-418-2:2025
    block_rms = np.sqrt((2/signal_rect_seg.shape[0])*np.sum(signal_rect_seg**2,
                                                            axis=0))

    # Transformation into Loudness
    # ----------------------------
    # Section 5.1.8 Equations 23 & 24 ECMA-418-2:2025
    band_loudness = (cal_N
                     * cal_Nx
                     * (block_rms/2e-5)*np.prod((1 + (np.moveaxis(np.broadcast_to(block_rms,
                                                                                  [p_threshold.size]
                                                                                  + list(block_rms.shape)),
                                                                  0, -1)
                                                      / p_threshold)**a)
                                                ** (np.diff(v)/a), axis=-1))

    # Section 5.1.9 Equation 25 ECMA-418-2:2025
    if band_centre_freq is not None and signal_segmented.ndim < 3:
        # half-Bark critical band basis loudness
        basis_loudness = band_loudness - loud_thresh[band_centre_freq == band_centre_freqs]
        basis_loudness[basis_loudness < 0] = 0
    else:
        # basis loudness for all bands
        basis_loudness = band_loudness - np.reshape(loud_thresh, (1, 53, 1))
        basis_loudness[basis_loudness < 0] = 0
    # end of if branch for determining basis loudness
    return (signal_rect_seg, basis_loudness, block_rms)
# end of shm_basis_loudness function


# %% shm_outmid_ear_filter
def shm_outmid_ear_filter(signal, soundfield='free_frontal', out_plot=False):
    """shm_outmid_ear_filter(signal, soundfield, out_plot)

    Returns signal filtered for outer and middle ear response according to
    ECMA-418-2:2025 (the Sottek Hearing Model) for an input calibrated
    audio (sound pressure) time-series signal. Optional plot output shows
    frequency and phase response of the filter.

    Parameters
    ----------
    signal : 1D or 2D array
        Input signal (sound pressure).

    soundfield: keyword string (default: 'free_frontal')
        Determines whether the 'free_frontal' or 'diffuse' field stages
        are applied in the outer-middle ear filter, or 'no_outer' uses
        only the middle ear stage, or 'no_ear' omits ear filtering.
        note: these last two options are beyond the scope of the
        standard, but may be useful if recordings made using
        artificial outer/middle ear are to be processed using the
        specific recorded responses.

    out_plot : Boolean true/false (default: false)
        Flag indicating whether to generate a frequency and phase
        response figure for the filter.

    Returns
    -------
    signal_filtered : 1D or 2D array
        Output filtered signal.

    Assumptions
    -----------
    The input signal is oriented with time on axis 0 (and channel # on axis
    1), ie, the filtering operation is applied along axis 0.
    The input signal must be sampled at 48 kHz.

    """

    # Signal processing

    if soundfield == "no_ear":
        signal_filtered = signal
        return signal_filtered
    else:
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
        if soundfield == "free_frontal":
            sos = np.array([b_0k, b_1k, b_2k, a_0k, a_1k, a_2k]).T.copy(order='C')
        elif soundfield == "diffuse":
            sos = np.array([b_0k[2:], b_1k[2:], b_2k[2:],
                            a_0k[2:], a_1k[2:], a_2k[2:]]).T.copy(order='C')
        elif soundfield == "no_outer":
            sos = np.array([b_0k[5:], b_1k[5:], b_2k[5:],
                            a_0k[5:], a_1k[5:], a_2k[5:]]).T.copy(order='C')
        else:
            raise ValueError("\nInput argument 'soundfield' must be one of 'free_frontal', 'diffuse', 'no_outer', or 'no_ear'.")

        # Section 5.1.3.2 ECMA-418-2:2022 Outer and middle/inner ear signal filtering
        try:
            signal_filtered = sosfilt(sos, signal, axis=0)
        except TypeError as err:
            raise TypeError("\nInput signal does not appear to be an array:", err)

        # Plot figures

        if out_plot:
            freqs, response = sosfreqz(sos, worN=10000, whole=True, fs=48e3)
            phir = np.angle(response, deg=False)
            phir_unwrap = np.unwrap(phir, discont=np.pi)
            phi_unwrap = phir_unwrap/np.pi*180
            # Plot frequency and phase response for filter
            fig, (ax1, ax2) = create_figure(nrows=2, ncols=1, figsize=(10, 7))
            ax1.semilogx(freqs[1:], 20*np.log10(np.abs(response[1:])),
                         color=[0.0, 0.2, 0.8])
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
            ax1.set_title(soundfield)

            ax2.semilogx(freqs[1:], phi_unwrap[1:], color=[0.8, 0.1, 0.8])
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

            show_plot(fig)

        return signal_filtered
# end of shm_outmid_ear_filter function


# %% shm_pre_proc
def shm_pre_proc(signal, block_size, hop_size, pad_start=True, pad_end=True):
    """shm_pre_proc(signal, block_size, hop_size, pad_start, pad_end)

    Returns signal with fade-in and zero-padding pre-processing according to
    ECMA-418-2:2025 (the Sottek Hearing Model) for an input signal.

    Parameters
    ----------
    signal : 1D or 2D array
        Input signal/s.

    block_size : integer
        Maximum signal segmentation block size.

    hop_size : integer
        Maximum signal segmentation hop size
        = (1 - overlap)*block_size.

    pad_start : Boolean (default: True)
        Flag to indicate whether to pad the start of the signal.

    pad_end : Boolean (default: True)
        Flag to indicate whether to pad the end of the signal.


    Returns
    -------
    signal_fade_pad : 1D or 2D array
        Output faded, padded signal.

    Assumptions
    -----------
    The input signal is oriented with time on axis 0 (and channel # on axis
    1), ie, the fade and padding operation is applied along axis 0.
    The input signal must be sampled at 48 kHz.

    """

    # %% Signal processing

    # Input pre-processing
    # --------------------
    #
    # Check signal dimensions and add axis if 1D input
    #
    if signal.ndim == 1:
        num_chans = 1
        signal = shm_dimensional(signal)
    else:
        num_chans = signal.shape[1]

    if num_chans > 2:
        raise ValueError("Input signal must be 1- or 2-channel")

    # Fade in weighting function Section 5.1.2 ECMA-418-2:2025

    fade_weight = 0.5 - 0.5*np.cos(np.pi*np.arange(0, 240)/240)[:, np.newaxis]
    # Apply fade in
    signal_fade = np.concatenate((fade_weight*signal[0:240, :],
                                  signal[240:, :]), axis=0)

    # Zero-padding Section 5.1.2 ECMA-418-2:2025
    if pad_start:
        n_zeross = int(block_size)  # start zero-padding
    else:
        n_zeross = 0

    if pad_end:
        n_samples = signal.shape[0]
        n_new = int(hop_size*(np.ceil((n_samples
                                       + hop_size
                                       + n_zeross)/hop_size)
                              - 1))
        n_zerose = n_new - n_samples  # end zero-padding
    else:
        n_zerose = 0

    # Apply zero-padding
    signal_fade_pad = np.concatenate([np.zeros((n_zeross, num_chans),
                                               order='F'), signal_fade,
                                      np.zeros((n_zerose, num_chans),
                                               order='F')], axis=0)

    # Check memory layout and convert to column-major, if row-major
    if signal_fade_pad.flags.f_contiguous:
        pass
    else:
        signal_fade_pad = np.asfortranarray(signal_fade_pad)

    return signal_fade_pad
# end of shmPreProc function


# %% shm_resample
def shm_resample(signal, samp_rate_in):
    """shm_resample(signal, samp_rate_in)

    Returns signal resampled to 48 kHz, according to ECMA-418-2:2025
    (the Sottek Hearing Model) for an input signal.

    Parameters
    ----------
    signal : 1D or 2D array
        Input signal.

    samp_rate_in : integer
        Sample rate (frequency) of the input signal(s).

    Returns
    -------
    resampled_signal : array
        Resampled signal (resampled at 48 kHz).

    samp_rate48k : integer
        Output sample rate (48 kHz).

    Assumptions
    -----------
    The input signal is oriented with time on axis 0 (and channel # on axis
    1), ie, the resample operation is applied along axis 0.
    The samp_rate_in is an integer value (this is to guarantee that the target
                                          rate is exactly met).


    """

    # %% Define constants

    # Section 5.1.1 ECMA-418-2:2025
    samp_rate48k = int(48e3)  # Signal sample rate prescribed to be 48 kHz

    # %% Signal processing

    # Input pre-processing
    # --------------------
    if samp_rate_in != samp_rate48k:  # Resample signal

        try:
            if isinstance(samp_rate_in, int) and samp_rate_in > 0:
                pass
            
            elif samp_rate_in.is_integer() is True and samp_rate_in > 0:
                samp_rate_in = int(samp_rate_in)

            else:
                raise TypeError("The input sample rate must be a positive integer to enable resampling to " + str(samp_rate48k) + " Hz:")
            
            # upsampling factor
            up = samp_rate48k/gcd(samp_rate48k, samp_rate_in)
            # downsampling factor
            down = samp_rate_in/gcd(samp_rate48k, samp_rate_in)

        except TypeError as err:
            raise TypeError("The input sample rate must be a positive integer to enable resampling to " + str(samp_rate48k) + " Hz:", err)

        try:
            # apply resampling
            resampled_signal = resample_poly(signal, up, down, axis=0)
        except TypeError as err:
            raise TypeError("TypeError: The input signal must be a numerical array:", err)
    else:  # don't resample
        resampled_signal = signal

    return resampled_signal, samp_rate48k
# end of shm_resample function


# %% shm_signal_segment_blocks
def shm_signal_segment_blocks(signal, block_size, overlap=0, i_start=0,
                              end_shrink=False):
    """shm_signal_segment_blocks(signal, block_size, overlap=0, i_start=0,
                                 end_shrink=False)

    Returns a truncated signal and the number of blocks to use for segmentation
    processing. The signal is truncated to ensure a whole number of blocks will
    be obtained, accounting for overlap.

    Parameters
    ----------
    signal : 1D array
        Input signal.

    block_size : integer
        Block size in samples.

    overlap : double (>=0, < 1, default: 0)
        Proportion of overlap for each successive block.

    i_start : integer (optional, default: 0)
        Sample index from which to start the segmented signal.

    end_shrink : Boolean (optional, default: false)
        Option to include the end of the signal data in a block using
        increased overlap with the preceding block.

    Returns
    -------
    signal_trunc :  1D array
        Truncated signal.

    n_blocks_total : integer
        Total number of blocks to use for signal segmentation.

    excess_signal : Boolean
        Flag indicating whether there is sufficient signal to
        accommodate an extra block with increased overlap.

    Assumptions
    -----------
    The input signal is oriented with time on axis 0.

    """

    # %% Signal pre-processing

    # ensure i_start is positive integer
    try:
        i_start = int(abs(i_start))
    except TypeError:
        raise TypeError("Input argument i_start must be a (positive real) number.")

    # Check signal dimensions and add axis if 1D input
    if signal.ndim == 1:
        signal = shm_dimensional(signal)

    # Check sample index start will allow segmentation to proceed
    if signal[i_start:, :].shape[0] <= block_size:
        raise ValueError("Signal is too short to apply segmentation using the selected parameters.")
    # end

    # Hop size
    hop_size = int((1 - overlap)*block_size)

    # Truncate the signal to start from i_start and to end at an index
    # corresponding with the truncated signal length that will fill an
    # integer number of overlapped blocks
    signal_trunc = signal[i_start:, :]
    n_blocks = int(np.floor((signal_trunc.shape[0]
                            - overlap*block_size)/hop_size))
    i_end = int(n_blocks*hop_size + overlap*block_size - 1)
    signal_trunc = signal_trunc[0:i_end + 1, :]

    # Determine total number of blocks including an extra block if adding a
    # frame with an increased overlap
    n_blocks_total = n_blocks
    if signal[i_start:, :].shape[0] > signal_trunc.shape[0]:
        excess_signal = True
        if end_shrink:
            n_blocks_total = n_blocks + 1
    else:
        excess_signal = False

    return signal_trunc, n_blocks_total, excess_signal
# end of shm_signal_segment_blocks function


# %% shm_signal_segment
def shm_signal_segment(signal, block_size, overlap=0, i_start=0,
                       end_shrink=False):
    """shm_signal_segment(signal, block_size, overlap=0, i_start=0,
                       end_shrink=False)

    Returns input signal segmented into blocks for processing.

    Parameters
    ----------
    signal : 1D array
        Input signal.

    block_size : integer
        Block size in samples.

    overlap : double (>=0, < 1, default: 0)
        Proportion of overlap for each successive block.

    i_start : integer (optional, default: 0)
        Sample index from which to start the segmented signal.

    end_shrink : Boolean (optional, default: false)
        Option to include the end of the signal data in a block using
        increased overlap with the preceding block.

    Returns
    -------
    signal_segmented : 2D array
        Segmented signal, arranged by samples (within each
        block) along the axis corresponding with axis 0, and
        block number along axis 1.

    i_start : integer (optional, default: 0)
        Sample index from which to start the segmented signal.

    Also:

    i_blocks_out : 1D array
        Indices corresponding with each output block starting
        index (NOTE: the indices corresponding with the input
        indexing can be recovered by adding i_start to i_blocks_out).

    Assumptions
    -----------
    The input signal is oriented with time on axis 0.

    """

    # %% Signal pre-processing

    # ensure i_start is positive integer
    try:
        i_start = int(abs(i_start))
    except TypeError:
        raise TypeError("Input argument i_start must be a (positive real) number.")

    # Check signal dimensions and add axis if 1D input
    #
    if signal.ndim == 1:
        num_chans = 1
        signal = shm_dimensional(signal)
    else:
        num_chans = signal.shape[1]

    # Check sample index start will allow segmentation to proceed
    if signal[i_start:, :].shape[0] <= block_size:
        raise ValueError("Signal is too short to apply segmentation using the selected parameters.")
    # end

    # Hop size
    hop_size = int((1 - overlap)*block_size)

    # Truncate the signal to start from i_start and to end at an index
    # corresponding with the truncated signal length that will fill an
    # integer number of overlapped blocks
    (signal_trunc,
     n_blocks_total,
     excess_signal) = shm_signal_segment_blocks(signal,
                                                block_size,
                                                overlap=overlap,
                                                i_start=i_start,
                                                end_shrink=end_shrink)

    # %% Signal segmentation

    # Arrange the signal into overlapped blocks - each block reads
    # along first axis, and each column is the succeeding overlapped
    # block. 3 columns of zeros are appended to the left side of the
    # matrix and the column shifted copies of this matrix are
    # concatenated. The first 6 columns are then discarded as these all
    # contain zeros from the appended zero columns.
    signal_segmented = np.concatenate((np.zeros((hop_size, 3), order='F'),
                                       np.reshape(signal_trunc[:, 0],
                                                  (hop_size, -1),
                                                  order='F')),
                                      axis=1)

    signal_segmented = np.concatenate((np.roll(signal_segmented,
                                               shift=3, axis=1),
                                       np.roll(signal_segmented,
                                               shift=2, axis=1),
                                       np.roll(signal_segmented,
                                               shift=1, axis=1),
                                       np.roll(signal_segmented,
                                               shift=0, axis=1)),
                                      axis=0)

    signal_segmented = signal_segmented[:, 6:]

    # if branch to include block of end data with increased overlap
    if end_shrink and excess_signal:
        signal_end = signal[-block_size:, 0]
        signal_segmented_out = np.concatenate((signal_segmented,
                                               signal_end[:,
                                                          np.newaxis]),
                                              axis=1)
        i_blocks_out = np.append(np.arange(0, (n_blocks_total
                                               - 1)*hop_size,
                                           hop_size),
                                 signal[i_start:, 0].size
                                 - block_size)
    else:
        signal_segmented_out = signal_segmented
        i_blocks_out = np.arange(0, n_blocks_total*hop_size,
                                 hop_size)
    # end of if branch for end data with increased overlap

    # squeeze singleton dimensions
    if num_chans == 1:
        signal_segmented_out = np.squeeze(signal_segmented_out)

    return (signal_segmented_out, i_blocks_out)
# end of shmSignalSegment function


# %% shm_downsample
def shm_downsample(ndArray, downsample=32):
    """shm_downsample(ndArray, downsample=32)

    Returns array downsampled by keeping every value at
    downsample indices after the first.

    Parameters
    ----------
    ndArray : array_like
        Input array.

    downsample : integer
        Downsampling factor.

    Returns
    -------
    targ_array : nD array
        Output numpy array downsampled.

    Assumptions
    -----------
    Input ndArray is 1D or 2D with time axis along axis 0.

    """

    # %% Downsampling
    # remove every value in between the values to be retained
    if ndArray.ndim == 2:
        targ_array = ndArray[0:-1:downsample, :]
    elif ndArray.ndim == 1:
        targ_array = ndArray[0:-1:downsample]
    else:
        raise ValueError("Input ndArray must be 1D or 2D")

    return targ_array


# %% shm_dimensional
def shm_dimensional(ndArray, target_dim=2, where='last'):
    """shm_dimensional(ndArray, target_dim=2, where='last')

    Returns array increased by dimensions depending on difference between
    target_dim and len(ndarray.shape), placed either 'first' or 'last'
    according to where keyword argument

    Parameters
    ----------
    ndArray : array_like
        Input array.

    target_dim : integer, optional
        Target number of dimensions. The default is 2.

    where : keyword string or corresponding integer (0, -1), optional
        Where the added dimensions are to be placed.
        The default is 'last' (-1). The alternative is 'first' (0).

    Returns
    -------
    targ_array : nD array
        Output numpy array increased by dimensions.

    Assumptions
    -----------
    Input is a numpy ndArray.

    """

    # assign dimensions to add
    dimsToAdd = target_dim - ndArray.ndim

    # add number of dimensions
    targ_array = ndArray
    if dimsToAdd <= 0:
        return targ_array
    else:
        if where == 'first' or where == 0:
            for ii in np.arange(dimsToAdd):
                targ_array = np.expand_dims(targ_array, 0)
        elif where == 'last' or where == -1:
            for ii in np.arange(dimsToAdd):
                targ_array = np.expand_dims(targ_array, -1)
        else:
            raise ValueError("Input argument 'where' must either have string values 'first' or 'last', or integer values 0 or -1")

    return targ_array
# end of shm_dimensional function


# %% shm_mod_weight
def shm_mod_weight(mod_rate, mod_freq_max_weight, weight_params):
    """shm_mod_weight(mod_rate, mod_freq_max_weight, weight_params)

    Returns modulation weightings according to ECMA-418-2:2025 (the Sottek Hearing Model)
    for a set of modulation rates and parameters.

    Parameters
    ----------
    mod_rate : 3D array
        Estimated modulation rates used to determine the weighting
        factors.

    mod_freq_max_weight : 1D array
        Modulation rate at which the weighting reaches its
        maximum value (one).

    weight_params : array
        Parameters for the the weighting functions.

    Returns
    -------
    mod_weight : array
        Weighting values for the input parameters.

    Assumptions
    -----------
    Inputs are in compatible parallelised (broadcastable) forms.

    """
    # Equation 85 [G_l,z,i(f_p,i(l,z))]
    mod_weight = np.divide(1, (1 + ((mod_rate/mod_freq_max_weight
                                     - np.divide(mod_freq_max_weight, mod_rate,
                                                 out=np.zeros_like(mod_rate),
                                                 where=mod_rate != 0))
                                    * weight_params[0, :, :])**2)
                           ** weight_params[1, :, :],
                           out=np.zeros_like(mod_rate),
                           where=mod_rate != 0)

    return mod_weight
# end of shm_mod_weight function


# %% shm_mod_lowpass
def shm_mod_lowpass(spec_modulation, samp_rate, rise_time, fall_time):
    """shm_mod_lowpass(spec_modulation, samp_rate, rise_time, fall_time)

    Returns specific modulation (roughness or fluctuation strength), low-pass
    filtered for smoothing according to ECMA-418-2:2025 (the Sottek Hearing Model)
    for an input specific modulation (roughness or fluctuation strength).

    Parameters
    ----------
    spec_modulation : 2D array
        Input specific modulation (roughness or fluctuation strength).

    samp_rate : double
        Sample rate (frequency) of the input specific modulation
        (NB: this is not the original signal sample
        rate; currently it should be set to 50 Hz).

    rise_time : double
        Rise time constant (s) for the low-pass filter.

    fall_time : double
        Fall time constant (s) for the low-pass filter.

    Returns
    -------
    spec_modulation : 2D array
        Low-pass filtered specific modulation
        (roughness or fluctuation strength).

    Assumptions
    -----------
    The input specific modulation is orientated with time on axis 0,
    and critical bands on axis 1.

    """
    rise_exp = np.exp(-1/(samp_rate*rise_time))*np.ones([spec_modulation.shape[1]])
    fall_exp = np.exp(-1/(samp_rate*fall_time))*np.ones([spec_modulation.shape[1]])

    spec_mod_filtered = spec_modulation.copy()

    for llBlock in range(1, spec_modulation.shape[0]):

        rise_mask = (spec_modulation[llBlock, :]
                    >= spec_mod_filtered[llBlock - 1, :])
        fall_mask = ~rise_mask

        if spec_modulation[llBlock, rise_mask].size != 0:
            spec_mod_filtered[llBlock,
                              rise_mask] = (spec_modulation[llBlock,
                                                            rise_mask]*(1 - rise_exp[rise_mask])
                                            + spec_mod_filtered[llBlock - 1,
                                                                rise_mask]*rise_exp[rise_mask])
        # end of rise branch
        if spec_modulation[llBlock, fall_mask].size != 0:
            spec_mod_filtered[llBlock,
                              fall_mask] = (spec_modulation[llBlock,
                                                            fall_mask]*(1 - fall_exp[fall_mask])
                                            + spec_mod_filtered[llBlock - 1,
                                                                fall_mask]*fall_exp[fall_mask])
        # end of fall branch

    return spec_mod_filtered
# end of shm_mod_lowpass function


# %% shm_round
def shm_round(vals, decimals=0):
    """shm_round(vals, decimals=0)

    Returns a set of rounded values to the specified number of decimal places
    using the traditional approach.

    Parameters
    ----------

    vals : float or array of floats
        Values to be rounded.

    decimals : int
        Number of decimal places to round to (default=0).

    Returns
    -------

     : array of floats
        Rounded values.

    """
    return np.round(vals + 10**(-vals.astype(str).size - 1), decimals)


# %% shm_rms
def shm_rms(vals, axis=0, keepdims=False):
    """shm_rms(vals, axis=0, keepdims=False)
    Returns the root-mean-square for values contained in an array.

    Parameters
    ----------

    vals : float or array of floats
        Values to be rounded.

    axis : integer
        Axis along which to calculate the RMS.

    keepdims : Boolean
        Flag to indicate whether to retain dimensions along the mean
        axis (with size of one), for broadcasting compatibility.

    Returns
    -------

     : array of floats
        Root-mean-squared values.

    """

    try:
        vals = np.asarray(vals)
    except TypeError:
        raise TypeError

    return np.sqrt(np.mean(np.square(vals), axis=axis, keepdims=keepdims))


# %% shm_in_check
def shm_in_check(signal, samp_rate_in, axis, soundfield,
                 wait_bar, out_plot, binaural=None,
                 parallel_cores=None, annoy_weight=None):
    """shm_in_check(signal, samp_rate_in, axis, soundfield,
                    wait_bar, out_plot, binaural=None,
                    parallel_cores=None, annoy_weight=None)

    Input checking and error handling function.

    Parameters
    ----------

    signal : 1D or 2D array
        Input signal as single mono or stereo audio (sound
        pressure) signals.

    samp_rate_in : integer
        Sample rate (frequency) of the input signal(s).

    axis : integer (0 or 1, default: 0)
        Time axis along which to calculate the sound quality metric.

    soundfield : keyword string (default: 'free_frontal')
        Determines whether the 'free_frontal' or 'diffuse' field stages
        are applied in the outer-middle ear filter, or 'no_outer' uses
        only the middle ear stage, or 'no_ear' omits ear filtering.
        Note: these last two options are beyond the scope of the
        standard, but may be useful if recordings made using
        artificial outer/middle ear are to be processed using the
        specific recorded responses.

    wait_bar : keyword string (default: True)
        Determines whether a progress bar displays during processing
        (set wait_bar to false for doing multi-file parallel calculations).

    out_plot : Boolean (default: False)
        Flag indicating whether to generate a figure from the output
        (set out_plot to false for doing multi-file parallel calculations).

    binaural : None or Boolean (default: True)
        Flag indicating whether to output combined binaural sound
        quality for stereo input signal. If None, this variable will
        not be checked.

    parallel_cores : integer or None (default: None)
        Number of parallel cores to use for processing
        (if None, the number of cores is automatically determined based
        on the number of available CPU cores; for multicore systems,
        1 core is always left free, to avoid system freeze.
        If 1, parallel processing is not applied).

    annoy_weight : Boolean (default: False)
        Flag indicating whether to include tonal annoyance-weighted results in
        the output. If None, this variable will
        not be checked.

    Returns
    -------
    signal : 1D or 2D array
        Input signal as single mono or stereo audio (sound
        pressure) signals, orientated to ensure time is on the first axis.

    chans_in : integer
        Number of input channels.

    chans : list of strings
        Text labels indicating the input signal channels.

    """

    # Check input is an array and try casting to array if not
    if type(signal) is np.ndarray:
        pass
    else:
        try:
            signal = np.array(signal)
        except TypeError:
            raise TypeError("\nInput signal must be an array-like object.")

    # Orient input signal
    if axis in [0, 1]:
        if axis == 1 and signal.ndim > 1:
            signal = signal.T
    else:
        raise ValueError("\nInput axis must be an integer 0 or 1")

    # Check signal dimensions and add axis if 1D input
    if signal.ndim == 1:
        chans_in = 1
        signal = shm_dimensional(signal)
    else:
        chans_in = signal.shape[1]

    # Check memory layout and convert to column-major, if row-major
    if signal.flags.f_contiguous:
        pass
    else:
        signal = np.asfortranarray(signal)

    # Check the channel number of the input data
    if signal.shape[1] > 2:
        raise ValueError("\nInput signal comprises more than two channels")

    # assign channel labels
    if chans_in > 1:
        chans = ["Stereo left",
                 "Stereo right"]
    else:
        chans = ["Mono"]

    # Check the length of the input data (must be longer than 300 ms)
    if signal.shape[0] <= 300/1000*samp_rate_in:
        raise ValueError("\nInput signal is too short along the specified axis (must be longer than 300 ms)")

    # Check if soundField has a valid value
    try:
        soundfield_in = soundfield.lower()
    except TypeError:
        raise TypeError("\nInput argument 'soundField' must be a keyword string. Options include: 'free_frontal', 'diffuse', 'no_outer' or 'no_ear'.")
        if soundfield_in not in ['free_frontal', 'diffuse', 'no_outer', 'no_ear']:
            raise ValueError("\nInput argument 'soundField' must be one of the options: 'free_frontal', 'diffuse', 'no_outer' or 'no_ear'.")

    # Check if out_plot has a valid value
    if not isinstance(out_plot, bool):
        raise ValueError("\nInput argument 'outplot' must be logical True/False")

    # Check if binaural has a valid value
    if binaural is not None:
        if not isinstance(binaural, bool):
            raise ValueError("\nInput argument 'binaural' must be logical True/False")

    # Check if parallel_cores has a valid value
    if parallel_cores is not None:
        try:
            # check if parallel_cores is an integer
            if not isinstance(parallel_cores, int):
                raise TypeError("\nInput argument 'parallel_cores' must be a positive integer or None.")

            if parallel_cores < 1:
                raise ValueError("\nInput argument 'parallel_cores' must be a positive integer or None.")

        except TypeError:
            raise TypeError("\nInput argument 'parallel_cores' must be a positive integer or None.")

    # Check if annoy_weight has a valid value
    if annoy_weight is not None:
        if not isinstance(annoy_weight, bool):
            raise ValueError("\nInput argument 'annoy_weight' must be logical True/False")

    return signal, chans_in, chans
# end of shm_in_check function


# %% shm_mov_median
def shm_mov_median(vals, win_len, axis=0):
    """shm_mov_median(vals, win_len, axis=0)

    Returns the centred moving median of an array along a given axis, with
    the window shrinking symmetrically at the endpoints (i.e. the median is
    taken over the elements that exist within the half-window either side
    of each point). This is equivalent to the MATLAB movmedian function
    with its default 'shrink' endpoint handling, which is the moving median
    filter used in Sections 9.1.3.2 and 9.1.11 of ECMA-418-2:2025.

    Parameters
    ----------
    vals : nD array
        Input array.

    win_len : integer (odd)
        Moving window length in samples.

    axis : integer (default: 0)
        Axis along which to apply the moving median.

    Returns
    -------
    med_vals : nD array
        Moving median of the input array, the same shape as vals.

    Assumptions
    -----------
    win_len is an odd positive integer, so that the window is symmetric
    about each point.

    """

    # %% Input check
    if win_len < 1 or win_len % 2 == 0:
        raise ValueError("Input argument 'win_len' must be an odd positive integer.")

    vals = np.asarray(vals, dtype=float)
    n = vals.shape[axis]
    half = (win_len - 1)//2

    # %% Signal processing

    if n <= win_len:
        # short input: form the shrinking-window median directly
        med_vals = np.empty_like(vals)
        for ii in range(n):
            lo = max(0, ii - half)
            hi = min(n, ii + half + 1)
            med_slice = np.median(np.take(vals, np.arange(lo, hi), axis=axis),
                                  axis=axis)
            index = [slice(None)]*vals.ndim
            index[axis] = ii
            med_vals[tuple(index)] = med_slice
        return med_vals

    # trailing (causal) moving median with shrinking window at the start:
    # med_fwd[i] = median(vals[i - win_len + 1:i + 1])
    med_fwd = bn.move_median(vals, window=win_len, min_count=1, axis=axis)
    # leading (anti-causal) moving median with shrinking window at the end,
    # obtained by applying the trailing filter to the flipped array:
    # med_bwd[i] = median(vals[i:i + win_len])
    med_bwd = np.flip(bn.move_median(np.flip(vals, axis=axis), window=win_len,
                                     min_count=1, axis=axis), axis=axis)

    # centred median: for c <= n - 1 - half, median(vals[c - half:c + half + 1])
    # = med_fwd[c + half] (which also shrinks correctly at the start); for the
    # final half points, = med_bwd[c - half] (which shrinks correctly at the
    # end)
    med_vals = np.concatenate((np.take(med_fwd, np.arange(half, n), axis=axis),
                               np.take(med_bwd, np.arange(n - 2*half, n - half),
                                       axis=axis)), axis=axis)

    return med_vals
# end of shm_mov_median function


# %% shm_loud_nonlin
def shm_loud_nonlin(p_rms):
    """shm_loud_nonlin(p_rms)

    Returns the loudness obtained by applying the nonlinear transformation
    of Section 5.1.8 Equation 23 ECMA-418-2:2025 (the Sottek Hearing Model)
    to an input root-mean-square sound pressure (without the subtraction of
    the threshold-in-quiet loudness of Section 5.1.9).

    Parameters
    ----------
    p_rms : float or array of floats
        Root-mean-square sound pressure(s) [Pa].

    Returns
    -------
    loud_nonlin : float or array of floats
        Loudness value(s) [sone_HMS/Bark_HMS], the same shape as p_rms.

    Assumptions
    -----------
    The input is calibrated to units of acoustic pressure in Pascals (Pa).

    """

    # %% Define constants

    # Section 5.1.8 Equation 23/24 ECMA-418-2:2025 [c_N], [alpha]
    cal_N = 0.0211668
    cal_Nx = 1.00132  # calibration adjustment factor (Footnote 8 ECMA-418-2:2025)
    a = 1.5

    # Section 5.1.8 Table 2 ECMA-418-2:2025 [p_ti], [nu_i] (nu_0 = 1 prepended)
    p_threshold = 2e-5*10**(np.arange(15, 95, 10)/20)
    v = np.array([1, 0.6602, 0.0864, 0.6384, 0.0328, 0.4068, 0.2082, 0.3994,
                  0.6434])

    # %% Signal processing

    p_rms = np.asarray(p_rms, dtype=float)

    # Section 5.1.8 Equation 23 ECMA-418-2:2025
    loud_nonlin = (cal_N*cal_Nx*(p_rms/2e-5)
                   * np.prod((1 + (p_rms[..., np.newaxis]/p_threshold)**a)
                             ** (np.diff(v)/a), axis=-1))

    return loud_nonlin
# end of shm_loud_nonlin function


# %% shm_fluct_weight
def shm_fluct_weight(mod_rate, band_centre_freq):
    """shm_fluct_weight(mod_rate, band_centre_freq)

    Returns the fluctuation strength band-pass modulation-rate weighting
    w_lh according to ECMA-418-2:2025 (the Sottek Hearing Model), Section
    9.1.6, Equation 148.

    Parameters
    ----------
    mod_rate : float or array of floats
        Modulation rate(s) [Hz] at which to evaluate the weighting (values
        of zero or less return a weighting of zero).

    band_centre_freq : float
        Critical band centre frequency F(z) [Hz], used in the carrier-
        frequency correction applied to the high-modulation-rate branch.

    Returns
    -------
    fluct_weight : array of floats
        The weighting values w_lh(mod_rate), the same shape as mod_rate.

    Assumptions
    -----------
    band_centre_freq is a scalar value corresponding with a single critical
    band, consistent with a single call of this function per band.

    """

    # %% Define constants

    # Section 9.1.6 Equation 148
    f_max = 4.8659  # [f_max], Hz: modulation rate of maximum (unity) weighting
    q1l = 0.33048
    q2l = 0.85902
    q1h = 0.21792
    q2h = 4.6728

    # carrier-frequency correction factor for the high-modulation-rate branch
    freq_correction = (1 + 0.092623*np.abs(np.log2(band_centre_freq/1000))**1.24)**-1

    # %% Signal processing

    mod_rate = np.atleast_1d(np.asarray(mod_rate, dtype=float))
    fluct_weight = np.zeros(mod_rate.shape)

    mask_lo = (mod_rate > 0) & (mod_rate <= f_max)
    mask_hi = mod_rate > f_max

    fluct_weight[mask_lo] = (1/(1 + ((mod_rate[mask_lo]/f_max
                                      - f_max/mod_rate[mask_lo])*q1l)**2))**q2l
    fluct_weight[mask_hi] = freq_correction*(1/(1 + ((mod_rate[mask_hi]/f_max
                                                      - f_max/mod_rate[mask_hi])
                                                     * q1h)**2))**q2h

    return fluct_weight
# end of shm_fluct_weight function


# %% shm_hsa_window_response
def shm_hsa_window_response(k_indices, mod_rate, block_size, samp_rate,
                            n_zeros_start, n_zeros_end, epsilon=1e-12):
    """shm_hsa_window_response(k_indices, mod_rate, block_size, samp_rate,
                               n_zeros_start, n_zeros_end, epsilon=1e-12)

    Returns the analysis-window frequency response used by the High-
    resolution Spectral Analysis (HSA), according to ECMA-418-2:2025 (the
    Sottek Hearing Model), Section 9.1.4, Equation 127.

    Parameters
    ----------
    k_indices : 1D array
        The (zero-based) DFT bin indices k at which to evaluate the window
        response, k = 0, ..., K_L - 1.

    mod_rate : float
        The candidate modulation rate f_c,m [Hz] at which the window is
        centred (may be zero, positive or negative).

    block_size : integer
        The downsampled analysis block size s~b (Section 9.1.2).

    samp_rate : float
        The downsampled analysis sample rate r~s (Section 9.1.2).

    n_zeros_start : integer
        Number of zeros at the start of the envelope analysis window,
        n_zb,l,z.

    n_zeros_end : integer
        Number of zeros at the end of the envelope analysis window,
        n_ze,l,z.

    epsilon : float (default: 1e-12)
        Small constant substituted for the standard's epsilon_0 (defined as
        the smallest positive double such that 1 + epsilon_0 > 1), added to
        avoid division by zero when a candidate modulation rate falls
        exactly on a DFT bin centre. The value used has no measurable
        effect on the result (see the Note in shm_hsa), and 1e-12 is used
        for consistency with the other functions in this package.

    Returns
    -------
    window_response : 1D complex array
        The complex window response W_E,l,z,fc,m(k), the same length as
        k_indices.

    Assumptions
    -----------
    k_indices are zero-based DFT bin indices, consistent with the zero-based
    indexing convention used throughout ECMA-418-2:2025.

    """

    # %% Signal processing

    # number of samples with unity weight in the analysis window [n_active]
    n_active = block_size - n_zeros_end - n_zeros_start

    # Section 9.1.4 Equation 127 [f_n(k)]
    freq_norm = k_indices/block_size - mod_rate/samp_rate + epsilon

    # CORRECTION relative to the printed standard: Equation 127 as typeset in
    # ECMA-418-2:2025 shows the phase term exp(-j*2*pi*f_n(k)*(...)), which
    # is a factor of 2 too large. A closed-form re-derivation of the
    # underlying geometric series (the DFT of a rectangular window of
    # n_active ones preceded by n_zb zeros), and a brute-force numerical DFT
    # comparison, both confirm that the correct phase term is
    # exp(-j*pi*f_n(k)*(...)) (see shmHSAWindowResponse.m and test_shmHSA.m
    # in the refmap-psychoacoustics repository).
    # As-written: window_response = (np.exp(-1j*2*np.pi*freq_norm*(block_size - n_zeros_end + n_zeros_start - 1))
    #                                *np.sin(np.pi*freq_norm*n_active)/np.sin(np.pi*freq_norm))
    window_response = (np.exp(-1j*np.pi*freq_norm*(block_size - n_zeros_end
                                                    + n_zeros_start - 1))
                       * np.sin(np.pi*freq_norm*n_active)/np.sin(np.pi*freq_norm))

    return window_response
# end of shm_hsa_window_response function


# %% shm_hsa
def shm_hsa(fc, spectrum_e, block_size, samp_rate, n_zeros_start, n_zeros_end,
            epsilon=1e-12):
    """shm_hsa(fc, spectrum_e, block_size, samp_rate, n_zeros_start,
               n_zeros_end, epsilon=1e-12)

    Returns the High-resolution Spectral Analysis (HSA) estimate of the
    constant (DC) component and the complex spectral line amplitude(s) at
    the candidate modulation rate(s) fc, together with the corresponding
    HSA error function value, according to ECMA-418-2:2025 (the Sottek
    Hearing Model), Section 9.1.4.

    Parameters
    ----------
    fc : float or 1D array
        Candidate modulation rate(s) [Hz] of the Mc non-zero spectral line
        pairs under consideration, fc = (fc_1, ..., fc_Mc). The zero
        (constant) component is handled internally and must NOT be included
        in fc. Negative values are accepted (the error function E_l,z is an
        even function of each fc, since W+ is even and W- is odd in fc), so
        that the Newton iteration of Section 9.1.7 may pass through zero and
        the resulting f_c,1,opt < 0.125 Hz is then discarded by the caller
        as the standard specifies, rather than raising an error.

    spectrum_e : 1D complex array
        The s~b-point complex DFT spectrum P_E,l,z(k) of the windowed,
        downsampled envelope (Section 9.1.4 Equation 121), full length
        (k = 0, ..., s~b - 1).

    block_size : integer
        Downsampled analysis block size s~b (Section 9.1.2).

    samp_rate : float
        Downsampled analysis sample rate r~s (Section 9.1.2).

    n_zeros_start : integer
        Number of zeros at the start of the envelope analysis window,
        n_zb,l,z.

    n_zeros_end : integer
        Number of zeros at the end of the envelope analysis window,
        n_ze,l,z.

    epsilon : float (default: 1e-12)
        Small constant substituted for the standard's epsilon_0 (see
        shm_hsa_window_response).

    Returns
    -------
    p_hat : 1D complex array, length Mc + 1
        The HSA-estimated complex spectral amplitudes: p_hat[0] = phat_0,l,z
        (real-valued constant part), p_hat[1:] = phat_fc,m,l,z (complex),
        m = 1, ..., Mc, in the same order as the input fc. The spectral
        line amplitudes are TWO-SIDED line amplitudes, i.e. an envelope
        component a*cos(2*pi*fc*t + phi) is returned as
        phat_fc = (a/2)*exp(1j*phi) (see the Note below).

    err_lz : float
        The HSA error function value E_l,z(fc) (Section 9.1.4 Equation 135).

    Assumptions
    -----------
    fc contains only the non-zero candidate modulation rate(s); the
    constant (DC) part is always included as an additional unknown and
    must not be passed explicitly.

    Note
    ----
    This function implements the general Mc-line case (Section 9.1.4,
    Equations 121-135) directly, solving the (2*Mc + 1)-by-(2*Mc + 1) linear
    system of Equation 130. This is also used for the single spectral line
    pair case (Mc = 1), for which Section 9.1.4.1 additionally provides a
    closed-form solution via Cramer's rule (Equations 136-142). The
    closed-form solution is mathematically identical to the general solution
    for Mc = 1 (it is simply an expanded, manual method of solving the same
    3-by-3 instance of Formula (130)), so solving the general system is not
    an approximation or a shortcut: it produces the same result (to
    floating-point rounding) while keeping a single, simpler, and more
    easily verified code path for all values of Mc.

    Note also that Formula (126) defines the column of W corresponding to
    the "minus" component (index i = 2*m + 1) as the COMPLEX CONJUGATE of
    W-_E,l,z,fc,m(k) (Equation 129); this conjugation is essential to
    obtaining correct results and is easy to miss when transcribing the
    formulae, since Equation 129 itself does not show the conjugation
    (it is introduced only in Formula (126), and repeated in the a13/a23/b3
    terms of Formulae (138)-(139) for the Mc = 1 case).

    Note on the amplitude convention of phat_fc,m (Equation 123): the
    real-valued normal equations of Formulae (130)-(134) are exactly the
    least-squares fit of the model
      Phat(k) = x_1*W_0(k) + sum_m [x_2m*W+_m(k) + j*x_2m+1*W-_m(k)]
    to P_E,l,z(k), which is the DFT of the windowed envelope model
      x_1 + sum_m 2*Re((x_2m + j*x_2m+1)*exp(j*2*pi*fc,m*ntilde/rs~)).
    Hence (x_2m + j*x_2m+1) is the two-sided line amplitude (the line at
    +fc,m, with its conjugate at -fc,m), and 2*(x_2m + j*x_2m+1) would be
    the one-sided (cosine) amplitude. Equation 123 as printed
    (x_i = Re(phat)/2, Im(phat)/2) implies the one-sided convention, but
    Equations 159-160 (phat_0^2 + 2*sum(A_i) as the mean-square power of
    the harmonic complex, and sqrt(0.5*(...)) as the RMS sound pressure
    passed to the nonlinearity of Equation 23) and footnote 46 (the DFT
    line values of Equation 66 equal the HSA line values multiplied by the
    DFT length s~b) are only consistent with the two-sided convention.
    The two-sided convention is therefore used here, and has been
    verified against reference results: using the literal Equation 123
    factor of 2 overestimates |phat|^2 by a factor of 4 and the
    harmonic-complex power by up to a factor of 2, and reproduces a
    modulation-depth-dependent overestimation of fluctuation strength
    (approx. 1.8x at m = 1 rising to 3x at m = 0.25) relative to the
    reference results.

    """

    # %% Define constants

    fc = np.atleast_1d(np.asarray(fc, dtype=float))
    mc = fc.size  # number of non-zero candidate spectral line pairs
    n_cols = 2*mc + 1  # number of unknowns/columns, size of x [2Mc + 1]

    # Section 9.1.4 Equation 125 [Delta f] and [K_L]
    # (np.floor(x + 0.5) reproduces the standard's rounding to the nearest
    # integer, half away from zero, for the non-negative argument here)
    delta_f = samp_rate/block_size
    kl = int(min(max(17, np.floor(np.max(np.abs(fc))/delta_f + 0.5) + 8), 49))

    k_indices = np.arange(kl)  # zero-based DFT bin indices used in the fit

    # %% Signal processing

    # Section 9.1.4 Equation 126 - build the matrix of window response
    # column vectors W = (W_1, ..., W_2Mc+1)
    w = np.zeros((kl, n_cols), dtype=complex)

    # constant (DC) part, i = 1 [W_E,l,z,0(k)]
    w[:, 0] = shm_hsa_window_response(k_indices, 0.0, block_size, samp_rate,
                                      n_zeros_start, n_zeros_end, epsilon)

    for m_line in range(mc):
        window_pos = shm_hsa_window_response(k_indices, fc[m_line], block_size,
                                             samp_rate, n_zeros_start,
                                             n_zeros_end, epsilon)
        window_neg = shm_hsa_window_response(k_indices, -fc[m_line], block_size,
                                             samp_rate, n_zeros_start,
                                             n_zeros_end, epsilon)

        # Equation 128 [W+_E,l,z,fc,m(k)], i = 2m
        w[:, 2*m_line + 1] = window_pos + window_neg

        # Equation 129 [W-_E,l,z,fc,m(k)], conjugated per Equation 126, i = 2m+1
        w[:, 2*m_line + 2] = np.conj(window_pos - window_neg)

    # Section 9.1.4 Equations 132-133 - index sets I_R (and, by the same
    # definition, J_R) identifying which columns are treated as the 'real'
    # (cosine-type) part of the system (i = 1 or mod(i, 2) = 0 in the
    # standard's 1-based column numbering)
    is_real_col = np.zeros(n_cols, dtype=bool)
    is_real_col[0] = True
    is_real_col[1::2] = True
    same_bucket = is_real_col[:, np.newaxis] == is_real_col[np.newaxis, :]

    # Section 9.1.4 Equation 131 - symmetric matrix A
    # (the formula for a_ij is symmetric under exchange of i and j in both
    # branches, so the full matrix can be built directly without separately
    # enforcing symmetry)
    wr = np.real(w)
    wi = np.imag(w)
    a = ((wr.T@wr + wi.T@wi)*same_bucket
         + (wi.T@wr + wr.T@wi)*(~same_bucket))

    # Section 9.1.4 Equation 134 - vector b
    spectrum_ek = spectrum_e[:kl]  # PE,l,z(k) at the KL bins used
    pe_r = np.real(spectrum_ek)
    pe_i = np.imag(spectrum_ek)

    b = np.zeros(n_cols)
    b[is_real_col] = wr[:, is_real_col].T@pe_r + wi[:, is_real_col].T@pe_i
    b[~is_real_col] = wr[:, ~is_real_col].T@pe_i + wi[:, ~is_real_col].T@pe_r

    # Section 9.1.4 Equation 130 - solve A*x = b
    # (a numerical robustness safeguard, not specified by the standard: fall
    # back to the minimum-norm least-squares solution if A is close to
    # singular, which can occur for pathological candidate frequency sets)
    try:
        rcond_a = 1/np.linalg.cond(a, p=1)
    except np.linalg.LinAlgError:
        rcond_a = 0.0
    if not np.isfinite(rcond_a) or rcond_a < 1e3*np.finfo(float).eps:
        x = np.linalg.pinv(a)@b
    else:
        x = np.linalg.solve(a, b)

    # Section 9.1.4 Equation 135 - error function
    # (a_ii*x_i^2 summed with 2*sum_{i<j}(a_ij*xi*xj) is exactly x'*A*x for
    # symmetric A, giving a compact, fully vectorised equivalent of Formula
    # (135))
    err_lz = np.sum(np.abs(spectrum_ek)**2) + x@a@x - 2*b@x

    # Section 9.1.4 Equation 123 (inverted) - recover the complex spectral
    # amplitudes from the solution vector x, as TWO-SIDED line amplitudes
    # phat_fc,m = x_2m + j*x_2m+1 (see the Note in the function help)
    p_hat = np.zeros(mc + 1, dtype=complex)
    p_hat[0] = x[0]  # [phat_0,l,z], real-valued
    p_hat[1:] = x[1::2] + 1j*x[2::2]  # [phat_fc,m,l,z], two-sided line amplitude

    return p_hat, err_lz
# end of shm_hsa function
