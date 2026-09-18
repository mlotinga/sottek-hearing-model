# -*- coding: utf-8 -*-
# %% Preamble
"""
shm_fluctuation_ecma.py
-----------------------

Returns fluctuation strength values according to ECMA-418-2:2025 (using the
Sottek Hearing Model) for an input calibrated single mono or single stereo
audio (sound pressure) time-series signal, p.

Requirements
------------
numpy
scipy
matplotlib
tqdm
bottleneck

Functions
---------

shm_fluctuation_ecma : This is the main fluctuation strength function, which
                       implements section 9 of ECMA-418-2:2025, and returns a
                       dict containing the fluctuation strength results as
                       numpy arrays.

shm_fluct_envelopes : Segments the signal into time blocks and computes the
                      downsampled signal envelopes for each critical band.

shm_env_window : Determines the envelope analysis window parameters (number
                 of leading and trailing zeros, and validity) for a single
                 time block and critical band.

shm_hsa_block : Runs the High-resolution Spectral Analysis (HSA) pipeline
                (candidate modulation rate search, fine tuning, harmonic
                analysis and HSA-based loudness) for a single time block and
                critical band.

Ownership and Quality Assurance
-------------------------------
Author: Mike JB Lotinga (m.j.lotinga@edu.salford.ac.uk)
Institution: University of Salford

Date created: 18/09/2026
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

This code was translated from the MATLAB file 'acousticSHMFluctuation.m'
(and its subfunctions) in the refmap-psychoacoustics repository.

"""

# %% Import block
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from scipy.fft import (fft)
from scipy.signal import (hilbert, find_peaks)
from scipy.interpolate import PchipInterpolator
from sottek_hearing_model.shm_subs import (shm_dimensional,
                                           shm_resample,
                                           shm_pre_proc,
                                           shm_outmid_ear_filter,
                                           shm_auditory_filtbank,
                                           shm_signal_segment_blocks,
                                           shm_signal_segment,
                                           shm_downsample,
                                           shm_mod_lowpass,
                                           shm_mov_median,
                                           shm_loud_nonlin,
                                           shm_fluct_weight,
                                           shm_hsa,
                                           shm_round, shm_rms,
                                           shm_in_check)
from tqdm import tqdm
from sottek_hearing_model.filters import weight_A_t
from sottek_hearing_model.plotting_tools import create_figure, show_plot
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
import itertools

# %% Module settings
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = 'Arial'
mpl.rcParams['mathtext.fontset'] = 'stixsans'

plt.rc('font', size=16)  # controls default text sizes
plt.rc('axes', titlesize=22,
       labelsize=22)  # fontsize of the axes title and x and y labels
plt.rc('xtick', labelsize=16)  # fontsize of the tick labels
plt.rc('ytick', labelsize=20)  # fontsize of the tick labels
plt.rc('legend', fontsize=16)  # legend fontsize
plt.rc('figure', titlesize=24)  # fontsize of the figure title

# parallel processing cpu cores
num_cores = max(1, multiprocessing.cpu_count() - 1)  # leave one free core


# %% shm_fluctuation_ecma
def shm_fluctuation_ecma(p, samp_rate_in, axis=0, soundfield='free_frontal',
                         wait_bar=True, out_plot=False, binaural=True,
                         parallel_cores=None):
    """shm_fluctuation_ecma(p, samp_rate_in, axis=0, soundfield='free_frontal',
                            wait_bar=True, out_plot=False, binaural=True,
                            parallel_cores=None)

    Returns fluctuation strength values according to ECMA-418-2:2025 (using
    the Sottek Hearing Model) for input audio signal.

    Parameters
    ----------
    p : 1d or 2d array
        Input signal as single mono or stereo audio (sound pressure)
        signals

    samp_rate_in : integer
        Sample rate (frequency) of the input signal(s)

    axis : integer (0 or 1, default: 0)
        Time axis along which to calculate the fluctuation strength

    soundfield : keyword string (default: 'free_frontal')
        Determines whether the 'free_frontal' or 'diffuse' field stages
        are applied in the outer-middle ear filter, or 'no_outer' uses
        only the middle ear stage, or 'no_ear' omits ear filtering.
        note: these last two options are beyond the scope of the
        standard, but may be useful if recordings made using
        artificial outer/middle ear are to be processed using the
        specific recorded responses.

    wait_bar : keyword string (default: true)
        Determines whether a progress bar displays during processing
        (set wait_bar to false for doing multi-file parallel calculations).

    out_plot : Boolean (default: False)
        Flag indicating whether to generate a figure from the output
        (set out_plot to false for doing multi-file parallel calculations).

    binaural : Boolean (default: True)
        Flag indicating whether to output combined binaural fluctuation
        strength for stereo input signal.

    parallel_cores : integer or None (default: None)
        Number of parallel cores to use for processing
        (if None, the number of cores is automatically determined based
        on the number of available CPU cores; for multicore systems,
        1 core is always left free, to avoid system freeze.
        If 1, parallel processing is not applied).

    Returns
    -------
    fluctuation : dict
        Contains the output.

    fluctuation contains the following outputs:

    spec_fluctuation : 2d or 3d array
        Time-dependent specific fluctuation strength for each critical band
        arranged as [time, bands(, channels)].

    spec_fluctuation_avg : 1d or 2d array
        Time-averaged specific fluctuation strength for each critical band
        arranged as [bands(, channels)].

    fluctuation_t : 1d or 2d array
        Time-dependent overall fluctuation strength
        arranged as [time(, channels)].

    fluctuation90pc : 1d or 2d array
        time-aggregated (90th percentile) overall fluctuation strength
        arranged as [fluctuation strength(, channels)].

    band_centre_freqs : 1d array
        Centre frequencies corresponding with each critical band
        rate.

    time_out : 1d array
        Time (seconds) corresponding with time-dependent outputs.

    soundfield : keyword string
        Identifies the soundfield type applied (the input argument
        soundfield).

    If out_plot=True, a set of plots is returned illustrating the energy
    time-averaged A-weighted sound level, the time-dependent specific and
    overall fluctuation strength, with the latter also indicating the
    time-aggregated value. A set of plots is returned for each input
    channel, with another set for the binaural fluctuation strength, if
    binaural=True. In that case, the indicated sound level corresponds with
    the channel with the highest sound level.

    If binaural=True, a corresponding set of outputs for the binaural
    fluctuation strength are also contained in fluctuation.

    Assumptions
    -----------
    The input signal is calibrated to units of acoustic pressure in Pascals
    (Pa).

    Note
    ----
    Three deliberate departures from the printed text of ECMA-418-2:2025
    are made in this implementation, each documented at the point of use
    (and in shm_subs.py for the HSA subfunctions), because the printed
    forms were found to be internally inconsistent:
    (1) the phase term of Equation 127 uses pi rather than 2*pi, and
    the HSA returns two-sided rather than one-sided line amplitudes
    (Equation 123 versus Equations 159-160);
    (2) Equation 144 omits the printed "- 1" offset;
    (3) the Equation 152 maximum Newton step is 2*10^-1 Hz rather than
    2*10^-4 Hz; and, as an interpretation of Section 9.1.8, a harmonic
    complex of assumed order 2 or 3 must contain a component at its
    fundamental modulation rate.

    """
    # %% Input checks
    p, chans_in, chans = shm_in_check(p, samp_rate_in, axis,
                                      soundfield, wait_bar, out_plot,
                                      binaural, parallel_cores)

    # Check the length of the input data (must be longer than the block used
    # for fluctuation strength segmentation, i.e. > 1.3653 s)
    if p.shape[0] <= 65536/48e3*samp_rate_in:
        raise ValueError("\nInput signal is too short along the specified axis to calculate fluctuation strength (must be longer than 1.3653 s)")

    # assign chansOut
    if chans_in > 1:
        if binaural:
            chans_out = 3
            chans += ["Binaural"]
        else:
            chans_out = chans_in
    else:
        chans_out = chans_in

    # %% Define constants

    signal_t = p.shape[0]/samp_rate_in  # duration of input signal
    # Signal sample rate prescribed to be 48kHz (to be used for resampling),
    # Section 5.1.1 ECMA-418-2:2025 [r_s]
    samp_rate48k = 48e3
    # defined in Section 5.1.4.1 ECMA-418-2:2025 [deltaf(f=0)]
    delta_freq0 = 81.9289
    # Half-overlapping Bark band centre-frequency denominator constant defined
    # in Section 5.1.4.1 ECMA-418-2:2025
    c = 0.1618

    dz = 0.5  # critical band overlap
    # half-overlapping critical band rate scale [z]
    half_bark = np.arange(0.5, 27, dz)
    n_bands = len(half_bark)  # number of critical bands
    # section 5.1.4.1 equation 9 ecma-418-2:2025 [f(z)]
    band_centre_freqs = (delta_freq0/c)*np.sinh(c*half_bark)

    # Block and hop sizes Section 9.1.1 ECMA-418-2:2025
    overlap = 0.75  # block overlap proportion
    # block size [s_b]
    block_size = 65536
    # hop size [s_h]
    hop_size = int(((1 - overlap)*block_size))

    # Downsampled block and hop sizes Section 9.1.2 ECMA-418-2:2025
    downsample = 32  # downsampling factor
    samp_rate1500 = samp_rate48k/downsample  # [r~s]
    block_size1500 = int(block_size/downsample)  # [s~b] = 2048
    # DFT resolution Section 9.1.5 [Delta f]
    delta_f1500 = samp_rate1500/block_size1500

    # Determination of envelope analysis windows Section 9.1.3 ECMA-418-2:2025
    num0_win_start = int(block_size1500/32)  # start number of zeros [n_zb], initial value = 64
    num0_win_end = num0_win_start  # end number of zeros [n_ze], initial value
    # default envelope analysis window [w_Elz(ntilde)]
    env_window = np.zeros(block_size1500)
    env_window[num0_win_start:block_size1500 - num0_win_end] = 1
    # moving median filter length (Section 9.1.3.2) = 33
    mov_med_len = int(block_size1500/64 + 1)
    # minimum duration of an interior quieter period (Section 9.1.3.4)
    # [ntilde_zeros,min] = 320
    quiet_zeros_min = int(block_size1500*5/32)
    # constraint on the (0-based) end index of the envelope analysis window
    # (Section 9.1.3.6): ntilde_2 >= s~b/4 - 1 = 511
    win_end_lim = int(block_size1500/4) - 1
    # shift on window indices for linear regression analysis
    # (Section 9.1.3.6) = 10
    lin_reg_idx_shift = int(block_size1500*5/1024)
    # Hilbert-transform distortion margin added when updating window
    # parameters (Section 9.1.3.5) = 64
    hilbert_margin = int(block_size1500/32)

    # Section 9.1.4 - standardised epsilon (substituting for the standard's
    # epsilon_0, the smallest positive double such that 1 + epsilon_0 > 1 -
    # see the Note in shm_hsa for why 1e-12 is used here without loss of
    # accuracy, and for consistency with the package's existing use of
    # epsilon = 1e-12 throughout, e.g. in shm_roughness_ecma.py)
    epsilon = 1e-12

    # Section 9.1.5 - candidate modulation rates for the search for local
    # minima of the HSA error function E_l,z((0,f_i)), f_i = 0.25*2^((i-2)/3)
    # Hz, i = 1,...,16
    mod_rate_initial = 0.25*2**((np.arange(1, 17) - 2)/3)
    phi_e_min = 0.15  # Section 9.1.5 Equation 143 [Phi_Emin]

    # Section 9.1.6 Equation 148 - fluctuation strength band-pass weighting
    # is applied via shm_fluct_weight using band_centre_freqs[z_band] directly

    # Section 9.1.7 - fine tuning (modified damped Newton method) parameters
    # are defined in shm_hsa_block (Equations 149-152)

    # Section 9.1.10 - threshold applied to A(l,z)
    a_threshold = 5.2519

    # Section 5.1.9 Table 3 ECMA-418-2:2025 - loudness threshold in quiet
    # [LTQ(z)]
    loud_thresh = np.array([0.3310, 0.1625, 0.1051, 0.0757, 0.0576, 0.0453,
                            0.0365, 0.0298, 0.0247, 0.0207, 0.0176, 0.0151,
                            0.0131, 0.0115, 0.0103, 0.0093, 0.0086, 0.0081,
                            0.0077, 0.0074, 0.0073, 0.0072, 0.0071, 0.0072,
                            0.0073, 0.0074, 0.0076, 0.0079, 0.0082, 0.0086,
                            0.0092, 0.0100, 0.0109, 0.0122, 0.0138, 0.0157,
                            0.0172, 0.0180, 0.0180, 0.0177, 0.0176, 0.0177,
                            0.0182, 0.0190, 0.0202, 0.0217, 0.0237, 0.0263,
                            0.0296, 0.0339, 0.0398, 0.0485, 0.0622])

    # Output sample rate (Section 9.1.11 ECMA-418-2:2025) [r_s50]
    samp_rate50 = 50

    # Calibration constant, Section 9.1.11 Equation 163 ECMA-418-2:2025 [c_F]
    cal_F = 0.003840572

    # Determine number of cores to use
    if parallel_cores is None:
        # number of cores to use in parallel processing
        n_cores = max(1, min(n_bands, num_cores))

    elif parallel_cores > 1:
        n_cores = max(1, min(n_bands, num_cores, parallel_cores))

    else:
        n_cores = parallel_cores  # no parallel processing

    # %% Signal processing

    # Input pre-processing
    # --------------------
    if samp_rate_in != samp_rate48k:  # Resample signal
        p_re, _ = shm_resample(p, samp_rate_in)
    else:  # don't resample
        p_re = p
    # end of if branch for resampling

    # Input signal samples
    n_samples = p_re.shape[0]

    # Section 9.1.11 - number of output samples at 50 Hz [l_50,last]
    l50_last = int(np.floor(n_samples/samp_rate48k*samp_rate50) + 1)

    # Section 5.1.2 ECMA-418-2:2025 Fade in weighting and zero-padding
    # (only the start is zero-padded)
    pn = shm_pre_proc(p_re, block_size=block_size, hop_size=hop_size,
                      pad_start=True, pad_end=False)

    # Apply outer & middle ear filter
    # -------------------------------
    #
    # Section 5.1.3.2 ECMA-418-2:2025 Outer and middle/inner ear signal filtering
    pn_om = shm_outmid_ear_filter(pn, soundfield)

    # Loop through channels in file
    # -----------------------------
    if wait_bar:
        chan_iter = tqdm(range(chans_in), desc="Channels")
    else:
        chan_iter = range(chans_in)

    spec_fluctuation = np.zeros([l50_last, n_bands, chans_out], order='F')

    for chan in chan_iter:
        # Apply auditory filter bank
        # --------------------------
        # Filter equalised signal using 53 1/2-overlapping Bark filters
        # according to Section 5.1.4.2 ECMA-418-2:2025
        pn_omz = shm_auditory_filtbank(pn_om[:, chan])

        # Note: At this stage, typical computer RAM limits impose a need to
        # loop through the critical bands rather than continue with a
        # parallelised approach, until later downsampling is applied

        # pre-allocate output arrays
        i_start = 0  # index to start segmentation block processing from
        _, n_blocks, _ = shm_signal_segment_blocks(pn_omz[:, 0],
                                                   block_size=block_size,
                                                   overlap=overlap,
                                                   i_start=i_start,
                                                   end_shrink=True)
        envelopes = np.zeros([block_size1500, n_blocks, n_bands], order='F')

        # Segmentation into blocks and envelope extraction
        # ------------------------------------------------
        # Section 5.1.5 ECMA-418-2:2025
        # Section 9.1.2 ECMA-418-2:2025
        # parallel processing loop over critical bands

        if wait_bar:
            band_iter = tqdm(range(n_bands), desc="Envelope extraction")
        else:
            band_iter = range(n_bands)

        if n_cores > 1:
            with ThreadPoolExecutor(max_workers=n_cores) as executor:
                results = executor.map(shm_fluct_envelopes,
                                       band_iter,
                                       itertools.repeat(block_size),
                                       itertools.repeat(overlap),
                                       itertools.repeat(i_start),
                                       itertools.repeat(downsample),
                                       (pn_omz[:, z] for z in band_iter))

            for (z_band, l_blocks_out,
                 band_envelopes) in results:

                envelopes[:, :, z_band] = band_envelopes

        else:  # no parallel processing: run in loop to save memory
            for z_band in band_iter:
                (z_band, l_blocks_out,
                 band_envelopes) = shm_fluct_envelopes(z_band, block_size,
                                                       overlap, i_start,
                                                       downsample,
                                                       pn_omz[:, z_band])

                envelopes[:, :, z_band] = band_envelopes

        # Note: With downsampled envelope signals, a parallelised approach can
        # continue for the window/spectrum determination

        # Determination of envelope analysis windows
        # -------------------------------------------
        # Section 9.1.3 ECMA-418-2:2025

        # Section 9.1.3.2 Envelope smoothing and first weighting
        # moving median filter of length mov_med_len, rounded to 8 decimal
        # places, windowed by the default window, then find the windowed max
        # [pBar(nTilde)_E,l,z] with initial window applied
        env_med_weight = (shm_dimensional(env_window, target_dim=3)
                          * np.round(shm_mov_median(envelopes, mov_med_len,
                                                    axis=0), 8))
        env_med_weight = np.asfortranarray(env_med_weight)
        env_med_wght_max = np.max(env_med_weight, axis=0)  # [p_Emax,l,z]
        # entire-block quieter period flag
        quiet_block_init = env_med_wght_max <= 5e-6

        # Section 9.1.3.3 Further quieter period detection
        quiet_threshold = np.round(0.01*env_med_wght_max, 8)  # [p_Ethr,l,z]
        quiet_periods = env_med_weight < quiet_threshold

        # get the first and last (0-based) indices of non-quiet samples
        # (equivalents of n_zb and s~b - 1 - n_ze respectively)
        non_quiet = ~quiet_periods
        any_non_quiet = np.any(non_quiet, axis=0)
        start_idx = np.argmax(non_quiet, axis=0)
        end_idx = block_size1500 - 1 - np.argmax(non_quiet[::-1, :, :], axis=0)

        # Section 9.1.3.1 default window parameters, broadcast over
        # blocks/bands, and updated number of zeros (Section 9.1.3.3)
        num0_win_start_new = np.full([n_blocks, n_bands], num0_win_start)
        num0_win_end_new = np.full([n_blocks, n_bands], num0_win_end)
        num0_win_start_new[any_non_quiet] = np.maximum(num0_win_start,
                                                       start_idx[any_non_quiet])
        num0_win_end_new[any_non_quiet] = np.maximum(num0_win_end,
                                                     block_size1500 - 1
                                                     - end_idx[any_non_quiet])

        # Section 9.1.3.4-9.1.3.6 - search for the longest interior quieter
        # period, update the window accordingly, and validate (these steps
        # are adaptive per block/band and are therefore looped, consistent
        # with the equivalent adaptive steps in shm_roughness_ecma.py)
        num0_win_start_final = num0_win_start_new.copy()
        num0_win_end_final = num0_win_end_new.copy()
        quiet_block = quiet_block_init.copy()

        if wait_bar:
            band_iter = tqdm(range(n_bands),
                             desc="Envelope windows")
        else:
            band_iter = range(n_bands)

        block_iter = range(n_blocks)

        if n_cores > 1:
            par_tasks = [(z, l) for z, l in itertools.product(band_iter,
                                                             block_iter)
                         if not quiet_block_init[l, z]]
            with ThreadPoolExecutor(max_workers=n_cores) as executor:
                results = executor.map(lambda p:
                                       shm_env_window(p[0], p[1],
                                                      quiet_periods[:, p[1], p[0]],
                                                      env_med_weight[:, p[1], p[0]],
                                                      num0_win_start_new[p[1], p[0]],
                                                      num0_win_end_new[p[1], p[0]],
                                                      block_size1500,
                                                      quiet_zeros_min,
                                                      win_end_lim,
                                                      lin_reg_idx_shift,
                                                      hilbert_margin),
                                       par_tasks)

            for (z_band, l_block,
                 num0_win_start_band_block,
                 num0_win_end_band_block,
                 quiet_block_band_block) in results:

                num0_win_start_final[l_block, z_band] = num0_win_start_band_block
                num0_win_end_final[l_block, z_band] = num0_win_end_band_block
                quiet_block[l_block, z_band] = quiet_block_band_block

        else:  # no parallel processing: run in nested loop to save memory
            for z_band in band_iter:
                for l_block in block_iter:
                    if quiet_block_init[l_block, z_band]:
                        continue  # already flagged as an entire-block quieter period

                    (z_band, l_block,
                     num0_win_start_band_block,
                     num0_win_end_band_block,
                     quiet_block_band_block) = shm_env_window(z_band, l_block,
                                                              quiet_periods[:, l_block, z_band],
                                                              env_med_weight[:, l_block, z_band],
                                                              num0_win_start_new[l_block, z_band],
                                                              num0_win_end_new[l_block, z_band],
                                                              block_size1500,
                                                              quiet_zeros_min,
                                                              win_end_lim,
                                                              lin_reg_idx_shift,
                                                              hilbert_margin)

                    num0_win_start_final[l_block, z_band] = num0_win_start_band_block
                    num0_win_end_final[l_block, z_band] = num0_win_end_band_block
                    quiet_block[l_block, z_band] = quiet_block_band_block

        # Section 9.1.3.6 - quieter-period blocks (including those failing
        # the validity checks) have their window parameters set to s~b/2 and
        # the envelope analysis window set to zero
        num0_win_start_final[quiet_block] = int(block_size1500/2)
        num0_win_end_final[quiet_block] = int(block_size1500/2)

        # envelope analysis window for each block/band [w_E,l,z(ntilde)]
        sample_idx = shm_dimensional(np.arange(block_size1500), target_dim=3)
        env_window_mat = ((sample_idx >= num0_win_start_final[np.newaxis, :, :])
                          & (sample_idx < (block_size1500
                                           - num0_win_end_final[np.newaxis, :, :]))
                          & ~quiet_block[np.newaxis, :, :]).astype(float)

        # High-resolution Spectral Analysis input spectra
        # -------------------------------------------------
        # Section 9.1.4 Equations 121-122 [P_E,l,z(k)], [Phi_E,l,z(k)]
        env_spectra = np.asfortranarray(fft(envelopes*env_window_mat,
                                            n=block_size1500, axis=0))
        env_mag_sq_spectra = np.abs(env_spectra)**2

        # Section 9.1.5 Equation 143 threshold [max(0.001*Phi_E,l,z(0), Phi_Emin)]
        mod_spec_criterion = np.maximum(0.001*env_mag_sq_spectra[0, :, :],
                                        phi_e_min)

        # High-resolution Spectral Analysis pipeline (Sections 9.1.4-9.1.10)
        # -------------------------------------------------------------------
        #
        # This part of the calculation is inherently adaptive (a
        # variable-dimension linear system is solved per block and per band,
        # with an iterative fine-tuning optimisation and harmonic-order
        # search), so - consistent with the equivalent adaptive steps in
        # shm_roughness_ecma.py (Sections 7.1.5.1 and 7.1.5.3) - it is
        # implemented with nested loops over bands and blocks rather than
        # parallelised across the whole array.

        a_hat_mat = np.zeros([n_blocks, n_bands], order='F')  # [Ahat(l,z)], Equation 157
        power_sum_mat = np.zeros([n_blocks, n_bands], order='F')  # [phat_0^2 + 2*sum(A_i)], Equation 159 denominator (raw, unweighted power)
        n_hsa_mat = np.zeros([n_blocks, n_bands], order='F')  # [N'_HSA(l,z)], Equation 161
        fund_rate_mat = np.zeros([n_blocks, n_bands], order='F')  # [f_1(l,z)], Equation 156

        if wait_bar:
            band_iter = tqdm(range(n_bands),
                             desc="High-resolution spectral analysis")
        else:
            band_iter = range(n_bands)

        block_iter = range(n_blocks)

        if n_cores > 1:
            par_tasks = [(z, l) for z, l in itertools.product(band_iter,
                                                             block_iter)
                         if not quiet_block[l, z]]
            with ThreadPoolExecutor(max_workers=n_cores) as executor:
                results = executor.map(lambda p:
                                       shm_hsa_block(p[0], p[1],
                                                     env_spectra[:, p[1], p[0]],
                                                     env_mag_sq_spectra[:49, p[1], p[0]],
                                                     num0_win_start_final[p[1], p[0]],
                                                     num0_win_end_final[p[1], p[0]],
                                                     mod_spec_criterion[p[1], p[0]],
                                                     band_centre_freqs[p[0]],
                                                     loud_thresh[p[0]],
                                                     block_size1500,
                                                     samp_rate1500,
                                                     mod_rate_initial,
                                                     epsilon),
                                       par_tasks)

            for (z_band, l_block,
                 a_hat_band_block,
                 power_sum_band_block,
                 n_hsa_band_block,
                 fund_rate_band_block) in results:

                a_hat_mat[l_block, z_band] = a_hat_band_block
                power_sum_mat[l_block, z_band] = power_sum_band_block
                n_hsa_mat[l_block, z_band] = n_hsa_band_block
                fund_rate_mat[l_block, z_band] = fund_rate_band_block

        else:  # no parallel processing: run in nested loop to save memory
            for z_band in band_iter:
                for l_block in block_iter:
                    if quiet_block[l_block, z_band]:
                        continue  # A(l,z) remains zero for this block (quieter period)

                    (z_band, l_block,
                     a_hat_band_block,
                     power_sum_band_block,
                     n_hsa_band_block,
                     fund_rate_band_block) = shm_hsa_block(z_band, l_block,
                                                           env_spectra[:, l_block, z_band],
                                                           env_mag_sq_spectra[:49, l_block, z_band],
                                                           num0_win_start_final[l_block, z_band],
                                                           num0_win_end_final[l_block, z_band],
                                                           mod_spec_criterion[l_block, z_band],
                                                           band_centre_freqs[z_band],
                                                           loud_thresh[z_band],
                                                           block_size1500,
                                                           samp_rate1500,
                                                           mod_rate_initial,
                                                           epsilon)

                    a_hat_mat[l_block, z_band] = a_hat_band_block
                    power_sum_mat[l_block, z_band] = power_sum_band_block
                    n_hsa_mat[l_block, z_band] = n_hsa_band_block
                    fund_rate_mat[l_block, z_band] = fund_rate_band_block

        # Section 9.1.10 Equation 159 - scaling with HSA-based loudness,
        # completed here (requires max_z(N'_HSA(l,z)) across all bands for
        # each block, hence deferred to this second pass)
        n_hsa_max = np.max(n_hsa_mat, axis=1)  # [max_z(N'_HSA(l,z))], per block
        a_lz = (a_hat_mat*(n_hsa_mat**2)/(n_hsa_max[:, np.newaxis] + epsilon)
                / (power_sum_mat + epsilon)*block_size1500)

        # threshold: values of A(l,z) below 5.2519 are set to zero, and the
        # corresponding fundamental modulation rates are also zeroed
        below_threshold = a_lz < a_threshold
        a_lz[below_threshold] = 0
        fund_rate_mat[below_threshold] = 0  # retained for completeness; not used further below

        # Time-dependent specific fluctuation strength
        # ---------------------------------------------
        # Section 9.1.11 ECMA-418-2:2025

        # interpolation to 50 Hz sampling rate
        # Section 9.1.11 Equation 162 [t(l)] (via l_blocks_out from
        # shm_signal_segment with end_shrink=True, which already places the
        # final block at the true end of the signal, matching Equation 162's
        # l = l_last special case)
        t = l_blocks_out/samp_rate48k
        t50 = np.linspace(0, signal_t, l50_last)

        spec_fluct_est = np.zeros([l50_last, n_bands], order='F')
        for z_band in range(n_bands):
            interpolator = PchipInterpolator(t, a_lz[:, z_band], axis=0)
            spec_fluct_est[:, z_band] = interpolator(t50)
        # end of for loop for interpolation
        spec_fluct_est[spec_fluct_est < 0] = 0  # [F'_est(l_50,z)]

        # Section 9.1.11 Equations 166-167 [Ftilde'_est(l_50)], [Fbar'_est(l_50)]
        spec_fluct_est_rms = np.sqrt(np.mean(spec_fluct_est**2, axis=1))
        spec_fluct_est_avg = np.mean(spec_fluct_est, axis=1)

        # Section 9.1.11 Equation 165 [Bhat(l_50)], smoothed with a moving
        # median filter of length 71 to give [B(l_50)]
        exp_bhat_l50 = np.zeros(spec_fluct_est_avg.size)
        mask = spec_fluct_est_avg != 0
        exp_bhat_l50[mask] = spec_fluct_est_rms[mask]/spec_fluct_est_avg[mask]
        exp_b_l50 = shm_mov_median(exp_bhat_l50, 71)

        # Section 9.1.11 Equation 164 [E(l_50)]
        exp_l50 = 0.37106*(np.tanh(1.6407*(exp_b_l50 - 2.5804)) + 1)*0.5 + 0.58449

        # Section 9.1.11 Equation 163 [Fhat'(l_50,z)]
        spec_fluct_est_tform = cal_F*(spec_fluct_est.T**exp_l50).T

        # Section 9.1.11 Equation 168 [F'(l_50,z)] - single time constant
        # low-pass filter of order one (equal rise/fall time constants, so
        # shm_mod_lowpass - which implements the structurally identical
        # Equations 109-110 for roughness - is reused directly)
        tau = 0.75
        spec_fluctuation[:, :, chan] = shm_mod_lowpass(spec_fluct_est_tform,
                                                       samp_rate50, tau, tau)

    # end of for loop over channels

    # Binaural fluctuation strength
    # Section 9.1.15 ECMA-418-2:2025 [F'_B(l_50,z)]
    if chans_in == 2 and binaural:
        # Equation 170
        spec_fluctuation[:, :, 2] = np.sqrt(np.sum(spec_fluctuation[:, :, 0:2]**2,
                                                   axis=2)/2)
    # end of if branch for combined binaural

    # Section 9.1.12 ECMA-418-2:2025
    # Time-averaged specific fluctuation strength [F'(z)], discarding
    # 0 <= l50 <= 35
    spec_fluctuation_avg = np.mean(spec_fluctuation[36:, :, :], axis=0)

    # Section 9.1.13 ECMA-418-2:2025
    # Time-dependent fluctuation strength Equation 169 [F(l_50)]
    fluctuation_t = np.sum(spec_fluctuation*dz, axis=1)

    # ensure channel dimension is retained (to ease plotting)
    if chans_out == 1:
        spec_fluctuation_avg = shm_dimensional(spec_fluctuation_avg)
        fluctuation_t = shm_dimensional(fluctuation_t)

    # Section 9.1.14 ECMA-418-2:2025
    # Overall fluctuation strength [F], discarding 0 <= l50 <= 35
    fluctuation90pc = np.percentile(fluctuation_t[36:, :], 90, axis=0)

    # time (s) corresponding with results output [t]
    time_out = np.arange(0, (spec_fluctuation.shape[0]))/samp_rate50

    # %% Output plotting

    # Plot figures
    # ------------
    if out_plot:
        # Plot results
        for chan in range(chans_out):
            # Plot results
            cmap_inferno = mpl.colormaps['inferno']
            chan_lab = chans[chan]
            fig, axs = create_figure(nrows=2, ncols=1, figsize=[10.5, 7.5],
                                     layout='constrained')

            ax1 = axs[0]
            pmesh = ax1.pcolormesh(time_out, band_centre_freqs,
                                   np.swapaxes(spec_fluctuation[:, :, chan], 0, 1),
                                   cmap=cmap_inferno,
                                   vmin=0,
                                   vmax=max(1e-6,
                                            np.ceil(np.max(spec_fluctuation[:, :,
                                                                            chan])*500)/500),
                                   shading='gouraud')
            ax1.set(xlim=[time_out[1],
                          time_out[-1] + (time_out[1] - time_out[0])],
                    xlabel="Time, s",
                    ylim=[band_centre_freqs[0], band_centre_freqs[-1]],
                    yscale='log',
                    yticks=[63, 125, 250, 500, 1e3, 2e3, 4e3, 8e3, 16e3],
                    yticklabels=["63", "125", "250", "500", "1k", "2k", "4k",
                                 "8k", "16k"],
                    ylabel="Frequency, Hz")
            ax1.minorticks_off()
            cbax = ax1.inset_axes([1.05, 0, 0.05, 1])
            fig.colorbar(pmesh, ax=ax1,
                         label=(r"Specific fluctuation strength,"
                                "\n"
                                r"$\mathregular{vacil_{HMS}}/\mathregular{Bark_{HMS}}$"),
                         aspect=10, cax=cbax)

            ax2 = axs[1]
            ax2.plot(time_out, fluctuation90pc[chan]*np.ones(time_out.size),
                     color=cmap_inferno(33/255), linewidth=1, linestyle='dotted',
                     label=("90th-" + "\n" + "percentile"))
            ax2.plot(time_out, fluctuation_t[:, chan],
                     color=cmap_inferno(165/255),
                     linewidth=0.75, label=("Time-" + "\n" + "dependent"))
            ax2.set(xlim=[time_out[0], time_out[-1] + time_out[1] - time_out[0]],
                    xlabel="Time, s",
                    ylabel=(r"Fluctuation strength, $\mathregular{vacil_{HMS}}$"))
            if np.max(fluctuation_t[:, chan]) > 0:
                ax2.set(ylim=[0, 1.1*np.ceil(np.max(fluctuation_t[:, chan])*10)/10])
            ax2.grid(alpha=0.075, linestyle='--')
            ax2.legend(bbox_to_anchor=(1.025, 0.8), loc='upper left', title="Overall")

            # Filter signal to determine A-weighted time-averaged level
            if chan == 2:
                pA = weight_A_t(p_re, fs=samp_rate48k)
                level_Aeq2 = 20*np.log10(shm_rms(pA, axis=0)/2e-5)
                # take the higher channel level as representative (PD ISO/TS
                # 12913-3:2019 Annex D)
                level_Aeq = np.max(level_Aeq2)
                lr = np.argmax(level_Aeq2)
                # identify which channel is higher
                if lr == 0:
                    which_ear = " left ear"
                else:
                    which_ear = " right ear"
                # end of if branch to identify which channel is higher

                chan_lab = chan_lab + which_ear
            else:
                pA = weight_A_t(p_re[:, chan], fs=samp_rate48k)
                level_Aeq = 20*np.log10(shm_rms(pA)/2e-5)

            fig.suptitle(t=(chan_lab + " signal sound pressure level = " +
                            str(shm_round(level_Aeq, 1)) +
                            r" dB $\mathregular{\mathit{L}_{Aeq}}$"))

            show_plot(fig)
        # end of for loop over channels
    # end of if branch for plotting

    # %% Output assignment

    # Discard singleton dimensions
    if chans_out == 1:
        spec_fluctuation = np.squeeze(spec_fluctuation)
        spec_fluctuation_avg = np.squeeze(spec_fluctuation_avg)
        fluctuation_t = np.squeeze(fluctuation_t)
    # end of if branch for singleton dimensions

    # Assign outputs to structure
    fluctuation = {}
    if chans_out == 3:
        fluctuation.update({'spec_fluctuation': spec_fluctuation[:, :, 0:2]})
        fluctuation.update({'spec_fluctuation_avg': spec_fluctuation_avg[:, 0:2]})
        fluctuation.update({'fluctuation_t': fluctuation_t[:, 0:2]})
        fluctuation.update({'fluctuation90pc': fluctuation90pc[0:2]})
        fluctuation.update({'spec_fluctuation_bin': spec_fluctuation[:, :, 2]})
        fluctuation.update({'spec_fluctuation_avg_bin': spec_fluctuation_avg[:, 2]})
        fluctuation.update({'fluctuation_t_bin': fluctuation_t[:, 2]})
        fluctuation.update({'fluctuation90pc_bin': np.array(fluctuation90pc[2])})
        fluctuation.update({'band_centre_freqs': band_centre_freqs})
        fluctuation.update({'time_out': time_out})
        fluctuation.update({'soundfield': soundfield})
    else:
        fluctuation.update({'spec_fluctuation': spec_fluctuation})
        fluctuation.update({'spec_fluctuation_avg': spec_fluctuation_avg})
        fluctuation.update({'fluctuation_t': fluctuation_t})
        fluctuation.update({'fluctuation90pc': fluctuation90pc})
        fluctuation.update({'band_centre_freqs': band_centre_freqs})
        fluctuation.update({'time_out': time_out})
        fluctuation.update({'soundfield': soundfield})

    return fluctuation
# end of shm_fluctuation_ecma function


# %% multiprocessing helper function for envelope extraction
def shm_fluct_envelopes(z_band, block_size, overlap, i_start, downsample,
                        pn_omz_band):
    """shm_fluct_envelopes(z_band, block_size, overlap, i_start, downsample,
                           pn_omz_band)

    Returns segmented, downsampled envelopes for a given critical band.

    Parameters
    ----------
    z_band : int
        Critical band index.

    block_size : int
        Block size for segmentation.

    overlap : float
        Overlap proportion for segmentation.

    i_start : int
        Start index for segmentation.

    downsample : int
        Downsampling factor for the envelopes.

    pn_omz_band : 1D array
        Output of outer/middle ear processing for given critical band.

    Returns
    -------
    z_band : int
        Critical band index.

    l_blocks_out : 1D array
        Time block indices for segmented envelopes.

    band_envelopes : 2D array
        Downsampled envelopes for segmented signal in given critical band.

    """

    # %% Signal processing
    # Segmentation into blocks
    # ------------------------

    # Section 5.1.5 ECMA-418-2:2025
    pn_lz, l_blocks_out = shm_signal_segment(pn_omz_band,
                                             block_size=block_size,
                                             overlap=overlap,
                                             i_start=i_start,
                                             end_shrink=True)

    # Envelope calculation and downsampling
    # -------------------------------------
    # Section 9.1.2 ECMA-418-2:2025
    # magnitude of Hilbert transform with downsample - Equation 119
    # [p(ntilde)_E,l,z]
    band_envelopes = shm_downsample(np.abs(hilbert(pn_lz,
                                                   axis=0)),
                                    downsample=downsample)

    return (z_band, l_blocks_out, band_envelopes)
# end of shm_fluct_envelopes function


# %% multiprocessing helper function for envelope analysis window determination
def shm_env_window(z_band, l_block, quiet_periods_band_block,
                   env_med_weight_band_block, num0_win_start_new,
                   num0_win_end_new, block_size1500, quiet_zeros_min,
                   win_end_lim, lin_reg_idx_shift, hilbert_margin):
    """shm_env_window(z_band, l_block, quiet_periods_band_block,
                      env_med_weight_band_block, num0_win_start_new,
                      num0_win_end_new, block_size1500, quiet_zeros_min,
                      win_end_lim, lin_reg_idx_shift, hilbert_margin)

    Determines the envelope analysis window parameters for a single time
    block and critical band, according to Sections 9.1.3.4 to 9.1.3.6 of
    ECMA-418-2:2025: the longest interior quieter period is identified and
    the window shortened accordingly, and the resulting window is validated.

    Parameters
    ----------
    z_band : int
        Critical band index.

    l_block : int
        Time block index.

    quiet_periods_band_block : 1D Boolean array
        Flags indicating the quieter-period samples of the downsampled
        envelope block (Section 9.1.3.3).

    env_med_weight_band_block : 1D array
        Moving-median-smoothed envelope block, weighted by the default
        analysis window (Section 9.1.3.2).

    num0_win_start_new : int
        Number of zeros at the start of the window after the update of
        Section 9.1.3.3.

    num0_win_end_new : int
        Number of zeros at the end of the window after the update of
        Section 9.1.3.3.

    block_size1500 : int
        Downsampled analysis block size s~b.

    quiet_zeros_min : int
        Minimum duration (samples) of an interior quieter period
        (Section 9.1.3.4).

    win_end_lim : int
        Minimum (0-based) index of the last active window sample
        (Section 9.1.3.6).

    lin_reg_idx_shift : int
        Shift on the window indices for the linear regression analysis
        (Section 9.1.3.6).

    hilbert_margin : int
        Hilbert-transform distortion margin added when updating the window
        parameters (Section 9.1.3.5).

    Returns
    -------
    z_band : int
        Critical band index.

    l_block : int
        Time block index.

    num0_win_start_final : int
        Final number of zeros at the start of the window [n_zb,l,z].

    num0_win_end_final : int
        Final number of zeros at the end of the window [n_ze,l,z].

    quiet_block : Boolean
        Flag indicating that the block is to be treated as a quieter
        period (the window validity checks of Section 9.1.3.6 failed).

    """

    # %% Signal processing

    num0_win_start_final = int(num0_win_start_new)
    num0_win_end_final = int(num0_win_end_new)

    # 0-based indices of the first and last samples of the updated interval
    start_idx_new = num0_win_start_final
    end_idx_new = block_size1500 - 1 - num0_win_end_final

    # assign updated interval
    quiet_periods_new = quiet_periods_band_block[start_idx_new:end_idx_new + 1]

    # pad to identify start and end of quieter periods within the updated
    # interval
    quiet_periods_pad = np.concatenate(([0], quiet_periods_new.astype(int),
                                        [0]))

    # (0-based) indices within the updated interval at which each quieter
    # period starts
    quiet_period_starts_new = np.flatnonzero(np.diff(quiet_periods_pad[:-1]) == 1)
    quiet_period_ends_new = np.flatnonzero(np.diff(quiet_periods_pad[1:]) == -1)

    quiet_period_lengths = quiet_period_ends_new - quiet_period_starts_new + 1

    # Section 9.1.3.4 - minimum length criterion mask (a quieter period
    # duration must be STRICTLY greater than quiet_zeros_min, per the
    # standard's ">" in this clause)
    mask = quiet_period_lengths > quiet_zeros_min
    if np.any(mask):
        quiet_period_starts_new = quiet_period_starts_new[mask]
        quiet_period_ends_new = quiet_period_ends_new[mask]
        quiet_period_lengths = quiet_period_lengths[mask]

        qp_longest_idx = np.argmax(quiet_period_lengths)

        # get the (0-based) start and end indices for the longest quieter
        # period within the block, adjusting for the updated start index
        # [n~_qpmb,l,z], [n~_qpme,l,z]
        quiet_period_starts_max = quiet_period_starts_new[qp_longest_idx] + start_idx_new
        quiet_period_ends_max = quiet_period_ends_new[qp_longest_idx] + start_idx_new

        # Section 9.1.3.5 - update of the analysis window parameters. The
        # standard's condition (using 0-based indices) is:
        #   (n~qpmb - (nzb + 64)) > ((s~b - 1 - nze - 64) - n~qpme)
        # which simplifies (the "+/-64" terms cancel) to a direct comparison
        # of the left-hand and right-hand candidate active-window lengths:
        left_len = quiet_period_starts_max - num0_win_start_new
        right_len = (block_size1500 - 1 - num0_win_end_new) - quiet_period_ends_max

        if left_len > right_len:
            # keep the left (longer) part: update n_ze only
            num0_win_end_final = block_size1500 - 1 - quiet_period_starts_max + hilbert_margin
        else:
            # keep the right (longer) part: update n_zb only
            num0_win_start_final = quiet_period_ends_max + hilbert_margin
        # end of if branch to determine which window end is modified
    # end of if branch for minimum quieter period length

    # (0-based) indices of the first and last samples of the final window
    start_idx_new_final = num0_win_start_final
    end_idx_new_final = block_size1500 - 1 - num0_win_end_final

    # Section 9.1.3.6 window interval validity checks
    # length of window, window end far enough into the block, and relative
    # standard deviation of a linear regression fit
    if ((end_idx_new_final - start_idx_new_final + 1 >= quiet_zeros_min)
            and (end_idx_new_final >= win_end_lim)):
        idx_range = np.arange(start_idx_new_final + lin_reg_idx_shift,
                              end_idx_new_final - lin_reg_idx_shift + 1)
        env_range = env_med_weight_band_block[idx_range]
        lin_reg = np.polyfit(idx_range, env_range, 1)
        pred = np.polyval(lin_reg, idx_range)
        resid = pred - env_range
        std_resid = np.std(resid, ddof=1)
        mean_val = np.mean(env_range)
        lin_reg_std_chk = (mean_val != 0) and (std_resid/np.abs(mean_val) >= 0.1/100)
        quiet_block = not lin_reg_std_chk
    else:
        quiet_block = True

    return (z_band, l_block, num0_win_start_final, num0_win_end_final,
            quiet_block)
# end of shm_env_window function


# %% multiprocessing helper function for the HSA pipeline
def shm_hsa_block(z_band, l_block, spectrum_e, spectrum_phi, n_zeros_start,
                  n_zeros_end, mod_spec_criterion, band_centre_freq,
                  loud_thresh, block_size1500, samp_rate1500, mod_rate_initial,
                  epsilon=1e-12):
    """shm_hsa_block(z_band, l_block, spectrum_e, spectrum_phi, n_zeros_start,
                     n_zeros_end, mod_spec_criterion, band_centre_freq,
                     loud_thresh, block_size1500, samp_rate1500,
                     mod_rate_initial, epsilon=1e-12)

    Runs the High-resolution Spectral Analysis (HSA) pipeline of Sections
    9.1.4 to 9.1.10 of ECMA-418-2:2025 for a single time block and critical
    band: candidate modulation rate search (Section 9.1.5), amplitude
    weighting (Section 9.1.6), fine tuning of the dominant modulation rate
    (Section 9.1.7), harmonic analysis (Section 9.1.8), weighting of the
    harmonic complex (Section 9.1.9) and HSA-based loudness (Section 9.1.10).

    Parameters
    ----------
    z_band : int
        Critical band index.

    l_block : int
        Time block index.

    spectrum_e : 1D complex array
        The s~b-point complex DFT spectrum P_E,l,z(k) of the windowed,
        downsampled envelope block (Section 9.1.4 Equation 121).

    spectrum_phi : 1D array
        The envelope power spectrum Phi_E,l,z(k) for k = 0, ..., 48
        (Section 9.1.4 Equation 122).

    n_zeros_start : int
        Number of zeros at the start of the envelope analysis window
        [n_zb,l,z].

    n_zeros_end : int
        Number of zeros at the end of the envelope analysis window
        [n_ze,l,z].

    mod_spec_criterion : float
        Threshold for the local maxima of the envelope power spectrum
        (Section 9.1.5 Equation 143).

    band_centre_freq : float
        Critical band centre frequency F(z) [Hz].

    loud_thresh : float
        Loudness threshold in quiet for the critical band [LTQ(z)]
        (Section 5.1.9 Table 3).

    block_size1500 : int
        Downsampled analysis block size s~b.

    samp_rate1500 : float
        Downsampled analysis sample rate r~s.

    mod_rate_initial : 1D array
        Candidate modulation rates f_i for the search for local minima of
        the HSA error function (Section 9.1.5).

    epsilon : float (default: 1e-12)
        Small constant substituted for the standard's epsilon_0 (see
        shm_hsa).

    Returns
    -------
    z_band : int
        Critical band index.

    l_block : int
        Time block index.

    a_hat : float
        Weighted sum of the harmonic complex [Ahat(l,z)] (Equation 157);
        zero if no modulation is retained for the block.

    power_sum : float
        Mean-square power of the harmonic complex [phat_0^2 + 2*sum(A_i)]
        (Equations 159-160); zero if no modulation is retained.

    n_hsa : float
        HSA-based loudness [N'_HSA(l,z)] (Equation 161); zero if no
        modulation is retained.

    fund_rate : float
        Fundamental modulation rate [f_1(l,z)] (Equation 156); zero if no
        modulation is retained.

    """

    # %% Define constants

    # DFT resolution Section 9.1.5 [Delta f]
    delta_f1500 = samp_rate1500/block_size1500

    # Section 9.1.7 - fine tuning (modified damped Newton method) parameters
    newton_dx = 1e-5  # finite-difference step [Delta x]
    newton_max_it = 40  # Equation 152 maximum number of iterations
    # Equation 152 maximum step size [Hz]
    #
    # CORRECTION relative to the printed standard: Equation 152 as typeset in
    # ECMA-418-2:2025 caps the Newton step at 2*10^-4 Hz. With the 1/4 damping
    # factor and the 40-iteration limit of Equation 152, the fine tuning can
    # then move the modulation rate by at most 40*0.25*2e-4 = 0.002 Hz, which
    # (i) is far smaller than the resolution of the initial estimates (Delta f
    # = 0.732 Hz for Equation 144, and the 1/3-octave grid of the f_min
    # candidates), so the optimisation almost never converges and simply
    # stops after 40 cap-limited steps, and (ii) makes the rejection test
    # |f_c,1,opt - f_c,imax| > 1.25*Delta f = 0.92 Hz in Section 9.1.7
    # unreachable. Diagnostics on real recordings confirmed the cap is active
    # in 77-87 % of all blocks (fine-tuned rates clustering at exactly
    # f_i +/- 0.002 Hz). With the cap raised to 2*10^-1 Hz the optimisation
    # converges, the rejection test becomes meaningful (maximum travel 2 Hz),
    # and the agreement of the time-dependent specific fluctuation strength
    # with reference results improves substantially on complex recordings
    # (e.g. band-spectrum relative error reduced by about one third),
    # with no effect on the calibration sinusoids (whose 4 Hz rate lies
    # exactly on the candidate grid). The printed value is therefore treated
    # as a typographical error in the exponent. As-written: newton_step_lim = 2e-4
    newton_step_lim = 2e-1
    newton_conv_tol = 1e-7  # Equation 152 convergence tolerance [Hz]
    newton_reject_tol = 1.25*delta_f1500  # Section 9.1.7 rejection tolerance

    # default outputs (no modulation retained for this block)
    a_hat = 0.0
    power_sum = 0.0
    n_hsa = 0.0
    fund_rate = 0.0

    # %% Signal processing

    # Section 9.1.5 Stage 1 - local maxima of Phi_E,l,z(k) for k = 1,...,47
    # (k = 0 and k = 48 cannot be local maxima given the bounded search range,
    # consistent with the need for both neighbours k-1 and k+1 in Equation
    # 144)
    k_locs, _ = find_peaks(spectrum_phi)
    phi_pks = spectrum_phi[k_locs]
    k_locs = k_locs[phi_pks >= mod_spec_criterion]  # 0-based peak indices
    # Section 9.1.5 - number of local maxima cannot exceed 24; this is
    # guaranteed by construction (at most floor(47/2) alternating local
    # maxima are possible over 47 interior points) and is not separately
    # enforced here.

    fp_candidates = np.zeros(k_locs.size)
    for i_pk in range(k_locs.size):
        k0 = k_locs[i_pk]  # 0-based peak index
        j_idx = np.arange(k0 - 1, k0 + 2)  # 0-based neighbour indices
        phi_neighbours = spectrum_phi[j_idx]
        # Section 9.1.5 Equation 144 [f_p,i(l,z)]
        #
        # CORRECTION relative to the printed standard: Equation 144 as
        # typeset in ECMA-418-2:2025 includes a "- 1" term inside the outer
        # brackets, i.e. fp,i = (centroid - 1)*Delta_f. Applying it produces
        # a systematic bias of very close to one full DFT bin (Delta_f)
        # below the true frequency, confirmed numerically across multiple
        # independent single-tone test cases (errors of -0.73 to -0.85 Hz,
        # i.e. essentially -Delta_f, versus +0.01 to -0.12 Hz - a small
        # fraction of one bin, consistent with a standard power-weighted
        # three-point centroid interpolator - with the "- 1" term removed).
        # The weighted centroid of the bin indices themselves (without an
        # additional offset) is the conventional and correct form of this
        # estimator, matching the analogous (unbiased) refinement step used
        # by Equations 73-76 for roughness. Removing this term also resolved
        # a >90% amplitude recovery error for a secondary (non-dominant)
        # spectral line in a two-tone synthetic test, which the biased
        # estimate could push far enough from the true frequency to
        # substantially corrupt that line's HSA fit.
        # As-written: fp_candidates[i_pk] = (np.sum(j_idx*phi_neighbours)/np.sum(phi_neighbours) - 1)*delta_f1500
        fp_candidates[i_pk] = (np.sum(j_idx*phi_neighbours)/np.sum(phi_neighbours))*delta_f1500

    # Section 9.1.5 Stage 2 - local minima of the HSA error function
    # E_l,z((0,f_i)) over the 16 log-spaced candidates. Only interior points
    # (i = 2,...,15) can be local minima; this is what the standard's
    # unqualified "local minima" means for a bounded, ordered set of
    # candidates.
    err_hsa = np.zeros(mod_rate_initial.size)
    for i_cand in range(mod_rate_initial.size):
        _, err_hsa[i_cand] = shm_hsa(mod_rate_initial[i_cand], spectrum_e,
                                     block_size1500, samp_rate1500,
                                     n_zeros_start, n_zeros_end, epsilon)
    is_local_min = np.zeros(mod_rate_initial.size, dtype=bool)
    is_local_min[1:-1] = ((err_hsa[1:-1] < err_hsa[:-2])
                          & (err_hsa[1:-1] < err_hsa[2:]))

    if not np.any(is_local_min):
        if fp_candidates.size == 0:
            # Section 9.1.5 - no local maximum and no local minimum: no
            # modulation in this block
            return (z_band, l_block, a_hat, power_sum, n_hsa, fund_rate)
        # Section 9.1.5 - no local minimum: use all local maxima
        fc_final = np.sort(fp_candidates)
        p_hat_all, _ = shm_hsa(fc_final, spectrum_e, block_size1500,
                               samp_rate1500, n_zeros_start, n_zeros_end,
                               epsilon)
    else:
        err_candidates = err_hsa[is_local_min]
        freq_candidates = mod_rate_initial[is_local_min]
        f_min = freq_candidates[np.argmin(err_candidates)]

        # Section 9.1.5 Equation 145 - duplicate detection
        id_dup = np.abs(f_min - fp_candidates) < 1.25*delta_f1500

        if fp_candidates.size == 0:
            fc_final = np.array([f_min])
            p_hat_all, _ = shm_hsa(fc_final, spectrum_e, block_size1500,
                                   samp_rate1500, n_zeros_start, n_zeros_end,
                                   epsilon)
        elif not np.any(id_dup):
            fc_final = np.sort(np.append(fp_candidates, f_min))
            p_hat_all, _ = shm_hsa(fc_final, spectrum_e, block_size1500,
                                   samp_rate1500, n_zeros_start, n_zeros_end,
                                   epsilon)
        else:
            # Case I: f_min plus all local maxima except duplicates
            fc_case_i = np.sort(np.append(fp_candidates[~id_dup], f_min))
            p_hat_case_i, err_case_i = shm_hsa(fc_case_i, spectrum_e,
                                               block_size1500, samp_rate1500,
                                               n_zeros_start, n_zeros_end,
                                               epsilon)

            # Case II: all local maxima only
            fc_case_ii = np.sort(fp_candidates)
            p_hat_case_ii, err_case_ii = shm_hsa(fc_case_ii, spectrum_e,
                                                 block_size1500, samp_rate1500,
                                                 n_zeros_start, n_zeros_end,
                                                 epsilon)

            if err_case_i <= err_case_ii:
                fc_final = fc_case_i
                p_hat_all = p_hat_case_i
            else:
                fc_final = fc_case_ii
                p_hat_all = p_hat_case_ii
        # end of if branch for candidate selection
    # end of if branch for local minima

    # Section 9.1.5 Equation 146 - amplitude threshold (raw, unweighted power
    # A_i(l,z) = |P_HSA,i|^2; strictly greater than, per the standard)
    # Note: shm_hsa returns TWO-SIDED spectral line amplitudes (half the
    # cosine amplitude of each envelope component), so that
    # phat_0^2 + 2*sum(A_i) in Equations 159-160 is the mean-square power of
    # the harmonic complex - see the Note in shm_hsa regarding Equation 123
    # and footnote 46.
    a_raw = np.abs(p_hat_all[1:])**2
    keep_final = a_raw > 0.05*np.max(a_raw)
    if not np.any(keep_final):
        return (z_band, l_block, a_hat, power_sum, n_hsa, fund_rate)
    fc_survive = fc_final[keep_final]
    a_raw_survive = a_raw[keep_final]

    # Section 9.1.6 Equations 147-148 - weighted power spectrum and dominant
    # candidate [Atilde_i(l,z)], [i_max]
    wlh = shm_fluct_weight(fc_survive, band_centre_freq)
    a_tilde_survive = a_raw_survive*wlh
    i_max = np.argmax(a_tilde_survive)

    # Section 9.1.7 Equations 149-152 - fine tuning of the dominant
    # modulation rate (modified damped Newton method)
    x0 = fc_survive[i_max]
    xk = x0
    for k_iter in range(newton_max_it):
        _, e_mid = shm_hsa(xk, spectrum_e, block_size1500, samp_rate1500,
                           n_zeros_start, n_zeros_end, epsilon)
        _, e_plus = shm_hsa(xk + newton_dx, spectrum_e, block_size1500,
                            samp_rate1500, n_zeros_start, n_zeros_end, epsilon)
        _, e_minus = shm_hsa(xk - newton_dx, spectrum_e, block_size1500,
                             samp_rate1500, n_zeros_start, n_zeros_end, epsilon)

        d_e = (e_plus - e_minus)/(2*newton_dx)  # Equation 149
        d2_e = (e_plus - 2*e_mid + e_minus)/newton_dx**2  # Equation 150

        # Equation 152
        delta_x = 0.25*np.sign(d_e)*min(np.abs(d_e)/(np.abs(d2_e) + epsilon),
                                        newton_step_lim)
        xk = xk - delta_x  # Equation 151

        if np.abs(delta_x) <= newton_conv_tol:
            break
    # end of for loop for Newton iterations
    fc_opt = xk

    if np.abs(fc_opt - x0) > newton_reject_tol:
        fc_opt = x0  # optimisation rejected, retain original estimate
    else:
        # Section 9.1.7 - replace the modulation rate of the maximum with the
        # fine-tuned value and update the corresponding spectral component of
        # P_HSA and Atilde_imax accordingly (constant part plus one spectral
        # line pair, as used by the optimisation)
        fc_survive[i_max] = fc_opt
        p_hat_opt, _ = shm_hsa(fc_opt, spectrum_e, block_size1500,
                               samp_rate1500, n_zeros_start, n_zeros_end,
                               epsilon)
        a_raw_survive[i_max] = np.abs(p_hat_opt[1])**2
        a_tilde_survive[i_max] = (a_raw_survive[i_max]
                                  * shm_fluct_weight(fc_opt, band_centre_freq)[0])
    # end of if branch for fine-tuning rejection

    if fc_opt < 0.125:
        return (z_band, l_block, a_hat, power_sum, n_hsa, fund_rate)  # Section 9.1.7 - modulation discarded

    # Section 9.1.8 Equations 153-156 - harmonic analysis
    # Test assumed orders o = 1, 2, 3 of fc_opt
    best_energy = -np.inf
    best_iset = None
    best_ratios = None
    best_order = 0
    for order in range(1, 4):
        fc1o = fc_opt/order
        # Equation 153 (np.floor(x + 0.5) reproduces the standard's rounding
        # to the nearest integer, half away from zero, for the non-negative
        # ratios here)
        ratios = np.floor(fc_survive/fc1o + 0.5)
        ratios[ratios > 5] = 0  # ratios greater than 5 excluded
        valid_ratio = ratios > 0
        tol_check = np.zeros(fc_survive.size, dtype=bool)
        # Equation 154
        tol_check[valid_ratio] = np.abs(fc_survive[valid_ratio]
                                        / (ratios[valid_ratio]*fc1o) - 1) < 0.04
        # Section 9.1.8 - a "harmonic complex with fundamental modulation
        # rate f_c,1,o" (Equation 154) is only taken to exist if one of the
        # components is itself at that fundamental (integer ratio R = 1
        # within the 4 % tolerance). For o = 1 this is always satisfied by
        # f_c,1,opt; for o = 2, 3 it requires a component near f_c,1,opt/o.
        # INTERPRETATION NOTE: the printed text does not state this
        # explicitly, but it mirrors the roughness procedure of Section
        # 7.1.5.3 (Equations 88-91), where every candidate fundamental is
        # itself one of the detected components, and without it the o = 2, 3
        # index sets (which admit all half- and third-integer multiples of
        # f_c,1,opt) almost always accumulate more energy than the o = 1 set
        # purely by admitting more members. On complex recordings this
        # reading, combined with the Equation 152 step-cap correction above,
        # reduced the relative error of the time-dependent specific
        # fluctuation strength against reference results by about one third
        # to one half, whereas testing only o = 1, dropping the harmonic
        # complex, or dropping w_bw all made agreement worse.
        if not np.any(tol_check) or not np.any(tol_check & (ratios == 1)):
            continue
        energy_order = np.sum(a_tilde_survive[tol_check])  # Equation 155
        if energy_order > best_energy:
            best_energy = energy_order
            best_iset = tol_check
            best_ratios = ratios
            best_order = order
    # end of for loop over assumed orders

    if best_iset is None:
        return (z_band, l_block, a_hat, power_sum, n_hsa, fund_rate)

    # Section 9.1.8 - correct the modulation rates of the retained components
    # to exact integer multiples of the fundamental (the component already
    # equal to fc_opt is left numerically unchanged by this correction, since
    # its own ratio is exactly best_order by construction)
    fc1_fund = fc_opt/best_order  # Equation 156 [f_1(l,z)]
    fc_harm_corrected = best_ratios[best_iset]*fc1_fund
    n_harm = fc_harm_corrected.size

    # Section 9.1.8 - re-run the HSA individually (constant part plus one
    # spectral line) for each retained harmonic, and average the resulting
    # constant-part estimates
    p0_estimates = np.zeros(n_harm)
    a_raw_harm = np.zeros(n_harm)
    for i_harm in range(n_harm):
        p_hat_harm, _ = shm_hsa(fc_harm_corrected[i_harm], spectrum_e,
                                block_size1500, samp_rate1500, n_zeros_start,
                                n_zeros_end, epsilon)
        p0_estimates[i_harm] = np.real(p_hat_harm[0])
        a_raw_harm[i_harm] = np.abs(p_hat_harm[1])**2
    wlh_harm = shm_fluct_weight(fc_harm_corrected, band_centre_freq)
    a_tilde_harm = a_raw_harm*wlh_harm
    p0_final = np.mean(p0_estimates)

    # Section 9.1.9 Equations 157-158 - weighting the sum of the harmonic
    # complex
    sum_a_tilde = np.sum(a_tilde_harm)
    cog = np.sum(fc_harm_corrected*a_tilde_harm)/(sum_a_tilde + epsilon)  # centre of gravity [Hz]
    w_bw = 1 + 0.79577*np.abs(cog - fc_opt)**0.43461  # Equation 158
    a_hat = w_bw*sum_a_tilde  # Equation 157 [Ahat(l,z)]

    # Section 9.1.10 Equations 159-161 - power of the harmonic complex (RAW,
    # unweighted amplitudes - see the Note in the HSA explanation regarding
    # A_i vs Atilde_i) and HSA-based loudness
    power_sum = p0_final**2 + 2*np.sum(a_raw_harm)  # Equation 159/160 denominator & argument
    p_rms_hsa = np.sqrt(0.5*max(0.0, power_sum))
    n_tilde_hsa = shm_loud_nonlin(p_rms_hsa)  # Equation 160
    if n_tilde_hsa >= loud_thresh:
        n_hsa = n_tilde_hsa - loud_thresh  # Equation 161
    else:
        n_hsa = 0.0

    fund_rate = fc1_fund

    return (z_band, l_block, a_hat, power_sum, n_hsa, fund_rate)
# end of shm_hsa_block function
