# -*- coding: utf-8 -*-
# %% Preamble
"""
shm_roughness_ecma.py
---------------------

Returns roughness values according to ECMA-418-2:2025 using the Sottek Hearing
Model) for an input calibrated single mono or single stereo audio (sound
pressure) time-series signal, p.

Requirements
------------
numpy
scipy
matplotlib
tqdm

Functions
---------

shm_roughness_ecma : This is the main roughness function, which implements section
                     7 of ECMA-418-2:2025, and returns a dict containing
                     the roughness results as numpy arrays.

Ownership and Quality Assurance
-------------------------------
Author: Mike JB Lotinga (m.j.lotinga@edu.salford.ac.uk)
Institution: University of Salford

Date created: 29/05/2023
Date last modified: 21/10/2025
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

# %% Import block
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import matplotlib as mpl
from matplotlib import pyplot as plt
from scipy.fft import (fft)
from scipy.signal import (hilbert, windows, find_peaks)
from scipy.interpolate import PchipInterpolator
from sottek_hearing_model.shm_subs import (shm_dimensional,
                                           shm_resample,
                                           shm_pre_proc,
                                           shm_outmid_ear_filter,
                                           shm_auditory_filtbank,
                                           shm_signal_segment_blocks,
                                           shm_signal_segment,
                                           shm_basis_loudness,
                                           shm_downsample,
                                           shm_mod_weight,
                                           shm_mod_low_pass,
                                           shm_round, shm_rms,
                                           shm_in_check, shm_signal_segment)
from tqdm import tqdm
from sottek_hearing_model.filters import weight_A_t
from sottek_hearing_model.plotting_tools import create_figure, show_plot

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

# %% shm_roughness_ecma
def shm_roughness_ecma(p, samp_rate_in, axis=0, soundfield='freeFrontal',
                       wait_bar=True, out_plot=False, binaural=True):
    """
    inputs
    ------
    p : 1d or 2d array
        the input signal as single mono or stereo audio (sound pressure)
        signals

    samp_rate_in : integer
                    the sample rate (frequency) of the input signal(s)

    axis : integer (0 or 1, default: 0)
           the time axis along which to calculate the roughness

    soundfield : keyword string (default: 'free_frontal')
                 determines whether the 'free_frontal' or 'diffuse' field stages
                 are applied in the outer-middle ear filter, or 'no_outer' uses
                 only the middle ear stage, or 'no_ear' omits ear filtering.
                 note: these last two options are beyond the scope of the
                 standard, but may be useful if recordings made using
                 artificial outer/middle ear are to be processed using the
                 specific recorded responses.

    wait_bar : keyword string (default: true)
               determines whether a progress bar displays during processing
               (set wait_bar to false for doing multi-file parallel calculations)

    out_plot : Boolean (default: False)
              flag indicating whether to generate a figure from the output
              (set out_plot to false for doing multi-file parallel calculations)

    binaural : Boolean (default: True)
               flag indicating whether to output combined binaural roughness
               for stereo input signal

    Returns
    -------
    roughness : dict
                contains the output

    roughness contains the following outputs:

    spec_roughness : 2d or 3d array
                    time-dependent specific roughness for each critical band
                    arranged as [time, bands(, channels)]

    spec_roughness_avg : 1d or 2d array
                      time-averaged specific roughness for each critical band
                      arranged as [bands(, channels)]

    roughness_t : 1d or 2d array
                  time-dependent overall roughness
                  arranged as [time(, channels)]

    roughness90pc : 1d or 2d array
                    time-aggregated (90th percentile) overall roughness
                    arranged as [roughness(, channels)]

    band_centre_freqs : 1d array
                        centre frequencies corresponding with each critical band
                        rate

    time_out : 1d array
               time (seconds) corresponding with time-dependent outputs

    soundfield : string
                 identifies the soundfield type applied (the input argument
                 soundfield)

    If outPlot=True, a set of plots is returned illustrating the energy
    time-averaged A-weighted sound level, the time-dependent specific and
    overall roughness, with the latter also indicating the time-aggregated
    value. A set of plots is returned for each input channel.

    If binaural=true, a corresponding set of outputs for the binaural
    loudness are also contained in roughnessSHM.

    Assumptions
    -----------
    The input signal is calibrated to units of acoustic pressure in Pascals
    (Pa).


    """
    # %% Input checks
    p, chans_in, chans = shm_in_check(p, samp_rate_in, axis,
                                      soundfield, wait_bar, out_plot,
                                      binaural)

    # assign chansOut
    if chans_in > 1:
        if binaural:
            chans_out = 3
            chans += ["binaural"]
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

    # Block and hop sizes Section 6.2.2 Table 4 ECMA-418-2:2025
    overlap = 0.75  # block overlap proportion
    # block size [s_b(z)]
    block_size = 16384
    # hop sizes (section 5.1.2 footnote 3 ECMA 418-2:2022) [s_h(z)]
    hop_size = int(((1 - overlap)*block_size))

    # Downsampled block and hop sizes Section 7.1.2 ECMA-418-2:2025
    downsample = 32  # downsampling factor
    samp_rate1500 = samp_rate48k/downsample
    block_size1500 = int(block_size/downsample)
    # hop_size1500 = (1 - overlap)*block_size1500
    # DFT resolution (section 7.1.5.1) [deltaf]
    res_dft1500 = samp_rate1500/block_size1500

    # Modulation rate error correction values Table 8, Section 7.1.5.1
    # ECMA-418-2:2025 [E(theta)]
    error_correction = np.array([0.0000, 0.0457, 0.0907, 0.1346, 0.1765, 0.2157,
                                 0.2515, 0.2828, 0.3084, 0.3269, 0.3364, 0.3348,
                                 0.3188, 0.2844, 0.2259, 0.1351, 0.0000])
    error_correction = np.hstack((error_correction, np.flip(-error_correction[0:-1]), 0))

    # High modulation rate roughness perceptual scaling function
    # (section 7.1.5.2 ECMA-418-2:2025)
    # Table 11 ECMA-418-2:2025 [r_1; r_2]
    rough_scale_params = np.vstack(([0.3560, 0.8024], [0.8049, 0.9333]))
    rough_scale_params = np.vstack((rough_scale_params[:, 0]*np.ones([np.sum(band_centre_freqs < 1e3), 2]),
                                    rough_scale_params[:, 1]*np.ones([np.sum(band_centre_freqs >= 1e3), 2]))).T

    # Equation 84 ECMA-418-2:2025 [r_max(z)]
    rough_scale = 1/(1 + rough_scale_params[0, :]
                     * np.abs(np.log2(band_centre_freqs/1000))
                     ** rough_scale_params[1, :])
    # Note: this is to ease parallelised calculations
    rough_scale = shm_dimensional(rough_scale, target_dim=3, where='first')

    # High/low modulation rate roughness perceptual weighting function
    # parameters (section 7.1.5.2 ECMA-418-2:2025)
    # Equation 86 ECMA-418-2:2025 [f_max(z)]
    mod_freq_max_weight = 72.6937*(1 - 1.1739
                                   * np.exp(-5.4583*band_centre_freqs/1000))

    # Equation 87 ECMA-418-2:2025 [q_1; q_2(z)]
    rough_hi_weight_params = np.vstack((1.2822*np.ones(band_centre_freqs.size),
                                        0.2471*np.ones(band_centre_freqs.size)))
    mask = band_centre_freqs/1000 >= 2**-3.4253
    rough_hi_weight_params[1, mask] = 0.2471 + 0.0129*(np.log2(band_centre_freqs[mask]/1000)
                                                       + 3.4253)**2
    # Note: this is to ease parallelised calculations
    rough_hi_weight_params = np.expand_dims(rough_hi_weight_params, axis=1)

    # (section 7.1.5.4 ECMA-418-2:2025)
    # Equation 96 ECMA-418-2:2025 [q_1; q_2(z)]
    rough_lo_weight_params = np.vstack((0.7066*np.ones(band_centre_freqs.size),
                                        1.0967 - 0.064*np.log2(band_centre_freqs/1000)))
    rough_lo_weight_params = np.expand_dims(rough_lo_weight_params, axis=1)

    # Output sample rate (section 7.1.7 ECMA-418-2:2025) [r_s50]
    samp_rate50 = 50

    # Calibration constant
    # calibration factor in Section 7.1.7 Equation 104 ECMA-418-2:2025 [c_R]
    cal_R = 0.0180909
    cal_Rx = 1/1.0011565  # calibration adjustment factor

    # Footnote 14 (/0 epsilon)
    epsilon = 1e-12

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

    # Section 5.1.5.2 and equation 103 [l_last]
    # NOTE: the corresponding definition of l_last in section 5.1.5.2 is
    # incorrect.
    l50_last = int(np.floor(n_samples/samp_rate48k*samp_rate50) + 1)

    # Section 5.1.2 ECMA-418-2:2025 Fade in weighting and zero-padding
    pn = shm_pre_proc(p_re, block_size=block_size, hop_size=hop_size, pad_start=True,
                    pad_end=False)

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

    spec_roughness = np.zeros([l50_last, n_bands, chans_out], order='F')
    for chan in chan_iter:
        # Apply auditory filter bank
        # --------------------------
        # Filter equalised signal using 53 1/2Bark ERB filters according to
        # Section 5.1.4.2 ECMA-418-2:2025
        pn_omz = shm_auditory_filtbank(pn_om[:, chan])

        # Note: At this stage, typical computer RAM limits impose a need to
        # loop through the critical bands rather than continue with a
        # parallelised approach, until later downsampling is applied

        if wait_bar:
            env_iter = tqdm(range(n_bands), desc="Envelope extraction")
        else:
            env_iter = range(n_bands)

        # pre-allocate output arrays
        i_start = 0  # index to start segmentation block processing from
        _, n_blocks, _ = shm_signal_segment_blocks(pn_omz[:, 0],
                                                  block_size=block_size,
                                                  overlap=overlap,
                                                  i_start=i_start,
                                                  end_shrink=True)
        basis_loudness = np.zeros([n_blocks, n_bands], order='F')
        envelopes = np.zeros([block_size1500, n_blocks, n_bands], order='F')
        for z_band in env_iter:
            # Segmentation into blocks
            # ------------------------

            # Section 5.1.5 ECMA-418-2:2025
            pn_lz, l_blocks_out = shm_signal_segment(pn_omz[:, z_band],
                                                   block_size=block_size,
                                                   overlap=overlap,
                                                   i_start=i_start,
                                                   end_shrink=True)

            # Transformation into Loudness
            # ----------------------------
            # Sections 5.1.6 to 5.1.9 ECMA-418-2:2025
            _, band_basis_loudness, _ = shm_basis_loudness(signal_segmented=pn_lz.copy(),
                                                           band_centre_freq=band_centre_freqs[z_band])
            basis_loudness[:, z_band] = band_basis_loudness

            # Envelope power spectral analysis
            # --------------------------------
            # Sections 7.1.2 ECMA-418-2:2025
            # magnitude of Hilbert transform with downsample - Equation 65
            # [p(ntilde)_E,l,z]
            envelopes[:, :, z_band] = shm_downsample(np.abs(hilbert(pn_lz,
                                                                    axis=0)),
                                                     downsample=32)

        # end of for loop for obtaining low frequency signal envelopes

        # Note: With downsampled envelope signals,
        # parallelised approach can continue

        # Section 7.1.3 equation 66 ECMA-418-2:2025 [Phi(k)_E,l,z]
        mod_spectra = np.zeros(envelopes.shape, order='F')
        envelope_win = envelopes*np.tile(shm_dimensional(windows.hann(block_size1500,
                                                                      sym=False),
                                                         target_dim=3),
                                         reps=[1, envelopes.shape[1],
                                               n_bands])/np.sqrt(0.375)
        envelope_win = np.asfortranarray(envelope_win)

        # Equation 66 & 67
        denom = shm_dimensional(np.max(basis_loudness, 1))*np.sum(envelope_win**2,
                                                                  0)
        mask = denom != 0  # Equation 66 criteria for masking
        # broadcast mask
        mask_rep = np.asfortranarray(np.broadcast_to(mask, mod_spectra.shape))
        scaling = np.divide(basis_loudness**2, denom, out=np.zeros_like(denom),
                            where=mask)  # Equation 66 factor
        # broadcast scaling
        scaling_rep = np.broadcast_to(scaling, mask_rep.shape)
        mod_spectra[mask_rep] = np.ravel((np.reshape(scaling_rep[mask_rep],
                                                     (block_size1500, -1, n_bands))
                                          * np.abs(fft(np.reshape(envelope_win[mask_rep],
                                                                  (block_size1500,
                                                                   -1, n_bands)),
                                                       axis=0))**2))

        # Envelope noise reduction
        # ------------------------
        # section 7.1.4 ECMA-418-2:2025
        mod_spectra_avg = sliding_window_view(mod_spectra, window_shape=3,
                                             axis=2).mean(axis=3)
        mod_spectra_avg = np.concatenate((shm_dimensional(mod_spectra[:, :, 0],
                                                          target_dim=3,
                                                          where='last'),
                                          mod_spectra_avg,
                                          shm_dimensional(mod_spectra[:, :, -1],
                                                          target_dim=3,
                                                          where='last')), axis=2)

        # Equation 68 [s(l,k)]
        mod_spectra_avg_sum = np.sum(mod_spectra_avg, axis=2)

        # Equation 71 ECMA-418-2:2025 [wtilde(l,k)]
        clip_weight = (0.0856*mod_spectra_avg_sum[0:int(mod_spectra_avg.shape[0]/2) + 1, :]
                       / (np.median(mod_spectra_avg_sum[2:int(mod_spectra_avg.shape[0]/2), :],
                                    axis=0) + 1e-10)
                       * shm_dimensional(np.minimum(np.maximum(0.1891*np.exp(0.012*np.arange(0,
                                                                                             int(mod_spectra_avg.shape[0]/2)
                                                                                             + 1)),
                                                               0),
                                                    1)))

        # Equation 70 ECMA-418-2:2025 [w(l,k)]
        weighting_factor1 = np.zeros(mod_spectra_avg_sum[0:257, :].shape,
                                     order='F')
        mask = clip_weight >= 0.05*np.max(clip_weight[2:256, :], axis=0)
        weighting_factor1[mask] = np.minimum(np.maximum(clip_weight[mask]
                                                        - 0.1407, 0), 1)
        weighting_factor = np.concatenate((weighting_factor1,
                                           np.flipud(weighting_factor1[1:256,
                                                                       :])),
                                          axis=0)

        # Calculate noise-reduced, scaled, weighted modulation power spectra
        # Equation 69 [Phihat(k)_E,l,z]
        mod_weight_spectra_avg = mod_spectra_avg*shm_dimensional(weighting_factor,
                                                                 target_dim=3)

        # Spectral weighting
        # ------------------
        # Section 7.1.5 ECMA-418-2:2025
        # theta used in equation 79, including additional index for
        # errorCorrection terms from table 10
        theta = np.arange(34)
        mod_amp = np.zeros([10, n_blocks, n_bands], order='F')
        mod_rate = np.zeros([10, n_blocks, n_bands], order='F')

        if wait_bar:
            rate_iter = tqdm(range(n_bands),
                            desc="Modulation rates")
        else:
            rate_iter = range(n_bands)

        for z_band in rate_iter:
            # Section 7.1.5.1 ECMA-418-2:2025
            for l_block in range(n_blocks):
                # identify peaks in each block (for each band)
                start_idx = 2
                end_idx = 255
                mod_weight_spectra_avg_band_block = mod_weight_spectra_avg[start_idx:end_idx,
                                                                           l_block,
                                                                           z_band]
                k_locs, pk_props = find_peaks(mod_weight_spectra_avg_band_block,
                                              prominence=0)
                phi_pks = mod_weight_spectra_avg_band_block[k_locs]
                proms = pk_props['prominences']

                # reindex k_locs to match spectral start index used in findpeaks
                # for indexing into modulation spectra matrices
                k_locs = k_locs + start_idx

                # we can only have peaks at k = 3:254
                mask = np.isin(k_locs, range(3, 255))
                k_locs = k_locs[mask]
                phi_pks = phi_pks[mask]

                # consider 10 highest prominence peaks only
                if len(proms) > 10:
                    proms_sorted = np.sort(proms)[::-1]
                    ii_sort = np.argsort(proms)[::-1]
                    mask = proms >= proms_sorted[9]

                    # if branch to deal with duplicated peak prominences
                    if sum(mask) > 10:
                        mask = mask[ii_sort <= 9]
                    # end of if branch for duplicated peak prominences

                    phi_pks = phi_pks[mask]
                    k_locs = k_locs[mask]

                # end of if branch to select 10 highest prominence peaks

                # consider peaks meeting criterion
                if phi_pks.size != 0:
                    mask = phi_pks > 0.05*np.max(phi_pks)  # Equation 72 criterion
                    phi_pks = phi_pks[mask]  # [Phihat(k_p,i(l,z))]
                    k_locs = k_locs[mask]
                    # loop over peaks to obtain modulation rates
                    for i_peak in range(len(phi_pks)):
                        # Equation 74 ECMA-418-2:2025
                        # [Phihat_E,l,z]
                        mod_amp_mat = np.vstack((mod_weight_spectra_avg[k_locs[i_peak] - 1,
                                                                        l_block, z_band],
                                                 mod_weight_spectra_avg[k_locs[i_peak],
                                                                        l_block, z_band],
                                                 mod_weight_spectra_avg[k_locs[i_peak] + 1,
                                                                        l_block, z_band]))

                        # Equation 82 [A_i(l,z)]
                        mod_amp[i_peak, l_block, z_band] = np.sum(mod_amp_mat)

                        # Equation 75 ECMA-418-2:2025
                        # [K]
                        mod_index_mat = np.vstack((np.hstack(((k_locs[i_peak] - 1)**2,
                                                              k_locs[i_peak] - 1, 1)),
                                                   np.hstack(((k_locs[i_peak])**2,
                                                              k_locs[i_peak], 1)),
                                                   np.hstack(((k_locs[i_peak] + 1)**2,
                                                              k_locs[i_peak] + 1, 1))))

                        # Equation 73 solution [C]
                        coeff_vec = np.linalg.solve(mod_index_mat, mod_amp_mat)

                        # Equation 76 ECMA-418-2:2025 [ftilde_p,i(l,z)]
                        mod_rate_est = (-(coeff_vec[1]/(2*coeff_vec[0]))*res_dft1500).item(0)

                        # Equation 79 ECMA-418-2:2025 [beta(theta)]
                        error_beta = ((np.floor(mod_rate_est/res_dft1500)
                                       + theta[:33]/32)*res_dft1500
                                      - (mod_rate_est
                                         + error_correction[theta[:33]]))

                        # Equation 80 ECMA-418-2:2025 [theta_min]
                        theta_min_error = np.argmin(np.abs(error_beta))

                        # Equation 81 ECMA-418-2:2025 [theta_corr]
                        if ((theta_min_error > 0)
                            and (error_beta[theta_min_error]*error_beta[theta_min_error - 1]
                                 < 0)):
                            theta_corr = theta_min_error
                        else:
                            theta_corr = theta_min_error + 1
                        # end of eq 81 if-branch

                        # Equation 78 ECMA-418-2:2025
                        # [rho(ftilde_p,i(l,z))]
                        bias_adjust = (error_correction[theta_corr - 1]
                                       - (error_correction[theta_corr]
                                          - error_correction[theta_corr - 1])
                                       * error_beta[theta_corr - 1]
                                       / (error_beta[theta_corr]
                                          - error_beta[theta_corr - 1]))

                        # Equation 77 ECMA-418-2:2025 [f_p,i(l,z)]
                        mod_rate[i_peak, l_block, z_band] = mod_rate_est + bias_adjust

                    # end of for loop over peaks in block per band
                # end of if branch for detected peaks in modulation spectrum
            # end of for loop over blocks for peak detection
        # end  of for loop over bands for modulation spectral weighting

        # Section 7.1.5.2 ECMA-418-2:2025 - Weighting for high modulation rates
        # Equation 85 [G_l,z,i(f_p,i(l,z))]
        rough_hi_weight = shm_mod_weight(mod_rate, mod_freq_max_weight,
                                         rough_hi_weight_params)

        # Equation 83 [Atilde_i(l,z)]
        mod_amp_hi_weight = mod_amp * rough_scale
        mask = mod_rate <= res_dft1500
        mod_amp_hi_weight[mask] = 0
        mask = mod_rate > mod_freq_max_weight
        mod_amp_hi_weight[mask] = mod_amp_hi_weight[mask]*rough_hi_weight[mask]

        # Section 7.1.5.3 ECMA-418-2:2025 - Estimation of fundamental modulation rate
        # TODO: replace the loop approach with a parallelised approach!
        # matrix initialisation to ensure zero rates do not cause missing bands in output
        mod_fund_rate = np.zeros([n_blocks, n_bands], order='F')
        mod_max_weight = np.zeros([10, n_blocks, n_bands], order='F')

        if wait_bar:
            rate_iter = tqdm(range(n_bands),
                            desc="Modulation weightings")
        else:
            rate_iter = range(n_bands)

        for z_band in rate_iter:
            for l_block in range(n_blocks):
                # Proceed with rate detection if non-zero modulation rates
                if np.max(mod_rate[:, l_block, z_band]) > 0:
                    mod_rate_for_loop = mod_rate[mod_rate[:, l_block, z_band] > 0,
                                                 l_block, z_band]

                    n_peaks = len(mod_rate_for_loop)

                    # initialise empty list for equation 90
                    ind_set_i_peak = np.empty([n_peaks,], dtype=object)
                    # initialise empty matrix for equation 91
                    harm_comp_energy = np.empty([n_peaks,], dtype=object)

                    for i_peak in range(n_peaks):
                        # Equation 88 [R_i_0(i)]
                        mod_rate_ratio = shm_round(mod_rate_for_loop/mod_rate_for_loop[i_peak])
                        uniq_ratios, start_group_inds, count_dupes = np.unique(mod_rate_ratio,
                                                                               return_index=True,
                                                                               return_counts=True)

                        # add any non-duplicated ratio indices
                        test_indices = -np.ones([10,]).astype(int)
                        if len(start_group_inds[count_dupes == 1]) > 0:
                            test_indices[0:len(start_group_inds[count_dupes == 1])] = start_group_inds[count_dupes == 1]
                        # end of non-duplicated ratio if branch

                        # loop over duplicated values to select single index
                        if np.max(count_dupes) > 1:
                            dupe_ratio_vals = uniq_ratios[count_dupes > 1]
                            for j_dupe in range(len(dupe_ratio_vals)):

                                # Equation 89 [i]
                                dupe_group_inds = (mod_rate_ratio == dupe_ratio_vals[j_dupe]).nonzero()[0]
                                denom = mod_rate_ratio[dupe_group_inds]*mod_rate_for_loop[i_peak]
                                test_dupe = np.abs(np.divide(mod_rate_for_loop[dupe_group_inds],
                                                             denom,
                                                             out=np.zeros_like(denom),
                                                             where=denom != 0) - 1)

                                # discard if no testDupes
                                if len(test_dupe) > 0:
                                    test_dupe_min = np.argmin(test_dupe)
                                    # append selected index
                                    test_indices[len(start_group_inds[count_dupes == 1])
                                                 + j_dupe] = dupe_group_inds[test_dupe_min]
                                # end of if branch for testDupes
                            # end of for loop over duplicated ratios
                        # end of if branch for duplicated ratios

                        # discard negative indices
                        test_indices = test_indices[test_indices >= 0]

                        # Equation 90 [I_i_0]
                        denom = mod_rate_ratio[test_indices]*mod_rate_for_loop[i_peak]
                        harm_complex_test = np.abs(np.divide(mod_rate_for_loop[test_indices],
                                                             denom,
                                                             out=np.zeros_like(denom),
                                                             where=denom != 0) - 1)
                        ind_set_i_peak[i_peak] = test_indices[harm_complex_test < 0.04]

                        # Equation 91 [E_i_0]
                        harm_comp_energy[i_peak] = np.sum(mod_amp_hi_weight[ind_set_i_peak[i_peak],
                                                                            l_block,
                                                                            z_band])

                    # end of loop over peaks

                    harm_comp_energy = harm_comp_energy.astype(float)
                    i_max_energy = np.argmax(harm_comp_energy)
                    ind_set_max = ind_set_i_peak[i_max_energy]
                    mod_fund_rate[l_block, z_band] = mod_rate_for_loop[i_max_energy]
                    # Equation 94 [i_peak]
                    i_peak_amp = np.argmax(mod_amp_hi_weight[ind_set_max, l_block, z_band])
                    i_peak = ind_set_max[i_peak_amp]

                    # Equation 93 [w_peak]
                    gravity_weight = 1 + 0.1*np.abs(np.sum(mod_rate_for_loop[ind_set_max]
                                                           * mod_amp_hi_weight[ind_set_max,
                                                                               l_block,
                                                                               z_band],
                                                           axis=0)
                                                    / np.sum(mod_amp_hi_weight[ind_set_max,
                                                                               l_block,
                                                                               z_band]
                                                             + epsilon, axis=0)
                                                    - mod_rate_for_loop[i_peak])**0.749

                    # Equation 92 [Ahat(i)]
                    mod_max_weight[ind_set_max,
                                   l_block,
                                   z_band] = gravity_weight*mod_amp_hi_weight[ind_set_max,
                                                                              l_block,
                                                                              z_band]

                # end of if branch for non-zero modulation rates
            # end of for loop over blocks
        # end of for loop over bands

        # Equation 95 [A(l,z)]
        rough_lo_weight = shm_mod_weight(mod_fund_rate, mod_freq_max_weight,
                                         rough_lo_weight_params)
        mod_max_weight_sum = np.sum(mod_max_weight, axis=0)
        mod_max_lo_weight = np.sum(rough_lo_weight*mod_max_weight, axis=0)
        mask = mod_fund_rate <= res_dft1500
        mod_max_lo_weight[mask] = 0
        mask = mod_fund_rate > mod_freq_max_weight
        mod_max_lo_weight[mask] = mod_max_weight_sum[mask]
        mod_amp_max = mod_max_lo_weight
        mod_amp_max[mod_amp_max < 0.074376] = 0

        # Time-dependent specific roughness
        # ---------------------------------
        # Section 7.1.7 ECMA-418-2:2025

        # Section 7.1.7 interpolation to 50 Hz sampling rate    
        t = l_blocks_out/samp_rate48k
        t50 = np.linspace(0, signal_t, l50_last)

        spec_rough_est = np.zeros([l50_last, n_bands], order='F')
        for z_band in range(n_bands):
            interpolator = PchipInterpolator(t, mod_amp_max[:, z_band], axis=0)
            spec_rough_est[:, z_band] = interpolator(t50)
        # end of for loop for interpolation
        spec_rough_est[spec_rough_est < 0] = 0  # [R'_est(l_50,z)]

        # Section 7.1.7 Equation 107 [Rtilde'_est(l_50)]
        spec_rough_est_rms = np.sqrt(np.mean(spec_rough_est**2, axis=1))

        # Section 7.1.7 Equation 108 [Rbar'_est(l_50)]
        spec_rough_est_avg = np.mean(spec_rough_est, axis=1)

        # Section 7.1.7 Equation 106 [B(l_50)]
        exp_b_l50 = np.zeros(spec_rough_est_avg.size)
        mask = spec_rough_est_avg != 0
        exp_b_l50[mask] = spec_rough_est_rms[mask]/spec_rough_est_avg[mask]

        # Section 7.1.7 Equation 105 [E(l_50)]
        exp_l50 = (0.95555 - 0.58449)*(np.tanh(1.6407*(exp_b_l50 - 2.5804)) + 1)*0.5 + 0.58449

        # Section 7.1.7 Equation 104 [Rhat'(l_50,z)]
        spec_rough_est_tform = cal_R*cal_Rx*(spec_rough_est.T**exp_l50).T

        # Section 7.1.7 Equation 109-110 [R'(l_50,z)]
        rise_time = 0.0625
        fall_time = 0.5
        spec_roughness[:, :, chan] = shm_mod_low_pass(spec_rough_est_tform, samp_rate50,
                                                      rise_time, fall_time)

    # end of for loop over channels

    # Binaural roughness
    # Section 7.1.11 ECMA-418-2:2025 [R'_B(l_50,z)]
    if chans_in == 2 and binaural:
        # Equation 112
        spec_roughness[:, :, 2] = np.sqrt(np.sum(spec_roughness[:, :, 0:2]**2,
                                                 axis=2)/2)
    # end of if branch for combined binaural

    # Section 7.1.8 ECMA-418-2:2025
    # Time-averaged specific roughness [R'(z)]
    spec_roughness_avg = np.mean(spec_roughness[16:, :, :], axis=0)

    # Section 7.1.9 ECMA-418-2:2025
    # Time-dependent roughness Equation 111 [R(l_50)]
    roughness_t = np.sum(spec_roughness*dz, axis=1)

    # ensure channel dimension is retained (to ease plotting)
    if chans_out == 1:
        spec_roughness_avg = shm_dimensional(spec_roughness_avg)
        roughness_t = shm_dimensional(roughness_t)

    # Section 7.1.10 ECMA-418-2:2025
    # Overall roughness [R]
    roughness90pc = np.percentile(roughness_t[16:, :], 90, axis=0)

    # time (s) corresponding with results output [t]
    time_out = np.arange(0, (spec_roughness.shape[0]))/samp_rate50

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
                                   np.swapaxes(spec_roughness[:, :, chan], 0, 1),
                                   cmap=cmap_inferno,
                                   vmin=0,
                                   vmax=np.ceil(np.max(spec_roughness[:, :,
                                                                      chan])*500)/500,
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
                         label=(r"Specific roughness,"
                                "\n"
                                r"$\mathregular{asper_{SHM}}/\mathregular{Bark_{SHM}}$"),
                         aspect=10, cax=cbax)

            ax2 = axs[1]
            ax2.plot(time_out, roughness90pc[chan]*np.ones(time_out.size),
                     color=cmap_inferno(33/255), linewidth=1, linestyle='dotted',
                     label=("90th-" + "\n" + "percentile"))
            ax2.plot(time_out, roughness_t[:, chan],
                     color=cmap_inferno(165/255),
                     linewidth=0.75, label=("Time-" + "\n" + "dependent"))
            ax2.set(xlim=[time_out[0], time_out[-1] + time_out[1] - time_out[0]],
                    xlabel="Time, s",
                    ylim=[0, 1.1*np.ceil(np.max(roughness_t[:, chan])*10)/10],
                    ylabel=(r"Roughness, $\mathregular{asper_{SHM}}$"))
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
        spec_roughness = np.squeeze(spec_roughness)
        spec_roughness_avg = np.squeeze(spec_roughness_avg)
        roughness_t = np.squeeze(roughness_t)
    # end of if branch for singleton dimensions

    # Assign outputs to structure
    roughness = {}
    if chans_out == 3:
        roughness.update({'spec_roughness': spec_roughness[:, :, 0:2]})
        roughness.update({'spec_roughness_avg': spec_roughness_avg[:, 0:2]})
        roughness.update({'roughness_t': roughness_t[:, 0:2]})
        roughness.update({'roughness90pc': roughness90pc[0:2]})
        roughness.update({'spec_roughness_bin': spec_roughness[:, :, 2]})
        roughness.update({'spec_roughness_avg_bin': spec_roughness_avg[:, 2]})
        roughness.update({'roughness_t_bin': roughness_t[:, 2]})
        roughness.update({'roughness90pc_bin': np.array(roughness90pc[2])})
        roughness.update({'band_centre_freqs': band_centre_freqs})
        roughness.update({'time_out': time_out})
        roughness.update({'soundfield': soundfield})
    else:
        roughness.update({'spec_roughness': spec_roughness})
        roughness.update({'spec_roughness_avg': spec_roughness_avg})
        roughness.update({'roughness_t': roughness_t})
        roughness.update({'roughness90pc': roughness90pc})
        roughness.update({'band_centre_freqs': band_centre_freqs})
        roughness.update({'time_out': time_out})
        roughness.update({'soundfield': soundfield})

    return roughness
# end of shm_roughness_ecma function
