# -*- coding: utf-8 -*-
# %% Preamble
"""
shm_tonalityecma.py
------------------

Returns tonality values and frequencies according to ECMA-418-2:2025
(using the Sottek Hearing Model) for an input calibrated single mono
or single stereo audio (sound pressure) time-series signal, p.

Requirements
------------
numpy
scipy
matplotlib
tqdm
bottleneck

Functions
---------

shm_tonality_ecma : This is the main tonality function, which implements sections
                    5 and 6 of ECMA-418-2:2025, and returns a dict containing
                    the tonality results as numpy arrays. This function is also
                    called by shm_loudness_ecma() to obtain Sottek Hearing Model
                    psychoacoustic loudness.

shm_band_auto_correlation : Subfunction called by shm_tonality_ecma(), which
                            returns the unbiased, normalised estimate of the
                            autocorrelation function for a critical band.

shm_band_loud_components : Subfunction called by shm_tonality_ecma(),
                           which returns the tonal loudness and noise
                           loudness components in each critical band, as
                           well as the band tonality frequencies.

shm_noise_red_lowpass : Subfunction called by shm_band_loud_components(), which
                        applies a low-pass filter to the tonal and noise
                        loudness components of the signal.

Ownership and Quality Assurance
-------------------------------
Author: Mike JB Lotinga (m.j.lotinga@edu.salford.ac.uk)
Institution: University of Salford

Date created: 25/05/2023
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
from scipy.signal import lfilter
from scipy.special import comb
import matplotlib as mpl
from matplotlib import pyplot as plt
from sottek_hearing_model.shm_subs import (shm_resample, shm_pre_proc,
                                           shm_outmid_ear_filter,
                                           shm_auditory_filtbank,
                                           shm_signal_segment,
                                           shm_basis_loudness,
                                           shm_rms, shm_round,
                                           shm_in_check)
from tqdm import tqdm
from sottek_hearing_model.filters import weight_A_t
from sottek_hearing_model.plotting_tools import create_figure, show_plot
import bottleneck as bn
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing

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
num_threads = max(1, multiprocessing.cpu_count() - 1)  # leave one free core

# %% shm_tonality_ecma
def shm_tonality_ecma(p, samp_rate_in, axis=0, soundfield='free_frontal',
                      wait_bar=True, out_plot=False):
    """
    Inputs
    ------
    p : 1D or 2D array
        the input signal as single mono or stereo audio (sound pressure)
        signals

    samp_rate_in : integer
                   the sample rate (frequency) of the input signal(s)

    axis : integer (0 or 1, default: 0)
           the time axis along which to calculate the tonality

    soundfield : keyword string (default: 'free_frontal')
                 determines whether the 'free_frontal' or 'diffuse' field stages
                 are applied in the outer-middle ear filter, or 'no_outer' uses
                 only the middle ear stage, or 'no_ear' omits ear filtering.
                 Note: these last two options are beyond the scope of the
                 standard, but may be useful if recordings made using
                 artificial outer/middle ear are to be processed using the
                 specific recorded responses.

    wait_bar : keyword string (default: True)
               determines whether a progress bar displays during processing
               (set wait_bar to false for doing multi-file parallel calculations)

    out_plot : Boolean (default: False)
               flag indicating whether to generate a figure from the output
               (set out_plot to false for doing multi-file parallel calculations)

    Returns
    -------
    tonality : dict
               contains the output

    tonality contains the following outputs:

    spec_tonality : 2D or 3D array
                    time-dependent specific tonality for each critical band
                    arranged as [time, bands(, channels)]

    spec_tonality_freqs : 2D or 3D array
                          time-dependent frequencies of the dominant tonal
                          components corresponding with each of the
                          time-dependent specific tonality values in each
                          (half) critical band
                          arranged as [time, bands(, channels)]

    spec_tonality_avg : 1D or 2D array
                        time-averaged specific tonality for each critical band
                        arranged as [bands(, channels)]

    spec_tonality_avg_freqs : 1D or 2D array
                              frequencies of the dominant tonal components
                              corresponding with each of the
                              time-averaged specific tonality values in each
                              (half) critical band
                              arranged as [bands(, channels)]

    spec_tonal_loudness : 2D or 3D array
                          time-dependent specific tonal loudness for each
                          critical band
                          arranged as [time, bands(, channels)]

    spec_noise_loudness : 2D or 3D array
                          time-dependent specific noise loudness for each
                          critical band
                          arranged as [time, bands(, channels)]

    tonality_t : 1D or 2D array
                   time-dependent overall tonality
                   arranged as [time(, channels)]

    tonality_t_freqs : 1D or 2D array
                       time-dependent frequencies of the dominant tonal
                       components corresponding with the
                       time-dependent overall tonality values
                       arranged as [time(, channels)]

    tonality_avg : 1D or 2D array
                   time-averaged overall tonality
                   arranged as [tonality(, channels)]

    band_centre_freqs : 1D array
                        centre frequencies corresponding with each critical band
                        rate

    time_out : 1D array
               time (seconds) corresponding with time-dependent outputs

    soundfield : string
                 identifies the soundfield type applied (the input argument
                 soundfield)

    If out_plot=True, a set of plots is returned illustrating the energy
    time-averaged A-weighted sound level, the time-dependent specific and
    overall tonality, with the latter also indicating the time-aggregated
    value. A set of plots is returned for each input channel.

    Assumptions
    -----------
    The input signal is calibrated to units of acoustic pressure in Pascals
    (Pa).

    """
    # %% Input checks
    p, chans_in, chans = shm_in_check(p, samp_rate_in, axis,
                                      soundfield, wait_bar, out_plot)

    # %% Define constants

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

    # Section 5.1.4.1 Equation 9 ECMA-418-2:2025 [F(z)]
    band_centre_freqs = (delta_freq0/c)*np.sinh(c*half_bark)
    # Section 5.1.4.1 Equation 10 ECMA-418-2:2025 [deltaf(z)]
    dfz = np.sqrt(delta_freq0**2 + (c*band_centre_freqs)**2)

    # Block and hop sizes Section 6.2.2 Table 4 ECMA-418-2:2025
    overlap = 0.75  # block overlap proportion
    # block sizes [s_b(z)]
    block_size = np.hstack((8192*np.ones([3,]), 4096*np.ones([13,]),
                            2048*np.ones([9,]),
                            1024*np.ones([28,]))).astype(int)
    max_block_size = np.max(block_size)

    # hop sizes (section 5.1.2 footnote 3 ECMA 418-2:2022) [s_h(z)]
    hop_size = ((1 - overlap)*block_size).astype(int)

    # Output sample rate based on hop sizes - Resampling to common time basis
    # Section 6.2.6 ECMA-418-2:2025 [r_sd]
    samp_rate1875 = samp_rate48k/np.min(hop_size)

    # Number of bands that need averaging. Section 6.2.3 Table 5
    # ECMA-418-2:2025 [NB]
    # num_bands_avg = np.vstack((np.hstack((np.zeros([1, 1]), np.ones([1, 1]),
#                                          2*np.ones([1, 14]), np.ones([1, 9]),
    #                                      np.zeros([1, 28]))),
    #                            np.hstack((np.ones([1, 1]), np.ones([1, 1]),
    #                                       2*np.ones([1, 14]), np.ones([1, 9]),
    #                                      np.zeros([1, 28]))))).astype(int)

    # Noise reduction constants from Section 6.2.7 Table 7 ECMA-418-2:2025
    alpha = 20
    beta = 0.07

    # Scaling factor constants from Section 6.2.8 Table 9 ECMA-418-2:2025
    A = 35
    B = 0.003

    # calibration factor in Section 6.2.8 Equation 51 ECMA-418-2:2025 [c_T]
    cal_T = 2.8758615
    # Adjustment to calibration factor (Footnote 22 ECMA-418-2:2025)
    cal_Tx = 1/0.9999043734252

    # Footnote 14 (/0 epsilon)
    epsilon = 1e-12

    # Duplicate Banded Data for ACF
    # ACF averaging occurs over neighbouring bands, to do this the segmentation
    # needs to be duplicated for neigbouring bands. 'Dupe' has been added
    # to variables to indicate that the vectors/matrices have been modified
    # for duplicated neigbouring bands.
    block_size_dupe = (np.hstack((8192*np.ones([5,]), 4096*np.ones([17,]),
                                  2048*np.ones([11,]),
                                  1024*np.ones([28,])))).astype(int)

    band_centre_freqs_dupe = np.hstack((band_centre_freqs[0:5],
                                        band_centre_freqs[1:18],
                                        band_centre_freqs[15:26],
                                        band_centre_freqs[25:53]))

    # (duplicated) indices corresponding with the NB bands around each z band
    i_num_bands_avg_dupe = np.vstack((np.hstack(([0, 0, 0], np.arange(5, 18),
                                                 np.arange(22, 31),
                                                 np.arange(33, 61))),
                                      np.hstack(([2, 3, 5], np.arange(10, 23),
                                                 np.arange(25, 34),
                                                 np.arange(34, 62)))))

    # Determine number of threads to use
    n_threads = min(n_bands, num_threads)  # number of threads to use in parallel
    
    # %% Signal processing

    # Input pre-processing
    # --------------------
    if samp_rate_in != samp_rate48k:  # Resample signal
        p_re, _ = shm_resample(p, samp_rate_in)
    else:  # don't resample
        p_re = p
    # end of if branch for resampling

    # Section 5.1.2 ECMA-418-2:2025 Fade in weighting and zero-padding
    pn = shm_pre_proc(p_re, max_block_size, np.max(hop_size))

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

    # Equation 40 ECMA-418-2:2025 [l_end]
    end_block = int(np.ceil(p_re.shape[0]/samp_rate48k*samp_rate1875))
    # pre-allocate results arrays
    # spec_snr = np.zeros([end_block + 1, n_bands, chans_in])  # dev only
    # spec_loudness = np.zeros([end_block + 1, n_bands, chans_in])  # dev only
    spec_tonal_loudness = np.zeros([end_block + 1, n_bands, chans_in], order='F')
    spec_noise_loudness = np.zeros([end_block + 1, n_bands, chans_in], order='F')
    spec_tonality_freqs = np.zeros([end_block + 1, n_bands, chans_in], order='F')
    spec_tonality_avg = np.zeros([n_bands, chans_in], order='F')
    spec_tonality_avg_freqs = np.zeros([n_bands, chans_in], order='F')
    tonality_t = np.zeros([end_block + 1, chans_in], order='F')
    tonality_t_freqs = np.zeros([end_block + 1, chans_in], order='F')
    tonality_avg = np.zeros([chans_in])

    for chan in chan_iter:
        # Apply auditory filter bank
        # --------------------------
        # Filter equalised signal using 53 1/2Bark ERB filters according to
        # Section 5.1.4.2 ECMA-418-2:2025
        pn_omz = shm_auditory_filtbank(pn_om[:, chan])

        # Autocorrelation function analysis
        # ---------------------------------
        # Duplicate Banded Data for ACF
        # Averaging occurs over neighbouring bands, to do this the segmentation
        # needs to be duplicated for neigbouring bands. 'Dupe' has been added
        # to variables to indicate that the vectors/matrices have been modified
        # for duplicated neigbouring bands.

        pn_omz_dupe = np.hstack((pn_omz[:, 0:5], pn_omz[:, 1:18],
                                 pn_omz[:, 15:26], pn_omz[:, 25:53]))

        if wait_bar:
            band_acf_iter = tqdm(range(61), desc="Critical band autocorrelation")
        else:
            band_acf_iter = range(61)

        # pre-allocate arrays
        # basisLoudnessArray = np.empty(61, dtype=object)
        unbiased_norm_acf_dupe = np.empty(61, dtype=object)

        with ThreadPoolExecutor(max_workers=n_threads) as executor:
            futures = [executor.submit(shm_band_auto_correlation, z_band,
                                       band_centre_freqs_dupe,
                                       block_size_dupe, overlap, pn_omz_dupe)
                       for z_band in band_acf_iter]

        for future in as_completed(futures):
            z_band, unbiased_norm_acf_dupe_band = future.result()
            unbiased_norm_acf_dupe[z_band] = unbiased_norm_acf_dupe_band

        # Average the ACF over nB bands - Section 6.2.3 ECMA-418-2:2025
        if wait_bar:
            band_acf_avg_iter = tqdm(range(n_bands),
                                     desc="Component loudness estimation")
        else:
            band_acf_avg_iter = range(n_bands)

        with ThreadPoolExecutor(max_workers=n_threads) as executor:
            futures = [executor.submit(shm_band_loud_components, z_band,
                                       band_centre_freqs,
                                       block_size, end_block, dfz,
                                       unbiased_norm_acf_dupe, i_num_bands_avg_dupe)
                       for z_band in band_acf_avg_iter]

        for future in as_completed(futures):
            (z_band, band_tonal_loudness,
             band_noise_loudness, band_tonal_freqs) = future.result()
            spec_tonal_loudness[:, z_band, chan] = band_tonal_loudness
            spec_noise_loudness[:, z_band, chan] = band_noise_loudness
            spec_tonality_freqs[:, z_band, chan] = band_tonal_freqs

        # end of ACF averaging over bands
        
        # Ensure any tiny negative values are set to zero
        spec_tonal_loudness[spec_tonal_loudness < 0] = 0
        spec_noise_loudness[spec_noise_loudness < 0] = 0

        # Calculation of specific tonality
        # --------------------------------
        # Section 6.2.8 Equation 49 ECMA-418-2:2025 [SNR(l)]
        # loudness signal-noise-ratio
        overall_snr = (np.max(spec_tonal_loudness, axis=1)
                       / (np.sum(spec_noise_loudness, axis=1) + epsilon))

        # Section 6.2.8 Equation 50 ECMA-418-2:2025 [q(l)]
        crit = np.exp(-A*(overall_snr - B))
        ql = 1 - crit  # sigmoidal scaling factor
        ql[crit >= 1] = 0

        # Section 6.2.8 Equation 51 ECMA-418-2:2025 [T'(l,z)]
        # time-dependent specific tonality
        spec_tonality = cal_T*cal_Tx*np.swapaxes(np.tile(ql,
                                                        [53, 1, 1]),
                                                 0, 1)*spec_tonal_loudness

        # Calculation of time-averaged specific tonality Section 6.2.9
        # ECMA-418-2:2025 [T'(z)]
        for z_band in range(n_bands):
            # criterion Section 6.2.9 point 2
            mask = spec_tonality[:, z_band, chan] > 0.02
            mask[0:57] = False  # criterion Section 6.2.9 point 1

            # Section 6.2.9 Equation 53 ECMA-418-2:2025
            spec_tonality_avg[z_band,
                              chan] = np.sum(spec_tonality[mask,
                                                           z_band,
                                                           chan],
                                             0)/(np.count_nonzero(mask)
                                                 + epsilon)
            spec_tonality_avg_freqs[z_band,
                                    chan] = np.sum(spec_tonality_freqs[mask,
                                                                       z_band,
                                                                        chan],
                                                   0)/(np.count_nonzero(mask)
                                                       + epsilon)
        # end of specific tonality for loop over bands

        # Calculation of total (non-specific) tonality Section 6.2.10
        # -----------------------------------------------------------
        # Further update can add the user input frequency range to determine
        # total tonality - not yet incorporated

        # Section 6.2.8 Equation 52 ECMA-418-2:2025
        # time (s) corresponding with results output [t]
        time_out = np.arange(0, spec_tonality.shape[0])/samp_rate1875

        # Section 6.2.10 Equation 61 ECMA-418-2:2025
        # Time-dependent total tonality [T(l)]
        tonality_t[:, chan] = np.max(spec_tonality[:, :, chan], axis=1)
        zmax = np.argmax(spec_tonality[:, :, chan], axis=1)

        for ll in range(end_block + 1):
            tonality_t_freqs[ll, chan] = spec_tonality_freqs[ll, zmax[ll], chan]
        # end of total tonality for loop over blocks

        # Calculation of representative values Section 6.2.11 ECMA-418-2:2025
        # Time-averaged total tonality
        mask = tonality_t[:, chan] > 0.02  # criterion Section 6.2.9 point 2
        mask[0:57] = False    # criterion Section 6.2.9 point 1

        # Section 6.2.11 Equation 63 ECMA-418-2:2025
        # Time-averaged total tonality [T]
        tonality_avg[chan] = np.sum(tonality_t[mask, chan],
                                    axis=0)/(np.count_nonzero(mask) + epsilon)

        # %% Output plotting

        # Plot figures
        # ------------
        if out_plot:
            # Plot results
            cmap_plasma = mpl.colormaps['plasma']
            chan_lab = chans[chan]
            fig, axs = create_figure(nrows=2, ncols=1, figsize=[10.5, 7.5],
                                     layout='constrained')

            ax1 = axs[0]
            pmesh = ax1.pcolormesh(time_out, band_centre_freqs,
                                   np.swapaxes(spec_tonality[:, :, chan], 0, 1),
                                   cmap=cmap_plasma,
                                   vmin=0,
                                   vmax=np.ceil(np.max(tonality_t[:, chan])*10)/10,
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
                         label=(r"Specific Tonality,"
                                "\n"
                                r"$\mathregular{tu_{SHM}}/\mathregular{Bark_{SHM}}$"),
                         aspect=10, cax=cbax)

            ax2 = axs[1]
            ax2.plot(time_out, tonality_avg[chan]*np.ones(time_out.size),
                     color=cmap_plasma(33/255), linewidth=1, linestyle='dotted',
                     label=("Time-" + "\n" + "average"))
            ax2.plot(time_out, tonality_t[:, chan],
                     color=cmap_plasma(165/255),
                     linewidth=0.75, label=("Time-" + "\n" + "dependent"))
            ax2.set(xlim=[time_out[0], time_out[-1] + time_out[1] - time_out[0]],
                    xlabel="Time, s",
                    ylim=[0, 1.1*np.ceil(np.max(tonality_t[:, chan])*10)/10],
                    ylabel=(r"Tonality, $\mathregular{tu_{SHM}}$"))
            ax2.grid(alpha=0.075, linestyle='--')
            ax2.legend(bbox_to_anchor=(1.025, 0.8), loc='upper left', title="Overall")

            # Filter signal to determine A-weighted time-averaged level
            pA = weight_A_t(p_re[:, chan], fs=samp_rate48k)
            level_Aeq = 20*np.log10(shm_rms(pA)/2e-5)
            fig.suptitle(t=(chan_lab + " signal sound pressure level = " +
                            str(shm_round(level_Aeq, 1)) +
                            r" dB $\mathregular{\mathit{L}_{Aeq}}$"))
            
            show_plot(fig)

        # end of if branch for plotting
    # end of for loop over channels

    # %% Output assignment

    # Discard singleton dimensions
    if chans_in == 1:
        spec_tonality = np.squeeze(spec_tonality)
        spec_tonality_avg = np.squeeze(spec_tonality_avg)
        spec_tonality_freqs = np.squeeze(spec_tonality_freqs)
        spec_tonality_avg_freqs = np.squeeze(spec_tonality_avg_freqs)
        tonality_t = np.squeeze(tonality_t)
        tonality_t_freqs = np.squeeze(tonality_t_freqs)
    # end of if branch for singleton dimensions

    # Assign outputs to structure
    tonality = {}
    tonality.update({'spec_tonality': spec_tonality})
    tonality.update({'spec_tonality_avg': spec_tonality_avg})
    tonality.update({'spec_tonality_freqs': spec_tonality_freqs})
    tonality.update({'spec_tonality_avg_freqs': spec_tonality_avg_freqs})
    tonality.update({'spec_tonal_loudness': spec_tonal_loudness})
    tonality.update({'spec_noise_loudness': spec_noise_loudness})
    tonality.update({'tonality_t': tonality_t})
    tonality.update({'tonality_avg': tonality_avg})
    tonality.update({'tonality_t_freqs': tonality_t_freqs})
    tonality.update({'band_centre_freqs': band_centre_freqs})
    tonality.update({'time_out': time_out})
    tonality.update({'soundfield': soundfield})

    return tonality
# end of shm_tonality_ecma function


def shm_band_auto_correlation(z_band, band_centre_freqs, block_size_bands,
                              overlap, signal_bands):
    """
    shm_band_auto_correlation(z_band, band_centre_freqs, block_size_bands,
                                 overlap, signal_bands)

    Returns the critical band autocorrelation estimate.

    Inputs
    ------
    z_band : integer
             the input critical band index

    band_centre_freqs : array of floats
                        the centre frequencies of the critical bands

    block_size_bands : array of integers
                       the block size in samples for each critical band

    overlap : float
              the proportion of overlap

    signal_bands : 2D array
                   the input critical bandpass-filtered signals

    Returns
    -------
    z_band : integer
             the input critical band index as output

    unbiased_norm_band_acf : 2D array
                             unbiased and normalised estimate of the autocorrelation
                             function in the corresponding critical band

    """

    # %% Define constants
    epsilon = 1e-12  # Footnote 14 (/0 epsilon)
    
    # %% Signal processing

    block_size = block_size_bands[z_band]

    # Segmentation into blocks
    # ------------------------
    # Section 5.1.5 ECMA-418-2:2025
    i_start = block_size_bands[0] - block_size
    signal_seg, _ = shm_signal_segment(signal_bands[:, z_band],
                                       block_size=block_size,
                                       overlap=overlap,
                                       i_start=i_start)

    # Transformation into Loudness
    # ----------------------------
    # Sections 5.1.6 to 5.1.9 ECMA-418-2:2025 [N'_basis(z)]
    signal_rect_seg, band_basis_loudness, _ = shm_basis_loudness(signal_segmented=signal_seg,
                                                                 band_centre_freq=band_centre_freqs[z_band])

    # ACF implementation using DFT
    # Section 6.2.2 Equations 27 & 28 ECMA-418-2:2025
    # [phi_unscaled,l,z(m)]
    unscaled_acf = np.asfortranarray(np.fft.irfft(np.abs(np.fft.rfft(signal_rect_seg,
                                                                     n=2*block_size,
                                                                     axis=0))**2,
                                                  n=2*block_size,
                                                  axis=0))

    # Section 6.2.2 Equation 29 ECMA-418-2:2025 [phi_l,z(m)]
    denom = (np.sqrt(np.flip(np.cumsum(np.flip(signal_rect_seg, axis=0)**2,
                                       axis=0), axis=0)
                     * np.flip(np.cumsum(signal_rect_seg**2, axis=0),
                               axis=0)) + epsilon)

    # note that the block length is used here, rather than the 2*s_b,
    # for compatability with the remaining code - beyond 0.75*s_b is
    # assigned (unused) zeros in the next line
    unbiased_norm_acf = unscaled_acf[0:block_size, :]/denom
    unbiased_norm_acf[int(0.75*block_size):block_size, :] = 0

    unbiased_norm_acf_band = unbiased_norm_acf*band_basis_loudness

    return z_band, unbiased_norm_acf_band
# end of shm_band_auto_correlation function


def shm_band_loud_components(z_band, band_centre_freqs, block_size_bands,
                             last_block, dfz, unbiased_norm_acf, i_num_bands_avg):
    """
    shm_band_loud_components(z_band, band_centre_freqs, block_size_bands,
                             last_block, dfz, unbiased_norm_acf, i_num_bands_avg)

    Returns the tonal loudness, noise loudness and tonality frequencies for the critical band.

    Inputs
    ------
    z_band : integer
             the input critical band index
    band_centre_freqs : array of floats
                        the centre frequencies of the critical bands
    block_size_bands : array of integers
                       the block size in samples for each critical band
    last_block : integer
                 the index of the last block
    unbiased_norm_acf : 2D array
                        the unbiased and normalised estimate of the autocorrelation
                        function in the corresponding critical band
    i_num_bands_avg : array of integers
                      the indices for averaging adjacent critical bands

    Returns
    -------
    z_band : integer
             the input critical band index as output
    band_tonal_loudness : 2D array
                          the specific loudness of the tonal component in the critical band

    band_noise_loudness : 2D array
                          the specific loudness of the noise component in the critical band
    
    band_tonal_freqs : 2D array
                       the time-dependent frequencies of the tonal components in the critical band
    """

    # %% Define constants
    epsilon = 1e-12

    # Noise reduction constants from Section 6.2.7 Table 7 ECMA-418-2:2025
    alpha = 20
    beta = 0.07

    # sample rates
    samp_rate1875 = 187.5
    samp_rate48k = 48e3

    # largest block size
    max_block_size = np.max(block_size_bands)

    # Critical band interpolation factors from Section 6.2.6 Table 6
    # ECMA-418-2:2025 [i]
    i_interp = block_size_bands/np.min(block_size_bands)

    # Sigmoid function factor parameters Section 6.2.7 Table 8 ECMA-418-2:2025
    # [c(s_b(z))]
    csz_b = np.hstack((18.21*np.ones([3,]), 12.14*np.ones([13,]),
                       417.54*np.ones([9,]), 962.68*np.ones([28,])))
    # [d(s_b(z))]
    dsz_b = np.hstack((0.36*np.ones([3,]), 0.36*np.ones([13,]),
                       0.71*np.ones([9,]), 0.69*np.ones([28,])))


    # %% Signal processing
    # Averaging of frequency bands
    mean_scaled_acf = np.mean(unbiased_norm_acf[i_num_bands_avg[0, z_band]
                                                :i_num_bands_avg[1, z_band]],
                              axis=0)

    # Average the ACF over adjacent time blocks [phibar_z'(m)]
    if z_band < 16:
        mean_scaled_acf = np.asfortranarray(np.roll(np.nan_to_num(bn.move_mean(mean_scaled_acf,
                                                                               window=3,
                                                                               min_count=3,
                                                                               axis=1),
                                                                  copy=False),
                                                    shift=-1, axis=1))
    # end of if branch for moving mean over time blocks

    # Application of ACF lag window Section 6.2.4 ECMA-418-2:2025
    tauz_start = max(0.5/dfz[z_band], 2e-3)  # Equation 31 ECMA-418-2:2025 [tau_start(z)]
    tauz_end = max(4/dfz[z_band], tauz_start + 1e-3)  # Equation 32 ECMA-418-2:2025 [tau_end(z)]
    # Equations 33 & 34 ECMA-418-2:2025
    mz_start = int(np.ceil(tauz_start*samp_rate48k) - 1)  # Starting lag window index [m_start(z)]
    mz_end = int(np.floor(tauz_end*samp_rate48k) - 1)  # Ending lag window index [m_end(z)]
    num_win_samp = mz_end - mz_start + 1  # number of samples in lag window [M]
    # Equation 35 ECMA-418-2:2025
    # lag-windowed, detrended ACF [phi'_z,tau(m)]
    lag_win_acf = np.zeros(mean_scaled_acf.shape, order='F')
    lag_win_acf[mz_start
                :mz_end + 1, :] = (mean_scaled_acf[mz_start:
                                                   mz_end + 1, :]
                                   - np.mean(mean_scaled_acf[mz_start:
                                                             mz_end + 1, :],
                                             axis=0))

    # Estimation of tonal loudness
    # ----------------------------
    # Section 6.2.5 Equation 36 ECMA-418-2:2025
    # ACF spectrum in the lag window [Phi'_z,tau(k)]
    mag_fft_lag_win_acf = np.abs(np.fft.fft(lag_win_acf,
                                            n=2*max_block_size,
                                            axis=0))

    mag_fft_lag_win_acf[np.isnan(mag_fft_lag_win_acf, order='F')] = 0.0

    # added to avoid spurious tiny results affecting tonal frequency
    # identification
    mag_fft_lag_win_acf[mag_fft_lag_win_acf <= epsilon] = 0.0

    # Section 6.2.5 Equation 37 ECMA-418-2:2025 [Nhat'_tonal(z)]
    # first estimation of specific loudness of tonal component in critical band
    band_tonal_loudness = mean_scaled_acf[0, :].copy()
    mask = (2*np.max(mag_fft_lag_win_acf, axis=0)/(num_win_samp/2)
            <= mean_scaled_acf[0, :])
    band_tonal_loudness[mask] = 2*np.max(mag_fft_lag_win_acf[:, mask],
                                         axis=0)/(num_win_samp/2)

    # Section 6.2.5 Equation 38 & 39 ECMA-418-2:2025
    # [k_max(z)]
    # NOTE: round is used to resolve discrepancies due to tiny floating point values at spectral line pairs
    kz_max = np.argmax(np.round(mag_fft_lag_win_acf, 8), axis=0)
    # frequency of maximum tonal component in critical band [f_ton(z)]
    band_tonal_freqs = kz_max*(samp_rate48k/(2*max_block_size))

    # Section 6.2.7 Equation 41 ECMA-418-2:2025 [N'_signal(l,z)]
    # specific loudness of complete band-pass signal in critical band
    band_loudness = mean_scaled_acf[0, :].copy()

    # Resampling to common time basis Section 6.2.6 ECMA-418-2:2025
    if i_interp[z_band] > 1:
        # Note: use of interpolation function avoids rippling caused by
        # resample function, which otherwise affects specific loudness
        # calculation for tonal and noise components
        l_n = mean_scaled_acf.shape[1]
        xp = np.linspace(0, l_n - 1, l_n)
        x = np.linspace(0, l_n - 1, int(i_interp[z_band]*(l_n - 1) + 1))

        band_tonal_loudness = np.interp(x, xp, band_tonal_loudness)
        band_loudness = np.interp(x, xp, band_loudness)
        band_tonal_freqs = np.interp(x, xp, band_tonal_freqs)

    # end of if branch for interpolation

    # Remove end zero-padded samples Section 6.2.6 ECMA-418-2:2025
    band_tonal_loudness = band_tonal_loudness[0:last_block + 1]
    band_loudness = band_loudness[0:last_block + 1]
    band_tonal_freqs = band_tonal_freqs[0:last_block + 1]

    # Noise reduction Section 6.2.7 ECMA-418-2:2020
    # ---------------------------------------------
    # Equation 42 ECMA-418-2:2025 signal-noise-ratio first approximation
    # (ratio of tonal component loudness to non-tonal component
    # loudness in critical band)
    # [SNRhat(l,z)]
    snr_lz1 = band_tonal_loudness/((band_loudness - band_tonal_loudness)
                                   + epsilon)

    # Equation 43 ECMA-418-2:2025 low pass filtered specific loudness
    # of non-tonal component in critical band [Ntilde'_tonal(l,z)]
    band_tonal_loudness = shm_noise_red_lowpass(band_tonal_loudness,
                                                samp_rate1875)

    # Equation 44 ECMA-418-2:2025 lowpass filtered SNR (improved
    # estimation)
    # [SNRtilde(l,z)]
    snr_lz = shm_noise_red_lowpass(snr_lz1, samp_rate1875)

    # Equation 46 ECMA-418-2:2025 [g(z)]
    gz = csz_b[z_band]/(band_centre_freqs[z_band]**dsz_b[z_band])

    # Equation 45 ECMA-418-2:2025 [nr(l,z)]
    crit = np.exp(-alpha*((snr_lz/gz) - beta))
    nrlz = 1 - crit  # sigmoidal weighting function
    nrlz[crit >= 1] = 0

    # Equation 47 ECMA-418-2:2025 [N'_tonal(l,z)]
    band_tonal_loudness = nrlz*band_tonal_loudness

    # Section 6.2.8 Equation 48 ECMA-418-2:2025 [N'_noise(l,z)]
    # specific loudness of non-tonal component in critical band
    band_noise_loudness = (shm_noise_red_lowpass(band_loudness,
                                            samp_rate1875)
                           - band_tonal_loudness)

    return (z_band,
            band_tonal_loudness,
            band_noise_loudness,
            band_tonal_freqs)
# end of shm_band_loud_components function


# %% shm_noise_red_lowpass
def shm_noise_red_lowpass(signal, samp_rate_in):
    """
    signal_filtered = shm_noise_red_lowpass(signal, samp_rate_in)

    Returns signal low pass filtered for noise reduction according to
    ECMA-418-2:2025 (the Sottek Hearing Model) for an input signal.

    Inputs
    ------
    signal : 1D or 2D matrix
             the input signal as single mono or stereo audio (sound
             pressure) signals

    samp_rate_in : double
                   the sample rate (frequency) of the input signal(s)

    Returns
    -------
    signal_filtered : 1D or 2D matrix
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

    d = np.exp(-1/(samp_rate_in*tau))  # Section 5.1.4.2 ECMA-418-2:2025

    # Feed-backward coefficients, Equation 14 ECMA-418-2:2025
    m_a = range(1, k + 1)
    a = np.append(1, ((-d)**m_a)*comb(k, m_a))

    # Feed-forward coefficients, Equation 15 ECMA-418-2:2025
    m_b = range(k)
    i = range(1, k)
    b = (((1 - d)**k)/sum(e_i[1:]*(d**i)))*(d**m_b)*e_i

    # Recursive filter Equation 13 ECMA-418-2:2025
    signal_filtered = lfilter(b, a, signal, axis=0)

    return signal_filtered
