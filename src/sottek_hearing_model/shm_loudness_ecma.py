# -*- coding: utf-8 -*-
# %% Preamble
"""
shm_loudness_ecma.py
--------------------

Returns loudness values according to ECMA-418-2:2025 (using the Sottek Hearing
Model) for an input calibrated single mono or single stereo audio (sound
pressure) time-series signal, p.

A convenience function, shmLoudnessECMAFromComp, is also provided, which
enables loudness to be calculated directly from the loudness components obtained
using acousticSHMTonality. This reduces the calculation time for loudness to
negligible when also calculating tonality.

Requirements
------------
numpy
scipy
matplotlib

Functions
---------

shm_loudness_ecma : This is the main loudness function, which implements sections
                    8 of ECMA-418-2:2025, and returns a dict containing
                    the loudness results as numpy arrays. This function relies on
                    calling shm_tonality_ecma to obtain the tonal and noise loudness
                    components.

shm_loudness_ecma_from_comp : Convenience function to return loudness values
                              from the loudness components obtained using 
                              shm_tonality_ecma(), reducing calculation time
                              to negligible cost when also calculating tonality
                              for the same signal(s).

Ownership and Quality Assurance
-------------------------------
Author: Mike JB Lotinga (m.j.lotinga@edu.salford.ac.uk)
Institution: University of Salford

Date created: 29/05/2023
Date last modified: 20/10/2025
Python version: 3.11

Copyright statement: This code has been developed during work undertaken within
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
import matplotlib as mpl
from matplotlib import pyplot as plt
from sottek_hearing_model.plotting_tools import create_figure
from sottek_hearing_model.shm_subs import (shm_resample,
                                           shm_dimensional, shm_rms,
                                           shm_round, shm_in_check)
from sottek_hearing_model.shm_tonality_ecma import shm_tonality_ecma
from sottek_hearing_model.filters import weight_A_t
from sottek_hearing_model.plotting_tools import create_figure, show_plot

# %% Module settings
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = 'Arial'
mpl.rcParams['mathtext.fontset'] = 'stixsans'

plt.rc('font', size=16)  # controls default text sizes
plt.rc('axes', titlesize=18,
       labelsize=18)  # fontsize of the axes title and x and y labels
plt.rc('xtick', labelsize=16)  # fontsize of the tick labels
plt.rc('ytick', labelsize=16)  # fontsize of the tick labels
plt.rc('legend', fontsize=16)  # legend fontsize
plt.rc('figure', titlesize=20)  # fontsize of the figure title


# %% shm_loudness_ecma
def shm_loudness_ecma(p, samp_rate_in, axis=0, soundfield='free_frontal',
                      wait_bar=True, out_plot=False, binaural=True):
    """
    Inputs
    ------
    p : 1D or 2D array
        the input signal as single mono or stereo audio (sound
        pressure) signals

    samp_rate_in : integer
                   the sample rate (frequency) of the input signal(s)

    axis : integer (0 or 1, default: 0)
           the time axis along which to calculate the loudness

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

    binaural : Boolean (default: True)
               flag indicating whether to output combined binaural loudness for
               stereo input signal

    Returns
    -------
    loudness : dict
               contains the output

    loudness contains the following outputs:

    spec_loudness : 2D or 3D array
                    time-dependent specific loudness for each critical band
                    arranged as [time, bands(, channels)]

    spec_loudness_powavg : 1D or 2D array
                           time-averaged specific loudness for each critical band
                           arranged as [bands(, channels)]

    spec_tonal_loudness : 2D or 3D array
                          time-dependent specific tonal loudness for each
                          critical band arranged as [time, bands(, channels)]

    spec_noise_loudness : 2D or 3D array
                          time-dependent specific noise loudness for each
                          critical band arranged as [time, bands(, channels)]

    loudness_t : 1D or 2D array
                 time-dependent overall loudness
                 arranged as [time(, channels)]

    loudness_powavg : 1D or 2D array
                      time-averaged overall loudness
                      arranged as [loudness(, channels)]

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
    overall loudness, with the latter also indicating the time-aggregated
    value. A set of plots is returned for each input channel.

    If binaural=true, a corresponding set of outputs for the binaural
    loudness are also contained in loudnessSHM.

    Assumptions
    -----------
    The input signal is calibrated to units of acoustic pressure in Pascals
    (Pa).

    """
    # %% Input checks
    p, chans_in, chans = shm_in_check(p, samp_rate_in, axis,
                                      soundfield, wait_bar, out_plot,
                                      binaural)

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
    # Section 5.1.4.1 Equation 9 ECMA-418-2:2025 [F(z)]
    band_centre_freqs = (delta_freq0/c)*np.sinh(c*half_bark)

    # Section 8.1.1 ECMA-418-2:2025
    weight_n = 0.5331  # Equations 113 & 114 ECMA-418-2:2025 [w_n]
    # Table 12 ECMA-418-2:2025
    a = 0.2918
    b = 0.5459

    # Output sample rate based on tonality hop sizes (Section 6.2.6
    # ECMA-418-2:2025) [r_sd]
    samp_rate1875 = samp_rate48k/256

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

    # Calculate specific loudnesses for tonal and noise components
    # ------------------------------------------------------------

    # Obtain tonal and noise component specific loudnesses from Sections 5 & 6 ECMA-418-2:2025
    tonality = shm_tonality_ecma(p_re, samp_rate48k, axis=0,
                                 soundfield=soundfield,
                                 wait_bar=wait_bar, out_plot=False)

    spec_tonal_loudness = tonality['spec_tonal_loudness']  # [N'_tonal(l,z)]
    spec_noise_loudness = tonality['spec_noise_loudness']  # [N'_noise(l,z)]

    # expand dimensions to ease processing
    if chans_in == 1:
        spec_tonal_loudness = shm_dimensional(spec_tonal_loudness, target_dim=3,
                                              where='last')
        spec_noise_loudness = shm_dimensional(spec_noise_loudness, target_dim=3,
                                              where='last')

    # Section 8.1.1 ECMA-418-2:2025
    # Weight and combine component specific loudnesses
    # pre-allocate array
    spec_loudness = np.zeros(spec_tonal_loudness.shape, order='F')
    for chan in range(chans_in):
        # Equation 114 ECMA-418-2:2025 [e(z)]
        max_loudness_funcel = a/(np.max(spec_tonal_loudness[:, :, chan]
                                        + spec_noise_loudness[:, :, chan], axis=1)
                                 + epsilon) + b
        max_loudness_funcel = shm_dimensional(max_loudness_funcel)
        # Equation 113 ECMA-418-2:2025 [N'(l,z)]
        spec_loudness[:, :, chan] = (spec_tonal_loudness[:, :,
                                                      chan]**max_loudness_funcel
                                     + weight_n*spec_noise_loudness[:, :, chan]
                                     ** max_loudness_funcel)**(1/max_loudness_funcel)
    # end of loudness for loop over channels

    if chans_in == 2 and binaural:
        expand_zeros = shm_dimensional(np.zeros(spec_tonal_loudness[:, :, 0].shape),
                                       target_dim=3, where='last')
        spec_loudness = np.concatenate((spec_loudness, expand_zeros), axis=2)
        spec_tonal_loudness = np.concatenate((spec_tonal_loudness, expand_zeros),
                                           axis=2)
        spec_noise_loudness = np.concatenate((spec_noise_loudness, expand_zeros),
                                            axis=2)
        # Binaural loudness
        # Section 8.1.5 ECMA-418-2:2025 Equation 118 [N'_B(l,z)]
        spec_loudness[:, :, 2] = np.sqrt(np.sum(spec_loudness[:, :, 0:2]**2,
                                                axis=2)/2)
        spec_tonal_loudness[:, :, 2] = np.sqrt(np.sum(spec_tonal_loudness[:, :, 0:2]**2,
                                                      axis=2)/2)
        spec_noise_loudness[:, :, 2] = np.sqrt(np.sum(spec_noise_loudness[:, :, 0:2]**2,
                                                      axis=2)/2)
        chans_out = 3  # set number of 'channels' to stereo plus single binaural
        chans = chans + ["Binaural"]
    else:
        chans_out = chans_in  # assign number of output channels
    # end of if branch for channels

    # Section 8.1.2 ECMA-418-2:2025
    # Time-averaged specific loudness Equation 115 [N'(z)]
    spec_loudness_powavg = (np.sum(spec_loudness[57:, :, :]**(1/np.log10(2)), 0)
                            / spec_loudness[57:, :, :].shape[0])**np.log10(2)

    # Section 8.1.3 ECMA-418-2:2025
    # Time-dependent loudness Equation 116 [N(l)]
    loudness_t = np.sum(spec_loudness*dz, axis=1)

    # Section 8.1.4 ECMA-418-2:2025
    # Overall loudness Equation 117 [N]
    loudness_powavg = (np.sum(loudness_t[57:, :]**(1/np.log10(2)), 0)
                       / loudness_t[57:, :].shape[0])**np.log10(2)

    # time (s) corresponding with results output [t]
    time_out = np.arange(0, spec_loudness.shape[0])/samp_rate1875

    # %% Output plotting

    # Plot figures
    # ------------
    if out_plot:
        # Plot results
        for chan in range(chans_out):
            # Plot results
            cmap_viridis = mpl.colormaps['viridis']
            chan_lab = chans[chan]
            fig, axs = plt.subplots(nrows=2, ncols=1, figsize=[10.5, 7.5],
                                    layout='constrained')

            ax1 = axs[0]
            pmesh = ax1.pcolormesh(time_out, band_centre_freqs,
                                   np.swapaxes(spec_loudness[:, :, chan], 0, 1),
                                   cmap=cmap_viridis,
                                   vmin=0,
                                   vmax=np.ceil(np.max(spec_loudness[:, :, chan])*10)/10,
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
                         label=(r"Specific loudness,"
                                "\n"
                                r"$\mathregular{tu_{SHM}}/\mathregular{Bark_{SHM}}$"),
                         aspect=10, cax=cbax)

            ax2 = axs[1]
            ax2.plot(time_out, loudness_powavg[chan]*np.ones(time_out.size),
                     color=cmap_viridis(33/255), linewidth=1, linestyle='dotted',
                     label=("Power" + "\n" + "time-" + "\n" + "average"))
            ax2.plot(time_out, loudness_t[:, chan],
                     color=cmap_viridis(165/255),
                     linewidth=0.75, label=("Time-" + "\n" + "dependent"))
            ax2.set(xlim=[time_out[0], time_out[-1] + time_out[1] - time_out[0]],
                    xlabel="Time, s",
                    ylim=[0, 1.1*np.ceil(np.max(loudness_t[:, chan])*10)/10],
                    ylabel=(r"Loudness, $\mathregular{sone_{SHM}}$"))
            ax2.grid(alpha=0.075, linestyle='--')
            ax2.legend(bbox_to_anchor=(1, 0.85), title="Overall")

            # Filter signal to determine A-weighted time-averaged level
            if chan == 2:
                pA = weight_A_t(p_re, fs=samp_rate48k)
                level_Aeq2 = np.max(20*np.log10(shm_rms(pA, axis=0)/2e-5))
                # take the higher channel level as representative (PD ISO/TS
                # 12913-3:2019 Annex D)
                level_Aeq = np.max(level_Aeq2)
                lr = np.argmax(level_Aeq2)
                # identify which channel is higher
                if lr == 0:
                    whichEar = " left ear"
                else:
                    whichEar = " right ear"
                # end of if branch to identify which channel is higher

                chan_lab = chan_lab + whichEar
            else:
                pA = weight_A_t(p_re[:, chan], fs=samp_rate48k)
                level_Aeq = 20*np.log10(shm_rms(pA)/2e-5)

            fig.suptitle(t=(chan_lab + " signal sound pressure level = " +
                            str(shm_round(level_Aeq, 1)) +
                            r" dB $\mathregular{\mathit{L}_{Aeq}}$"))
            fig.show()
        # end of for loop over channels
    # end of if branch for plotting

    # %% Output assignment

    # Discard singleton dimensions
    if chans_out == 1:
        spec_loudness = np.squeeze(spec_loudness)
        spec_tonal_loudness = np.squeeze(spec_tonal_loudness)
        spec_noise_loudness = np.squeeze(spec_noise_loudness)
        spec_loudness_powavg = np.squeeze(spec_loudness_powavg)
        loudness_t = np.squeeze(loudness_t)
    # end of if branch for singleton dimensions

    # Assign outputs to structure
    loudness = {}
    if chans_out == 3:
        loudness.update({'spec_loudness': spec_loudness[:, :, 0:2]})
        loudness.update({'spec_tonal_loudness': spec_tonal_loudness[:, :, 0:2]})
        loudness.update({'spec_noise_loudness': spec_noise_loudness[:, :, 0:2]})
        loudness.update({'spec_loudness_powavg': spec_loudness_powavg[:, 0:2]})
        loudness.update({'loudness_t': loudness_t[:, 0:2]})
        loudness.update({'loudness_powavg': loudness_powavg[0:2]})
        loudness.update({'spec_loudness_bin': spec_loudness[:, :, 2]})
        loudness.update({'spec_tonal_loudness_bin': spec_tonal_loudness[:, :, 2]})
        loudness.update({'spec_noise_loudness_bin': spec_noise_loudness[:, :, 2]})
        loudness.update({'spec_loudness_powavg_bin': spec_loudness_powavg[:, 2]})
        loudness.update({'loudness_t_bin': loudness_t[:, 2]})
        loudness.update({'loudness_powavg_bin': np.array(loudness_powavg[2])})
        loudness.update({'band_centre_freqs': band_centre_freqs})
        loudness.update({'time_out': time_out})
        loudness.update({'soundfield': soundfield})
    else:
        loudness.update({'spec_loudness': spec_loudness})
        loudness.update({'spec_tonal_loudness': spec_tonal_loudness})
        loudness.update({'spec_noise_loudness': spec_noise_loudness})
        loudness.update({'spec_loudness_powavg': spec_loudness_powavg})
        loudness.update({'loudness_t': loudness_t})
        loudness.update({'loudness_powavg': loudness_powavg})
        loudness.update({'band_centre_freqs': band_centre_freqs})
        loudness.update({'time_out': time_out})
        loudness.update({'soundfield': soundfield})

    return loudness
# end of shm_loudness_ecma function


# %% shm_loudness_ecma_from_comp
def shm_loudness_ecma_from_comp(spec_tonal_loudness, spec_noise_loudness,
                                out_plot=False, binaural=True):
    """
    Inputs
    ------
    spec_tonal_loudness : 2D or 3D array
                          the specific tonal loudness values calculated for
                          a sound pressure signal (single mono or single
                          stereo audio) arranged as [time, bands(, chans)]

    spec_noise_loudness : 2D or 3D array
                          the specific noise loudness values calculated for
                          a sound pressure signal (single mono or single
                          stereo audio) arranged as [time, bands(, chans)]

    out_plot : Boolean (default: False)
               flag indicating whether to generate a figure from the output
               (set outplot to false for doing multi-file parallel calculations)

    binaural : Boolean (default: True)
               flag indicating whether to output combined binaural loudness for
               stereo input signal

    Returns
    -------
    loudness : dict
               contains the output

    loudness contains the following outputs:

    spec_loudness : 2D or 3D array
                    time-dependent specific loudness for each critical band
                    arranged as [time, bands(, channels)]

    spec_loudness_powavg : 1D or 2D array
                           time-averaged specific loudness for each critical band
                           arranged as [bands(, channels)]

    spec_tonal_loudness : 2D or 3D array
                          time-dependent specific tonal loudness for each
                          critical band
                          arranged as [time, bands(, channels)]

    spec_noise_loudness : 2D or 3D array
                          time-dependent specific noise loudness for each
                          critical band
                          arranged as [time, bands(, channels)]

    loudness_t : 1D or 2D array
                 time-dependent overall loudness
                 arranged as [time(, channels)]

    loudness_powavg : 1D or 2D array
                      time-averaged overall loudness
                      arranged as [loudness(, channels)]

    band_centre_freqs : 1D array
                        centre frequencies corresponding with each critical band
                        rate

    time_out : 1D array
               time (seconds) corresponding with time-dependent outputs

    If out_plot=True, a set of plots is returned illustrating the time-dependent
    specific and overall loudness, with the latter also indicating the
    time-aggregated value. A set of plots is returned for each input channel.

    If binaural=true, a corresponding set of outputs for the binaural
    loudness are also contained in loudness.

    Assumptions
    -----------
    The input arrays are ECMA-418-2:2025 specific tonal and specific noise
    loudness, with dimensions orientated as [critical bands, time blocks,
    signal channels]

    """
    # %% Input checks

    # ensure inputs are arrays
    try:
        spec_tonal_loudness = np.asarray(spec_tonal_loudness)
        spec_noise_loudness = np.asarray(spec_noise_loudness)
    except TypeError:
        raise TypeError("Inputs must be arrays or capable of being cast to arrays.")

    # Check the size of the input matrices (must match)
    if spec_tonal_loudness.shape != spec_noise_loudness.shape:
        raise ValueError('Error: Input loudness array shapes must match')
    # end

    # ensure that inputs are 3D
    spec_tonal_loudness = shm_dimensional(spec_tonal_loudness, target_dim=3,
                                          where='last')
    spec_noise_loudness = shm_dimensional(spec_noise_loudness, target_dim=3,
                                          where='last')

    if spec_tonal_loudness.shape[2] > 2:
        raise ValueError('Error: Input matrices cannot comprise more than two channels.')
    else:
        chans_in = spec_tonal_loudness.shape[2]
        if chans_in > 1:
            chans = ["Stereo left",
                     "Stereo right"]
        else:
            chans = ["Mono"]
        # end of if branch for more than one channel
    # end of if branch for maximum channel number check

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
    # Section 5.1.4.1 Equation 9 ECMA-418-2:2025 [F(z)]
    band_centre_freqs = (delta_freq0/c)*np.sinh(c*half_bark)

    # Section 8.1.1 ECMA-418-2:2025
    weight_n = 0.5331  # Equations 113 & 114 ECMA-418-2:2025 [w_n]
    # Table 12 ECMA-418-2:2025
    a = 0.2918
    b = 0.5459

    # Output sample rate based on tonality hop sizes (Section 6.2.6
    # ECMA-418-2:2025) [r_sd]
    samp_rate1875 = samp_rate48k/256

    # Footnote 14 (/0 epsilon)
    epsilon = 1e-12

    # %% Signal processing

    # Section 8.1.1 ECMA-418-2:2025
    # Weight and combine component specific loudnesses
    spec_loudness = np.zeros(spec_tonal_loudness.shape, order='F')  # pre-allocate array
    for chan in range(chans_in):
        # Equation 114 ECMA-418-2:2025 [e(z)]
        max_loudness_funcel = a/(np.max(spec_tonal_loudness[:, :, chan]
                                      + spec_noise_loudness[:, :, chan], axis=1)
                               + epsilon) + b
        max_loudness_funcel = shm_dimensional(max_loudness_funcel)
        # Equation 113 ECMA-418-2:2025 [N'(l,z)]
        spec_loudness[:, :, chan] = (spec_tonal_loudness[:, :,
                                                         chan]**max_loudness_funcel
                                    + weight_n*spec_noise_loudness[:, :, chan]
                                    ** max_loudness_funcel)**(1/max_loudness_funcel)

    # end of loudness for loop over channels

    if chans_in == 2 and binaural:
        expand_zeros = shm_dimensional(np.zeros(spec_tonal_loudness[:, :, 0].shape),
                                       target_dim=3, where='last')
        spec_loudness = np.concatenate((spec_loudness, expand_zeros), axis=2)
        spec_tonal_loudness = np.concatenate((spec_tonal_loudness, expand_zeros),
                                              axis=2)
        spec_noise_loudness = np.concatenate((spec_noise_loudness, expand_zeros),
                                              axis=2)
        # Binaural loudness
        # Section 8.1.5 ECMA-418-2:2025 Equation 118 [N'_B(l,z)]
        spec_loudness[:, :, 2] = np.sqrt(np.sum(spec_loudness[:, :, 0:2]**2,
                                                axis=2)/2)
        spec_tonal_loudness[:, :, 2] = np.sqrt(np.sum(spec_tonal_loudness[:, :, 0:2]**2,
                                                      axis=2)/2)
        spec_noise_loudness[:, :, 2] = np.sqrt(np.sum(spec_noise_loudness[:, :, 0:2]**2,
                                                      axis=2)/2)
        chans_out = 3  # set number of 'channels' to stereo plus single binaural
        chans = chans + ["Binaural"]
    else:
        chans_out = chans_in  # assign number of output channels
    # end of if branch for channels

    # Section 8.1.2 ECMA-418-2:2025
    # Time-averaged specific loudness Equation 115 [N'(z)]
    spec_loudness_powavg = (np.sum(spec_loudness[57:, :, :]**(1/np.log10(2)), 0)
                            / spec_loudness[57:, :, :].shape[0])**np.log10(2)

    # Section 8.1.3 ECMA-418-2:2025
    # Time-dependent loudness Equation 116 [N(l)]
    loudness_t = np.sum(spec_loudness*dz, axis=1)

    # Section 8.1.4 ECMA-418-2:2025
    # Overall loudness Equation 117 [N]
    loudness_powavg = (np.sum(loudness_t[57:, :]**(1/np.log10(2)), 0)
                       / loudness_t[57:, :].shape[0])**np.log10(2)

    # time (s) corresponding with results output [t]
    time_out = np.arange(0, spec_loudness.shape[0])/samp_rate1875

    # %% Output plotting

    # Plot figures
    # ------------
    if out_plot:
        # Plot results
        for chan in range(chans_out):
            # Plot results
            cmap_viridis = mpl.colormaps['viridis']
            chan_lab = chans[chan]
            fig, axs = create_figure(nrows=2, ncols=1, figsize=[10.5, 7.5],
                                     layout='constrained')

            ax1 = axs[0]
            pmesh = ax1.pcolormesh(time_out, band_centre_freqs,
                                   np.swapaxes(spec_loudness[:, :, chan], 0, 1),
                                   cmap=cmap_viridis,
                                   vmin=0,
                                   vmax=np.ceil(np.max(spec_loudness[:, :, chan])*10)/10,
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
                         label=(r"Specific loudness,"
                                "\n"
                                r"$\mathregular{tu_{SHM}}/\mathregular{Bark_{SHM}}$"),
                         aspect=10, cax=cbax)

            ax2 = axs[1]
            ax2.plot(time_out, loudness_powavg[chan]*np.ones(time_out.size),
                     color=cmap_viridis(33/255), linewidth=1, linestyle='dotted',
                     label=("Power" + "\n" + "time-" + "\n" + "average"))
            ax2.plot(time_out, loudness_t[:, chan],
                     color=cmap_viridis(165/255),
                     linewidth=0.75, label=("Time-" + "\n" + "dependent"))
            ax2.set(xlim=[time_out[0], time_out[-1] + time_out[1] - time_out[0]],
                    xlabel="Time, s",
                    ylim=[0, 1.1*np.ceil(np.max(loudness_t[:, chan])*10)/10],
                    ylabel=(r"Loudness, $\mathregular{sone_{SHM}}$"))
            ax2.grid(alpha=0.075, linestyle='--')
            ax2.legend(bbox_to_anchor=(1.025, 0.8), loc='upper left', title="Overall")

            fig.suptitle(t=(chan_lab + " signal"))
            show_plot(fig)
        # end of for loop over channels
    # end of if branch for plotting

    # %% Output assignment

    # Discard singleton dimensions
    if chans_out == 1:
        spec_loudness = np.squeeze(spec_loudness)
        spec_tonal_loudness = np.squeeze(spec_tonal_loudness)
        spec_noise_loudness = np.squeeze(spec_noise_loudness)
        spec_loudness_powavg = np.squeeze(spec_loudness_powavg)
        loudness_t = np.squeeze(loudness_t)
    # end of if branch for singleton dimensions

    # Assign outputs to structure
    loudness = {}
    if chans_out == 3:
        loudness.update({'spec_loudness': spec_loudness[:, :, 0:2]})
        loudness.update({'spec_loudness_powavg': spec_loudness_powavg[:, 0:2]})
        loudness.update({'loudness_t': loudness_t[:, 0:2]})
        loudness.update({'loudness_powavg': loudness_powavg[0:2]})
        loudness.update({'spec_loudness_bin': spec_loudness[:, :, 2]})
        loudness.update({'spec_tonal_loudness_bin': spec_tonal_loudness[:, :, 2]})
        loudness.update({'spec_noise_loudness_bin': spec_noise_loudness[:, :, 2]})
        loudness.update({'spec_loudness_powavg_bin': spec_loudness_powavg[:, 2]})
        loudness.update({'loudness_t_bin': loudness_t[:, 2]})
        loudness.update({'loudness_powavg_bin': np.array(loudness_powavg[2])})
        loudness.update({'band_centre_freqs': band_centre_freqs})
        loudness.update({'time_out': time_out})
    else:
        loudness.update({'spec_loudness': spec_loudness})
        loudness.update({'spec_loudness_powavg': spec_loudness_powavg})
        loudness.update({'loudness_t': loudness_t})
        loudness.update({'loudness_powavg': loudness_powavg})
        loudness.update({'band_centre_freqs': band_centre_freqs})
        loudness.update({'time_out': time_out})

    return loudness
# end of shm_loudness_ecma_from_comp function
