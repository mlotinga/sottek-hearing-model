# -*- coding: utf-8 -*-
# %% Preamble
"""
shmTonalityECMA.py
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

shmTonalityECMA : This is the main tonality function, which implements sections
                  5 and 6 of ECMA-418-2:2025, and returns a dict containing
                  the tonality results as numpy arrays. This function is also
                  called by shmLoudnessECMA() to obtain Sottek Hearing Model
                  psychoacoustic loudness.

shmCritBandAutoCorrelation : A subfunction called by shmTonalityECMA(), which
                             returns the unbiased, normalised estimate of the
                             autocorrelation function for a critical band.
                             
shmCritBandTonalityComponents : A subfunction called by shmTonalityECMA(),
                                which returns the tonal loudness and noise
                                loudess components in each critical band, as
                                well as the band tonality frequencies.

Ownership and Quality Assurance
-------------------------------
Author: Mike JB Lotinga (m.j.lotinga@edu.salford.ac.uk)
Institution: University of Salford

Date created: 25/05/2023
Date last modified: 20/10/2025
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
# mpl.use('QtAgg')
from matplotlib import pyplot as plt
from sottek_hearing_model.shmSubs import (shmResample, shmPreProc,
                                          shmOutMidEarFilter,
                                          shmAuditoryFiltBank,
                                          shmSignalSegment,
                                          shmBasisLoudness,
                                          shmRMS, shmRound,
                                          shmInCheck)
from tqdm import tqdm
from sottek_hearing_model.filters import A_weight_T
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
n_threads = max(1, multiprocessing.cpu_count() - 1)  # leave one free core


# %% shmTonalityECMA
def shmTonalityECMA(p, sampleRateIn, axisN=0, soundField='freeFrontal',
                    waitBar=True, outPlot=False):
    """
    Inputs
    ------
    p : 1D or 2D array
        the input signal as single mono or stereo audio (sound pressure)
        signals

    sampleRateIn : integer
                   the sample rate (frequency) of the input signal(s)

    axisN : integer (0 or 1, default: 0)
            the time axis along which to calculate the tonality

    soundField : keyword string (default: 'freeFrontal')
                 determines whether the 'freeFrontal' or 'diffuse' field stages
                 are applied in the outer-middle ear filter, or 'noOuter' uses
                 only the middle ear stage, or 'noEar' omits ear filtering.
                 Note: these last two options are beyond the scope of the
                 standard, but may be useful if recordings made using
                 artificial outer/middle ear are to be processed using the
                 specific recorded responses.

    waitBar : keyword string (default: True)
              determines whether a progress bar displays during processing
              (set waitBar to false for doing multi-file parallel calculations)

    outPlot : Boolean (default: False)
              flag indicating whether to generate a figure from the output
              (set outplot to false for doing multi-file parallel calculations)

    Returns
    -------
    tonalitySHM : dict
                  contains the output

    tonalitySHM contains the following outputs:

    specTonality : 2D or 3D array
                   time-dependent specific tonality for each critical band
                   arranged as [time, bands(, channels)]

    specTonalityFreqs : 2D or 3D array
                        time-dependent frequencies of the dominant tonal
                        components corresponding with each of the
                        time-dependent specific tonality values in each
                        (half) critical band
                        arranged as [time, bands(, channels)]

    specTonalityAvg : 1D or 2D array
                      time-averaged specific tonality for each critical band
                      arranged as [bands(, channels)]

    specTonalityAvgFreqs : 1D or 2D array
                           frequencies of the dominant tonal components
                           corresponding with each of the
                           time-averaged specific tonality values in each
                           (half) critical band
                           arranged as [bands(, channels)]

    specTonalLoudness : 2D or 3D array
                        time-dependent specific tonal loudness for each
                        critical band
                        arranged as [time, bands(, channels)]

    specNoiseLoudness : 2D or 3D array
                        time-dependent specific noise loudness for each
                        critical band
                        arranged as [time, bands(, channels)]

    tonalityTDep : 1D or 2D array
                   time-dependent overall tonality
                   arranged as [time(, channels)]

    tonalityTDepFreqs : 1D or 2D array
                        time-dependent frequencies of the dominant tonal
                        components corresponding with the
                        time-dependent overall tonality values
                        arranged as [time(, channels)]

    tonalityAvg : 1D or 2D array
                  time-averaged overall tonality
                  arranged as [tonality(, channels)]

    bandCentreFreqs : 1D array
                      centre frequencies corresponding with each critical band
                      rate

    timeOut : 1D array
              time (seconds) corresponding with time-dependent outputs

    soundField : string
                 identifies the soundfield type applied (the input argument
                 soundField)

    If outPlot=True, a set of plots is returned illustrating the energy
    time-averaged A-weighted sound level, the time-dependent specific and
    overall tonality, with the latter also indicating the time-aggregated
    value. A set of plots is returned for each input channel.

    Assumptions
    -----------
    The input signal is calibrated to units of acoustic pressure in Pascals
    (Pa).

    Checked by:
    Date last checked:

    """
    # %% Input checks
    p, chansIn, chans = shmInCheck(p, sampleRateIn, axisN,
                                   soundField, waitBar, outPlot)

    # %% Define constants

    # Signal sample rate prescribed to be 48kHz (to be used for resampling),
    # Section 5.1.1 ECMA-418-2:2025 [r_s]
    sampleRate48k = 48e3
    # defined in Section 5.1.4.1 ECMA-418-2:2025 [deltaf(f=0)]
    deltaFreq0 = 81.9289
    # Half-overlapping Bark band centre-frequency denominator constant defined
    # in Section 5.1.4.1 ECMA-418-2:2025
    c = 0.1618

    dz = 0.5  # critical band overlap
    # half-overlapping critical band rate scale [z]
    halfBark = np.arange(0.5, 27, dz)
    nBands = len(halfBark)  # number of critical bands

    # Section 5.1.4.1 Equation 9 ECMA-418-2:2025 [F(z)]
    bandCentreFreqs = (deltaFreq0/c)*np.sinh(c*halfBark)
    # Section 5.1.4.1 Equation 10 ECMA-418-2:2025 [deltaf(z)]
    dfz = np.sqrt(deltaFreq0**2 + (c*bandCentreFreqs)**2)

    # Block and hop sizes Section 6.2.2 Table 4 ECMA-418-2:2025
    overlap = 0.75  # block overlap proportion
    # block sizes [s_b(z)]
    blockSize = np.hstack((8192*np.ones([3,]), 4096*np.ones([13,]),
                           2048*np.ones([9,]),
                           1024*np.ones([28,]))).astype(int)
    maxBlockSize = np.max(blockSize)

    # hop sizes (section 5.1.2 footnote 3 ECMA 418-2:2022) [s_h(z)]
    hopSize = ((1 - overlap)*blockSize).astype(int)

    # Output sample rate based on hop sizes - Resampling to common time basis
    # Section 6.2.6 ECMA-418-2:2025 [r_sd]
    sampleRate1875 = sampleRate48k/np.min(hopSize)

    # Number of bands that need averaging. Section 6.2.3 Table 5
    # ECMA-418-2:2025 [NB]
    # NBandsAvg = np.vstack((np.hstack((np.zeros([1, 1]), np.ones([1, 1]),
    #                                   2*np.ones([1, 14]), np.ones([1, 9]),
    #                                   np.zeros([1, 28]))),
    #                        np.hstack((np.ones([1, 1]), np.ones([1, 1]),
    #                                   2*np.ones([1, 14]), np.ones([1, 9]),
    #                                   np.zeros([1, 28]))))).astype(int)

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
    blockSizeDupe = (np.hstack((8192*np.ones([5,]), 4096*np.ones([17,]),
                                2048*np.ones([11,]),
                                1024*np.ones([28,])))).astype(int)

    bandCentreFreqsDupe = np.hstack((bandCentreFreqs[0:5],
                                     bandCentreFreqs[1:18],
                                     bandCentreFreqs[15:26],
                                     bandCentreFreqs[25:53]))

    # (duplicated) indices corresponding with the NB bands around each z band
    i_NBandsAvgDupe = np.vstack((np.hstack(([0, 0, 0], np.arange(5, 18),
                                            np.arange(22, 31),
                                            np.arange(33, 61))),
                                 np.hstack(([2, 3, 5], np.arange(10, 23),
                                            np.arange(25, 34),
                                            np.arange(34, 62)))))

    # Determine number of threads to use
    nThreads = min(nBands, n_threads)  # number of threads to use in parallel
    
    # %% Signal processing

    # Input pre-processing
    # --------------------
    if sampleRateIn != sampleRate48k:  # Resample signal
        p_re, _ = shmResample(p, sampleRateIn)
    else:  # don't resample
        p_re = p
    # end of if branch for resampling

    # Section 5.1.2 ECMA-418-2:2025 Fade in weighting and zero-padding
    pn = shmPreProc(p_re, maxBlockSize, np.max(hopSize))

    # Apply outer & middle ear filter
    # -------------------------------
    #
    # Section 5.1.3.2 ECMA-418-2:2025 Outer and middle/inner ear signal filtering
    pn_om = shmOutMidEarFilter(pn, soundField)

    # Loop through channels in file
    # -----------------------------
    if waitBar:
        chanIter = tqdm(range(chansIn), desc="Channels")
    else:
        chanIter = range(chansIn)

    # Equation 40 ECMA-418-2:2025 [l_end]
    endBlock = int(np.ceil(p_re.shape[0]/sampleRate48k*sampleRate1875))
    # pre-allocate results arrays
    # specSNR = np.zeros([endBlock + 1, nBands, chansIn])  # dev only
    # specLoudness = np.zeros([endBlock + 1, nBands, chansIn])  # dev only
    specTonalLoudness = np.zeros([endBlock + 1, nBands, chansIn], order='F')
    specNoiseLoudness = np.zeros([endBlock + 1, nBands, chansIn], order='F')
    specTonalityFreqs = np.zeros([endBlock + 1, nBands, chansIn], order='F')
    specTonalityAvg = np.zeros([nBands, chansIn], order='F')
    specTonalityAvgFreqs = np.zeros([nBands, chansIn], order='F')
    tonalityTDep = np.zeros([endBlock + 1, chansIn], order='F')
    tonalityTDepFreqs = np.zeros([endBlock + 1, chansIn], order='F')
    tonalityAvg = np.zeros([chansIn])

    for chan in chanIter:
        # Apply auditory filter bank
        # --------------------------
        # Filter equalised signal using 53 1/2Bark ERB filters according to
        # Section 5.1.4.2 ECMA-418-2:2025
        pn_omz = shmAuditoryFiltBank(pn_om[:, chan])

        # Autocorrelation function analysis
        # ---------------------------------
        # Duplicate Banded Data for ACF
        # Averaging occurs over neighbouring bands, to do this the segmentation
        # needs to be duplicated for neigbouring bands. 'Dupe' has been added
        # to variables to indicate that the vectors/matrices have been modified
        # for duplicated neigbouring bands.

        pn_omzDupe = np.hstack((pn_omz[:, 0:5], pn_omz[:, 1:18],
                                pn_omz[:, 15:26], pn_omz[:, 25:53]))

        if waitBar:
            bandACFIter = tqdm(range(61), desc="Critical band autocorrelation")
        else:
            bandACFIter = range(61)

        # pre-allocate arrays
        # basisLoudnessArray = np.empty(61, dtype=object)
        unbiasedNormACFDupe = np.empty(61, dtype=object)

        with ThreadPoolExecutor(max_workers=nThreads) as executor:
            futures = [executor.submit(shmCritBandAutoCorrelation, zBand,
                                       bandCentreFreqsDupe,
                                       blockSizeDupe, overlap, pn_omzDupe)
                       for zBand in bandACFIter]

        for future in as_completed(futures):
            zBand, unbiasedNormACFDupe_band = future.result()
            unbiasedNormACFDupe[zBand] = unbiasedNormACFDupe_band

        # Average the ACF over nB bands - Section 6.2.3 ECMA-418-2:2025
        if waitBar:
            bandACFAvgIter = tqdm(range(nBands),
                                  desc="Component loudness estimation")
        else:
            bandACFAvgIter = range(nBands)

        with ThreadPoolExecutor(max_workers=nThreads) as executor:
            futures = [executor.submit(shmCritBandTonalityComponents, zBand,
                                       bandCentreFreqs,
                                       blockSize, endBlock, dfz,#
                                       unbiasedNormACFDupe, i_NBandsAvgDupe)
                       for zBand in bandACFAvgIter]

        for future in as_completed(futures):
            (zBand, bandTonalLoudness,
             bandNoiseLoudness, bandTonalFreqs) = future.result()
            specTonalLoudness[:, zBand, chan] = bandTonalLoudness
            specNoiseLoudness[:, zBand, chan] = bandNoiseLoudness
            specTonalityFreqs[:, zBand, chan] = bandTonalFreqs

        # end of ACF averaging over bands
        
        # Ensure any tiny negative values are set to zero
        specTonalLoudness[specTonalLoudness < 0] = 0
        specNoiseLoudness[specNoiseLoudness < 0] = 0

        # Calculation of specific tonality
        # --------------------------------
        # Section 6.2.8 Equation 49 ECMA-418-2:2025 [SNR(l)]
        # loudness signal-noise-ratio
        overallSNR = (np.max(specTonalLoudness, axis=1)
                      / (np.sum(specNoiseLoudness, axis=1) + epsilon))

        # Section 6.2.8 Equation 50 ECMA-418-2:2025 [q(l)]
        crit = np.exp(-A*(overallSNR - B))
        ql = 1 - crit  # sigmoidal scaling factor
        ql[crit >= 1] = 0

        # Section 6.2.8 Equation 51 ECMA-418-2:2025 [T'(l,z)]
        # time-dependent specific tonality
        specTonality = cal_T*cal_Tx*np.swapaxes(np.tile(ql,
                                                        [53, 1, 1]),
                                                0, 1)*specTonalLoudness

        # Calculation of time-averaged specific tonality Section 6.2.9
        # ECMA-418-2:2025 [T'(z)]
        for zBand in range(nBands):
            # criterion Section 6.2.9 point 2
            mask = specTonality[:, zBand, chan] > 0.02
            mask[0:57] = False  # criterion Section 6.2.9 point 1

            # Section 6.2.9 Equation 53 ECMA-418-2:2025
            specTonalityAvg[zBand, chan] = np.sum(specTonality[mask,
                                                               zBand,
                                                               chan],
                                                  0)/(np.count_nonzero(mask)
                                                      + epsilon)
            specTonalityAvgFreqs[zBand, chan] = np.sum(specTonalityFreqs[mask,
                                                                         zBand,
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
        timeOut = np.arange(0, specTonality.shape[0])/sampleRate1875

        # Section 6.2.10 Equation 61 ECMA-418-2:2025
        # Time-dependent total tonality [T(l)]
        tonalityTDep[:, chan] = np.max(specTonality[:, :, chan], axis=1)
        zmax = np.argmax(specTonality[:, :, chan], axis=1)

        for ll in range(endBlock + 1):
            tonalityTDepFreqs[ll, chan] = specTonalityFreqs[ll, zmax[ll], chan]
        # end of total tonality for loop over blocks

        # Calculation of representative values Section 6.2.11 ECMA-418-2:2025
        # Time-averaged total tonality
        mask = tonalityTDep[:, chan] > 0.02  # criterion Section 6.2.9 point 2
        mask[0:57] = False    # criterion Section 6.2.9 point 1

        # Section 6.2.11 Equation 63 ECMA-418-2:2025
        # Time-averaged total tonality [T]
        tonalityAvg[chan] = np.sum(tonalityTDep[mask, chan],
                                   axis=0)/(np.count_nonzero(mask) + epsilon)

        # %% Output plotting

        # Plot figures
        # ------------
        if outPlot:
            # Plot results
            cmap_plasma = mpl.colormaps['plasma']
            chan_lab = chans[chan]
            fig, axs = plt.subplots(nrows=2, ncols=1, figsize=[10.5, 7.5],
                                    layout='constrained')

            ax1 = axs[0]
            pmesh = ax1.pcolormesh(timeOut, bandCentreFreqs,
                                   np.swapaxes(specTonality[:, :, chan], 0, 1),
                                   cmap=cmap_plasma,
                                   vmin=0,
                                   vmax=np.ceil(np.max(tonalityTDep[:,
                                                                    chan])*10)/10,
                                   shading='gouraud')
            ax1.set(xlim=[timeOut[1],
                          timeOut[-1] + (timeOut[1] - timeOut[0])],
                    xlabel="Time, s",
                    ylim=[bandCentreFreqs[0], bandCentreFreqs[-1]],
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
            ax2.plot(timeOut, tonalityAvg[chan]*np.ones(timeOut.size),
                     color=cmap_plasma(33/255), linewidth=1, linestyle='dotted',
                     label=("Time-" + "\n" + "average"))
            ax2.plot(timeOut, tonalityTDep[:, chan],
                     color=cmap_plasma(165/255),
                     linewidth=0.75, label=("Time-" + "\n" + "dependent"))
            ax2.set(xlim=[timeOut[0], timeOut[-1] + timeOut[1] - timeOut[0]],
                    xlabel="Time, s",
                    ylim=[0, 1.1*np.ceil(np.max(tonalityTDep[:, chan])*10)/10],
                    ylabel=(r"Tonality, $\mathregular{tu_{SHM}}$"))
            ax2.grid(alpha=0.075, linestyle='--')
            ax2.legend(bbox_to_anchor=(1.05, 0.85), loc='upper left', title="Overall")

            # Filter signal to determine A-weighted time-averaged level
            pA = A_weight_T(p_re[:, chan], fs=sampleRate48k)
            LAeq = 20*np.log10(shmRMS(pA)/2e-5)
            fig.suptitle(t=(chan_lab + " signal sound pressure level = " +
                            str(shmRound(LAeq, 1)) +
                            r" dB $\mathregular{\mathit{L}_{Aeq}}$"))
            fig.show()

        # end of if branch for plotting
    # end of for loop over channels

    # %% Output assignment

    # Discard singleton dimensions
    if chansIn == 1:
        specTonality = np.squeeze(specTonality)
        specTonalityAvg = np.squeeze(specTonalityAvg)
        specTonalityFreqs = np.squeeze(specTonalityFreqs)
        specTonalityAvgFreqs = np.squeeze(specTonalityAvgFreqs)
        tonalityTDep = np.squeeze(tonalityTDep)
        tonalityTDepFreqs = np.squeeze(tonalityTDepFreqs)
    # end of if branch for singleton dimensions

    # Assign outputs to structure
    tonalitySHM = {}
    tonalitySHM.update({'specTonality': specTonality})
    tonalitySHM.update({'specTonalityAvg': specTonalityAvg})
    tonalitySHM.update({'specTonalityFreqs': specTonalityFreqs})
    tonalitySHM.update({'specTonalityAvgFreqs': specTonalityAvgFreqs})
    tonalitySHM.update({'specTonalLoudness': specTonalLoudness})
    tonalitySHM.update({'specNoiseLoudness': specNoiseLoudness})
    tonalitySHM.update({'tonalityTDep': tonalityTDep})
    tonalitySHM.update({'tonalityAvg': tonalityAvg})
    tonalitySHM.update({'tonalityTDepFreqs': tonalityTDepFreqs})
    tonalitySHM.update({'bandCentreFreqs': bandCentreFreqs})
    tonalitySHM.update({'timeOut': timeOut})
    tonalitySHM.update({'soundField': soundField})

    return tonalitySHM
# end of shmTonalityECMA function


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


def shmCritBandAutoCorrelation(zBand, bandCentreFreqs, blockSizeBands, overlap, signalBands):
    """shmCritBandAutoCorrelation(zBand, bandCentreFreqs, blockSizeBands, overlap, signalBands)

    Returns the critical band autocorrelation estimate.

    Inputs
    ------
    zBand : integer
            the input critical band index
    bandCentreFreqs : array of floats
                      the centre frequencies of the critical bands
    blockSizeBands : array of integers
                     the block size in samples for each critical band
    overlap : float
              the proportion of overlap
    signalBands : 2D array
                  the input critical bandpass-filtered signals

    Returns
    -------
    zBand : integer
            the input critical band index as output
    unbiasedNormBandACF : 2D array
                          unbiased and normalised estimate of the autocorrelation
                          function in the corresponding critical band

    """

    # %% Define constants
    epsilon = 1e-12  # Footnote 14 (/0 epsilon)
    

    # %% Signal processing

    blockSize = blockSizeBands[zBand]

    # Segmentation into blocks
    # ------------------------
    # Section 5.1.5 ECMA-418-2:2025
    i_start = blockSizeBands[0] - blockSize
    signalSeg, _ = shmSignalSegment(signalBands[:, zBand],
                                    blockSize=blockSize,
                                    overlap=overlap,
                                    i_start=i_start)

    # Transformation into Loudness
    # ----------------------------
    # Sections 5.1.6 to 5.1.9 ECMA-418-2:2025 [N'_basis(z)]
    signalRectSeg, bandBasisLoudness, _ = shmBasisLoudness(signalSegmented=signalSeg,
                                                           bandCentreFreq=bandCentreFreqs[zBand])

    # ACF implementation using DFT
    # Section 6.2.2 Equations 27 & 28 ECMA-418-2:2025
    # [phi_unscaled,l,z(m)]
    unscaledACF = np.asfortranarray(np.fft.irfft(np.abs(np.fft.rfft(signalRectSeg,
                                                                    n=2*blockSize,
                                                                    axis=0))**2,
                                                 n=2*blockSize,
                                                 axis=0))

    # Section 6.2.2 Equation 29 ECMA-418-2:2025 [phi_l,z(m)]
    denom = (np.sqrt(np.flip(np.cumsum(np.flip(signalRectSeg, axis=0)**2,
                                       axis=0), axis=0)
                     * np.flip(np.cumsum(signalRectSeg**2, axis=0), axis=0))
             + epsilon)

    # note that the block length is used here, rather than the 2*s_b,
    # for compatability with the remaining code - beyond 0.75*s_b is
    # assigned (unused) zeros in the next line
    unbiasedNormACF = unscaledACF[0:blockSize, :]/denom
    unbiasedNormACF[int(0.75*blockSize):blockSize, :] = 0

    unbiasedNormACFBand = unbiasedNormACF*bandBasisLoudness

    return zBand, unbiasedNormACFBand
# end of shmCritBandAutoCorrelation function


def shmCritBandTonalityComponents(zBand, bandCentreFreqs, blockSizeBands, lastBlock, dfz, unbiasedNormACF, i_NBandsAvg):
    """shmCritBandTonalityComponents(zBand, bandCentreFreqs, blockSizeBands, lastBlock, dfz, unbiasedNormACF, i_NBandsAvg)

    Returns the tonal loudness, noise loudness and tonality frequencies for the critical band.

    Inputs
    ------
    zBand : integer
            the input critical band index
    bandCentreFreqs : array of floats
                      the centre frequencies of the critical bands
    blockSizeBands : array of integers
                     the block size in samples for each critical band
    lastBlock : integer
                the index of the last block
    unbiasedNormACF : 2D array
                      the unbiased and normalised estimate of the autocorrelation
                      function in the corresponding critical band
    i_NBandsAvg : array of integers
                  the indices for averaging adjacent critical bands

    Returns
    -------
    zBand : integer
            the input critical band index as output
    bandTonalLoudness : 2D array
                        the specific loudness of the tonal component in the critical band

    bandNoiseLoudness : 2D array
                        the specific loudness of the noise component in the critical band

    bandTonalFreqs : 2D array
                     the time-dependent frequencies of the tonal components in the critical band
    """

    # %% Define constants
    epsilon = 1e-12

    # Noise reduction constants from Section 6.2.7 Table 7 ECMA-418-2:2025
    alpha = 20
    beta = 0.07

    # sample rates
    sampleRate1875 = 187.5
    sampleRate48k = 48e3

    # largest block size
    maxBlockSize = np.max(blockSizeBands)

    # Critical band interpolation factors from Section 6.2.6 Table 6
    # ECMA-418-2:2025 [i]
    i_interp = blockSizeBands/np.min(blockSizeBands)

    # Sigmoid function factor parameters Section 6.2.7 Table 8 ECMA-418-2:2025
    # [c(s_b(z))]
    csz_b = np.hstack((18.21*np.ones([3,]), 12.14*np.ones([13,]),
                       417.54*np.ones([9,]), 962.68*np.ones([28,])))
    # [d(s_b(z))]
    dsz_b = np.hstack((0.36*np.ones([3,]), 0.36*np.ones([13,]),
                       0.71*np.ones([9,]), 0.69*np.ones([28,])))


    # %% Signal processing
    # Averaging of frequency bands
    meanScaledACF = np.mean(unbiasedNormACF[i_NBandsAvg[0,
                                                        zBand]:i_NBandsAvg[1,
                                                                           zBand]],
                            axis=0)

    # Average the ACF over adjacent time blocks [phibar_z'(m)]
    if zBand < 16:
        meanScaledACF = np.asfortranarray(np.roll(np.nan_to_num(bn.move_mean(meanScaledACF,
                                                                             window=3,
                                                                             min_count=3,
                                                                             axis=1),
                                                                copy=False),
                                                  shift=-1, axis=1))
    # end of if branch for moving mean over time blocks

    # Application of ACF lag window Section 6.2.4 ECMA-418-2:2025
    tauz_start = max(0.5/dfz[zBand], 2e-3)  # Equation 31 ECMA-418-2:2025 [tau_start(z)]
    tauz_end = max(4/dfz[zBand], tauz_start + 1e-3)  # Equation 32 ECMA-418-2:2025 [tau_end(z)]
    # Equations 33 & 34 ECMA-418-2:2025
    mz_start = int(np.ceil(tauz_start*sampleRate48k) - 1)  # Starting lag window index [m_start(z)]
    mz_end = int(np.floor(tauz_end*sampleRate48k) - 1)  # Ending lag window index [m_end(z)]
    M = mz_end - mz_start + 1
    # Equation 35 ECMA-418-2:2025
    # lag-windowed, detrended ACF [phi'_z,tau(m)]
    lagWindowACF = np.zeros(meanScaledACF.shape, order='F')
    lagWindowACF[mz_start:mz_end + 1, :] = (meanScaledACF[mz_start:mz_end
                                                          + 1, :]
                                            - np.mean(meanScaledACF[mz_start:mz_end
                                                                    + 1,
                                                                    :],
                                                      axis=0))

    # Estimation of tonal loudness
    # ----------------------------
    # Section 6.2.5 Equation 36 ECMA-418-2:2025
    # ACF spectrum in the lag window [Phi'_z,tau(k)]
    magFFTlagWindowACF = np.abs(np.fft.fft(lagWindowACF,
                                           n=2*maxBlockSize,
                                           axis=0))

    magFFTlagWindowACF[np.isnan(magFFTlagWindowACF, order='F')] = 0.0

    # added to avoid spurious tiny results affecting tonal frequency
    # identification
    magFFTlagWindowACF[magFFTlagWindowACF <= epsilon] = 0.0

    # Section 6.2.5 Equation 37 ECMA-418-2:2025 [Nhat'_tonal(z)]
    # first estimation of specific loudness of tonal component in critical band
    bandTonalLoudness = meanScaledACF[0, :].copy()
    mask = (2*np.max(magFFTlagWindowACF, axis=0)/(M/2)
            <= meanScaledACF[0, :])
    bandTonalLoudness[mask] = 2*np.max(magFFTlagWindowACF[:, mask],
                                       axis=0)/(M/2)

    # Section 6.2.5 Equation 38 & 39 ECMA-418-2:2025
    # [k_max(z)]
    # NOTE: round is used to resolve discrepancies due to tiny floating point values at spectral line pairs
    kz_max = np.argmax(np.round(magFFTlagWindowACF, 8), axis=0)
    # frequency of maximum tonal component in critical band [f_ton(z)]
    bandTonalFreqs = kz_max*(sampleRate48k/(2*maxBlockSize))

    # Section 6.2.7 Equation 41 ECMA-418-2:2025 [N'_signal(l,z)]
    # specific loudness of complete band-pass signal in critical band
    bandLoudness = meanScaledACF[0, :].copy()

    # Resampling to common time basis Section 6.2.6 ECMA-418-2:2025
    if i_interp[zBand] > 1:
        # Note: use of interpolation function avoids rippling caused by
        # resample function, which otherwise affects specific loudness
        # calculation for tonal and noise components
        l_n = meanScaledACF.shape[1]
        xp = np.linspace(0, l_n - 1, l_n)
        x = np.linspace(0, l_n - 1, int(i_interp[zBand]*(l_n - 1) + 1))

        bandTonalLoudness = np.interp(x, xp, bandTonalLoudness)
        bandLoudness = np.interp(x, xp, bandLoudness)
        bandTonalFreqs = np.interp(x, xp, bandTonalFreqs)

    # end of if branch for interpolation

    # Remove end zero-padded samples Section 6.2.6 ECMA-418-2:2025
    bandTonalLoudness = bandTonalLoudness[0:lastBlock + 1]
    bandLoudness = bandLoudness[0:lastBlock + 1]
    bandTonalFreqs = bandTonalFreqs[0:lastBlock + 1]

    # Noise reduction Section 6.2.7 ECMA-418-2:2020
    # ---------------------------------------------
    # Equation 42 ECMA-418-2:2025 signal-noise-ratio first approximation
    # (ratio of tonal component loudness to non-tonal component
    # loudness in critical band)
    # [SNRhat(l,z)]
    SNRlz1 = bandTonalLoudness/((bandLoudness - bandTonalLoudness)
                                      + epsilon)

    # Equation 43 ECMA-418-2:2025 low pass filtered specific loudness
    # of non-tonal component in critical band [Ntilde'_tonal(l,z)]
    bandTonalLoudness = shmNoiseRedLowPass(bandTonalLoudness,
                                                sampleRate1875)

    # Equation 44 ECMA-418-2:2025 lowpass filtered SNR (improved
    # estimation)
    # [SNRtilde(l,z)]
    SNRlz = shmNoiseRedLowPass(SNRlz1, sampleRate1875)

    # Equation 46 ECMA-418-2:2025 [g(z)]
    gz = csz_b[zBand]/(bandCentreFreqs[zBand]**dsz_b[zBand])

    # Equation 45 ECMA-418-2:2025 [nr(l,z)]
    crit = np.exp(-alpha*((SNRlz/gz) - beta))
    nrlz = 1 - crit  # sigmoidal weighting function
    nrlz[crit >= 1] = 0

    # Equation 47 ECMA-418-2:2025 [N'_tonal(l,z)]
    bandTonalLoudness = nrlz*bandTonalLoudness

    # Section 6.2.8 Equation 48 ECMA-418-2:2025 [N'_noise(l,z)]
    # specific loudness of non-tonal component in critical band
    bandNoiseLoudness = (shmNoiseRedLowPass(bandLoudness,
                                            sampleRate1875)
                         - bandTonalLoudness)

    return (zBand,
            bandTonalLoudness,
            bandNoiseLoudness,
            bandTonalFreqs)

# end of shmCritBandTonalityComponents function
