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

Ownership and Quality Assurance
-------------------------------
Author: Mike JB Lotinga (m.j.lotinga@edu.salford.ac.uk)
Institution: University of Salford

Date created: 25/05/2023
Date last modified: 09/10/2025
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
import matplotlib as mpl
mpl.use('QtAgg')
from matplotlib import pyplot as plt
from sottek_hearing_model.shmSubs import (shmResample, shmPreProc,
                                          shmOutMidEarFilter,
                                          shmAuditoryFiltBank,
                                          shmSignalSegment,
                                          shmBasisLoudness,
                                          shmNoiseRedLowPass,
                                          shmRMS, shmRound,
                                          shmInCheck,
                                          shmCritBandAutoCorrelation,
                                          shmCritBandTonalityComponents)
from tqdm import tqdm
from sottek_hearing_model.filters import A_weight_T
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
            bandACFIter = tqdm(prange(61), desc="Critical band autocorrelation")
        else:
            bandACFIter = prange(61)

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
            bandACFAvgIter = tqdm(prange(nBands),
                                  desc="Component loudness estimation")
        else:
            bandACFAvgIter = prange(nBands)

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
            ax2.legend(bbox_to_anchor=(1, 0.85), title="Overall")

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
