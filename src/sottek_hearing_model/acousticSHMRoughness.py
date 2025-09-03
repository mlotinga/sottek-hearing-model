# -*- coding: utf-8 -*-
# %% Preamble
"""
acousticSHMRoughness.py
----------------------

Returns roughness values according to ECMA-418-2:2025 using the Sottek Hearing
Model) for an input calibrated single mono or single stereo audio (sound
pressure) time-series signal, p.

Requirements
------------
numpy
scipy
matplotlib
tqdm
refmap-psychoacoustics (metrics.ecma418_2, dsp.filterFuncs and
                        utils.formatFuncs)

Ownership and Quality Assurance
-------------------------------
Author: Mike JB Lotinga (m.j.lotinga@edu.salford.ac.uk)
Institution: University of Salford

Date created: 29/05/2023
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

"""

# %% Import block
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from matplotlib import pyplot as plt
import matplotlib as mpl
from scipy.fft import (fft)
from scipy.signal import (hilbert, windows, find_peaks)
from scipy.interpolate import PchipInterpolator
from src.py.metrics.ecma418_2.acousticSHMSubs import (shmDimensional,
                                                      shmResample,
                                                      shmPreProc,
                                                      shmOutMidEarFilter,
                                                      shmAuditoryFiltBank,
                                                      shmSignalSegmentBlocks,
                                                      shmSignalSegment,
                                                      shmBasisLoudness,
                                                      shmDownsample,
                                                      shmRoughWeight,
                                                      shmRoughLowPass,
                                                      shmRound, shmRMS)
from tqdm import tqdm
from src.py.dsp.filterFuncs import A_weight_T
from src.py.utils.formatFuncs import roundTrad

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


# %% acousticSHMRoughness
def acousticSHMRoughness(p, sampleRateIn, axisN=0, soundField='freeFrontal',
                         waitBar=True, outPlot=False, binaural=True):
    """
    Inputs
    ------
    p : 1D or 2D array
        the input signal as single mono or stereo audio (sound
        pressure) signals

    sampleRateIn : integer
                   the sample rate (frequency) of the input signal(s)

    axisN : integer (0 or 1, default: 0)
            the time axis along which to calculate the roughness

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

    binaural : Boolean (default: True)
               flag indicating whether to output combined binaural roughness
               for stereo input signal
    Returns
    -------
    roughnessSHM : dict
                  contains the output

    roughnessSHM contains the following outputs:

    specRoughness : 2D or 3D array
                    time-dependent specific roughness for each critical band
                    arranged as [time, bands(, channels)]

    specRoughnessAvg : 1D or 2D array
                      time-averaged specific roughness for each critical band
                      arranged as [bands(, channels)]

    roughnessTDep : 1D or 2D array
                   time-dependent overall roughness
                   arranged as [time(, channels)]

    roughness90pc : 1D or 2D array
                    time-aggregated (90th percentile) overall roughness
                    arranged as [roughness(, channels)]

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
    overall roughness, with the latter also indicating the time-aggregated
    value. A set of plots is returned for each input channel.

    If binaural=true, a corresponding set of outputs for the binaural
    loudness are also contained in roughnessSHM.

    Assumptions
    -----------
    The input signal is calibrated to units of acoustic pressure in Pascals
    (Pa).

    Checked by:
    Date last checked:

    """
    # %% Input checks
    # Orient input matrix
    if axisN in [0, 1]:
        if axisN == 1:
            p = p.T
    else:
        raise ValueError("Input axisN must be an integer 0 or 1")

    # Check the length of the input data (must be longer than 300 ms)
    if p.shape[0] <= 300/1000*sampleRateIn:
        raise ValueError('Input signal is too short along the specified axis to calculate tonality (must be longer than 300 ms)')

    # Check the channel number of the input data
    if p.shape[1] > 2:
        raise ValueError('Input signal comprises more than two channels')

    chansIn = p.shape[1]
    if chansIn > 1:
        chans = ["Stereo left",
                 "Stereo right"]
        if binaural:
            chansOut = 3
            chans += ["Binaural"]
        else:
            chansOut = chansIn
    else:
        chans = ["Mono"]
        chansOut = chansIn
    # end of if branch for channel number check

    # %% Define constants

    signalT = p.shape[0]/sampleRateIn  # duration of input signal
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

    # Block and hop sizes Section 6.2.2 Table 4 ECMA-418-2:2025
    overlap = 0.75  # block overlap proportion
    # block size [s_b(z)]
    blockSize = 16384
    # hop sizes (section 5.1.2 footnote 3 ECMA 418-2:2022) [s_h(z)]
    hopSize = int(((1 - overlap)*blockSize))

    # Downsampled block and hop sizes Section 7.1.2 ECMA-418-2:2025
    downSample = 32  # downsampling factor
    sampleRate1500 = sampleRate48k/downSample
    blockSize1500 = int(blockSize/downSample)
    # hopSize1500 = (1 - overlap)*blockSize1500
    # DFT resolution (section 7.1.5.1) [deltaf]
    resDFT1500 = sampleRate1500/blockSize1500

    # Modulation rate error correction values Table 8, Section 7.1.5.1
    # ECMA-418-2:2025 [E(theta)]
    errorCorrection = np.array([0.0000, 0.0457, 0.0907, 0.1346, 0.1765, 0.2157,
                                0.2515, 0.2828, 0.3084, 0.3269, 0.3364, 0.3348,
                                0.3188, 0.2844, 0.2259, 0.1351, 0.0000])
    errorCorrection = np.hstack((errorCorrection, np.flip(-errorCorrection[0:-1]), 0))

    # High modulation rate roughness perceptual scaling function
    # (section 7.1.5.2 ECMA-418-2:2025)
    # Table 11 ECMA-418-2:2025 [r_1; r_2]
    roughScaleParams = np.vstack(([0.3560, 0.8024], [0.8049, 0.9333]))
    roughScaleParams = np.vstack((roughScaleParams[:, 0]*np.ones([np.sum(bandCentreFreqs < 1e3), 2]),
                                  roughScaleParams[:, 1]*np.ones([np.sum(bandCentreFreqs >= 1e3), 2]))).T

    # Equation 84 ECMA-418-2:2025 [r_max(z)]
    roughScale = 1/(1 + roughScaleParams[0, :]
                    * np.abs(np.log2(bandCentreFreqs/1000))
                    ** roughScaleParams[1, :])
    # Note: this is to ease parallelised calculations
    roughScale = shmDimensional(roughScale, targetDim=3, where='first')

    # High/low modulation rate roughness perceptual weighting function
    # parameters (section 7.1.5.2 ECMA-418-2:2025)
    # Equation 86 ECMA-418-2:2025 [f_max(z)]
    modfreqMaxWeight = 72.6937*(1 -
                                1.1739*np.exp(-5.4583*bandCentreFreqs/1000))

    # Equation 87 ECMA-418-2:2025 [q_1; q_2(z)]
    roughHiWeightParams = np.vstack((1.2822*np.ones(bandCentreFreqs.size),
                                     0.2471*np.ones(bandCentreFreqs.size)))
    mask = bandCentreFreqs/1000 >= 2**-3.4253
    roughHiWeightParams[1, mask] = 0.2471 + 0.0129*(np.log2(bandCentreFreqs[mask]/1000)
                                                    + 3.4253)**2
    # Note: this is to ease parallelised calculations
    roughHiWeightParams = np.expand_dims(roughHiWeightParams, axis=1)

    # (section 7.1.5.4 ECMA-418-2:2025)
    # Equation 96 ECMA-418-2:2025 [q_1; q_2(z)]
    roughLoWeightParams = np.vstack((0.7066*np.ones(bandCentreFreqs.size),
                                     1.0967
                                     - 0.064*np.log2(bandCentreFreqs/1000)))
    roughLoWeightParams = np.expand_dims(roughLoWeightParams, axis=1)

    # Output sample rate (section 7.1.7 ECMA-418-2:2025) [r_s50]
    sampleRate50 = 50

    # Calibration constant
    # calibration factor in Section 7.1.7 Equation 104 ECMA-418-2:2025 [c_R]
    cal_R = 0.0180909
    cal_Rx = 1/1.0011565  # calibration adjustment factor

    # Footnote 14 (/0 epsilon)
    epsilon = 1e-12

    # %% Signal processing

    # Input pre-processing
    # --------------------
    if sampleRateIn != sampleRate48k:  # Resample signal
        p_re, _ = shmResample(p, sampleRateIn)
    else:  # don't resample
        p_re = p
    # end of if branch for resampling

    # Input signal samples
    n_samples = p_re.shape[0]

    # Section 7.1.7 Equation 103 [l_50,end]
    l_50Last = int(np.floor(n_samples/sampleRate48k*sampleRate50) + 1)

    # Section 5.1.2 ECMA-418-2:2025 Fade in weighting and zero-padding
    pn = shmPreProc(p_re, blockSize=blockSize, hopSize=hopSize, padStart=True,
                    padEnd=False)

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

    specRoughness = np.zeros([l_50Last, nBands, chansOut])
    for chan in chanIter:
        # Apply auditory filter bank
        # --------------------------
        # Filter equalised signal using 53 1/2Bark ERB filters according to
        # Section 5.1.4.2 ECMA-418-2:2025
        pn_omz = shmAuditoryFiltBank(pn_om[:, chan])

        # Note: At this stage, typical computer RAM limits impose a need to
        # loop through the critical bands rather than continue with a
        # parallelised approach, until later downsampling is applied

        if waitBar:
            envIter = tqdm(range(nBands), desc="Envelope extraction")
        else:
            envIter = range(nBands)

        # pre-allocate output arrays
        i_start = 0  # index to start segmentation block processing from
        _, nBlocks, _ = shmSignalSegmentBlocks(pn_omz[:, 0],
                                               blockSize=blockSize,
                                               overlap=overlap, axisN=0,
                                               i_start=i_start,
                                               endShrink=True)
        basisLoudness = np.zeros([nBlocks, nBands])
        envelopes = np.zeros([blockSize1500, nBlocks, nBands])
        for zBand in envIter:
            # Segmentation into blocks
            # ------------------------

            # Section 5.1.5 ECMA-418-2:2025
            pn_lz, lBlocksOut = shmSignalSegment(pn_omz[:, zBand],
                                                 blockSize=blockSize,
                                                 overlap=overlap, axisN=0,
                                                 i_start=i_start,
                                                 endShrink=True)

            # Transformation into Loudness
            # ----------------------------
            # Sections 5.1.6 to 5.1.9 ECMA-418-2:2025
            _, bandBasisLoudness, _ = shmBasisLoudness(signalSegmented=pn_lz.copy(),
                                                       bandCentreFreq=bandCentreFreqs[zBand])
            basisLoudness[:, zBand] = bandBasisLoudness

            # Envelope power spectral analysis
            # --------------------------------
            # Sections 7.1.2 ECMA-418-2:2025
            # magnitude of Hilbert transform with downsample - Equation 65
            # [p(ntilde)_E,l,z]
            envelopes[:, :, zBand] = shmDownsample(np.abs(hilbert(pn_lz,
                                                                  axis=0)),
                                                   axisN=0, downSample=32)

        # end of for loop for obtaining low frequency signal envelopes

        # Note: With downsampled envelope signals,
        # parallelised approach can continue

        # Section 7.1.3 equation 66 ECMA-418-2:2025 [Phi(k)_E,l,z]
        modSpectra = np.zeros(envelopes.shape)
        envelopeWin = envelopes*np.tile(shmDimensional(windows.hann(blockSize1500,
                                                                    sym=False),
                                                       targetDim=3),
                                        reps=[1, envelopes.shape[1],
                                              nBands])/np.sqrt(0.375)
        # Equation 66 & 67
        denom = shmDimensional(np.max(basisLoudness, 1))*np.sum(envelopeWin**2,
                                                                0)
        mask = denom != 0  # Equation 66 criteria for masking
        maskRep = np.broadcast_to(mask, modSpectra.shape)  # broadcast mask
        scaling = np.divide(basisLoudness**2, denom, out=np.zeros_like(denom),
                            where=mask)  # Equation 66 factor
        # broadcast scaling
        scalingRep = np.broadcast_to(scaling, maskRep.shape)
        modSpectra[maskRep] = np.ravel((np.reshape(scalingRep[maskRep],
                                                   (blockSize1500, -1, nBands))
                                        * np.abs(fft(np.reshape(envelopeWin[maskRep],
                                                                (blockSize1500,
                                                                 -1, nBands)),
                                                     axis=0))**2))

        # Envelope noise reduction
        # ------------------------
        # section 7.1.4 ECMA-418-2:2025
        modSpectraAvg = sliding_window_view(modSpectra, window_shape=3,
                                            axis=2).mean(axis=3)
        modSpectraAvg = np.concatenate((shmDimensional(modSpectra[:, :, 0],
                                                       targetDim=3,
                                                       where='last'),
                                        modSpectraAvg,
                                        shmDimensional(modSpectra[:, :, -1],
                                                       targetDim=3,
                                                       where='last')), axis=2)

        # Equation 68 [s(l,k)]
        modSpectraAvgSum = np.sum(modSpectraAvg, axis=2)

        # Equation 71 ECMA-418-2:2025 [wtilde(l,k)]
        clipWeight = (0.0856*modSpectraAvgSum[0:int(modSpectraAvg.shape[0]/2) + 1, :]
                      / (np.median(modSpectraAvgSum[2:int(modSpectraAvg.shape[0]/2), :],
                                   axis=0) + 1e-10)
                      * shmDimensional(np.minimum(np.maximum(0.1891*np.exp(0.012*np.arange(0,
                                                                                           int(modSpectraAvg.shape[0]/2)
                                                                                           + 1)),
                                                             0),
                                                  1)))

        # Equation 70 ECMA-418-2:2025 [w(l,k)]
        weightingFactor1 = np.zeros(modSpectraAvgSum[0:257, :].shape)
        mask = clipWeight >= 0.05*np.max(clipWeight[2:256, :], axis=0)
        weightingFactor1[mask] = np.minimum(np.maximum(clipWeight[mask]
                                                       - 0.1407, 0), 1)
        weightingFactor = np.concatenate((weightingFactor1,
                                          np.flipud(weightingFactor1[1:256,
                                                                     :])),
                                         axis=0)

        # Calculate noise-reduced, scaled, weighted modulation power spectra
        # Equation 69 [Phihat(k)_E,l,z]
        modWeightSpectraAvg = modSpectraAvg*shmDimensional(weightingFactor,
                                                           targetDim=3)

        # Spectral weighting
        # ------------------
        # Section 7.1.5 ECMA-418-2:2025
        # theta used in equation 79, including additional index for
        # errorCorrection terms from table 10
        theta = np.arange(34)
        modAmp = np.zeros([10, nBlocks, nBands])
        modRate = np.zeros([10, nBlocks, nBands])

        if waitBar:
            rateIter = tqdm(range(nBands),
                            desc="Modulation rates")
        else:
            rateIter = range(nBands)

        for zBand in rateIter:
            # Section 7.1.5.1 ECMA-418-2:2025
            for lBlock in range(nBlocks):
                # identify peaks in each block (for each band)
                startIdx = 2
                endIdx = 255
                modWeightSpectraAvgBandBlock = modWeightSpectraAvg[startIdx:endIdx,
                                                                   lBlock,
                                                                   zBand]
                kLocs, pkProps = find_peaks(modWeightSpectraAvgBandBlock,
                                            prominence=0)
                PhiPks = modWeightSpectraAvgBandBlock[kLocs]
                proms = pkProps['prominences']

                # reindex kLocs to match spectral start index used in findpeaks
                # for indexing into modulation spectra matrices
                kLocs = kLocs + startIdx

                # we canonly have peaks at k = 3:254
                mask = np.isin(kLocs, range(3, 255))
                kLocs = kLocs[mask]
                PhiPks = PhiPks[mask]

                # consider 10 highest prominence peaks only
                if len(proms) > 10:
                    promsSorted = np.sort(proms)[::-1]
                    iiSort = np.argsort(proms)[::-1]
                    mask = proms >= promsSorted[9]

                    # if branch to deal with duplicated peak prominences
                    if sum(mask) > 10:
                        mask = mask[iiSort <= 9]
                    # end of if branch for duplicated peak prominences

                    PhiPks = PhiPks[mask]
                    kLocs = kLocs[mask]

                # end of if branch to select 10 highest prominence peaks

                # consider peaks meeting criterion
                if PhiPks.size != 0:
                    mask = PhiPks > 0.05*np.max(PhiPks)  # Equation 72 criterion
                    PhiPks = PhiPks[mask]  # [Phihat(k_p,i(l,z))]
                    kLocs = kLocs[mask]
                    # loop over peaks to obtain modulation rates
                    for iPeak in range(len(PhiPks)):
                        # Equation 74 ECMA-418-2:2025
                        # [Phihat_E,l,z]
                        modAmpMat = np.vstack((modWeightSpectraAvg[kLocs[iPeak] - 1,
                                                                   lBlock, zBand],
                                               modWeightSpectraAvg[kLocs[iPeak],
                                                                   lBlock, zBand],
                                               modWeightSpectraAvg[kLocs[iPeak] + 1,
                                                                   lBlock, zBand]))

                        # Equation 82 [A_i(l,z)]
                        modAmp[iPeak, lBlock, zBand] = np.sum(modAmpMat)

                        # Equation 75 ECMA-418-2:2025
                        # [K]
                        modIndexMat = np.vstack((np.hstack(((kLocs[iPeak] - 1)**2,
                                                            kLocs[iPeak] - 1, 1)),
                                                 np.hstack(((kLocs[iPeak])**2,
                                                            kLocs[iPeak], 1)),
                                                 np.hstack(((kLocs[iPeak] + 1)**2,
                                                            kLocs[iPeak] + 1, 1))))

                        # Equation 73 solution [C]
                        coeffVec = np.linalg.solve(modIndexMat, modAmpMat)

                        # Equation 76 ECMA-418-2:2025 [ftilde_p,i(l,z)]
                        modRateEst = (-(coeffVec[1]/(2*coeffVec[0]))*resDFT1500).item(0)

                        # Equation 79 ECMA-418-2:2025 [beta(theta)]
                        errorBeta = ((np.floor(modRateEst/resDFT1500)
                                      + theta[:33]/32)*resDFT1500
                                     - (modRateEst
                                        + errorCorrection[theta[:33]]))

                        # Equation 80 ECMA-418-2:2025 [theta_min]
                        thetaMinError = np.argmin(np.abs(errorBeta))

                        # Equation 81 ECMA-418-2:2025 [theta_corr]
                        if (thetaMinError > 0) and (errorBeta[thetaMinError]*errorBeta[thetaMinError
                                                                                       - 1]
                                                    < 0):
                            thetaCorr = thetaMinError
                        else:
                            thetaCorr = thetaMinError + 1
                        # end of eq 81 if-branch

                        # Equation 78 ECMA-418-2:2025
                        # [rho(ftilde_p,i(l,z))]
                        biasAdjust = (errorCorrection[thetaCorr - 1]
                                      - (errorCorrection[thetaCorr]
                                         - errorCorrection[thetaCorr - 1])
                                      * errorBeta[thetaCorr - 1]
                                      / (errorBeta[thetaCorr]
                                         - errorBeta[thetaCorr - 1]))

                        # Equation 77 ECMA-418-2:2025 [f_p,i(l,z)]
                        modRate[iPeak, lBlock, zBand] = modRateEst + biasAdjust

                    # end of for loop over peaks in block per band
                # end of if branch for detected peaks in modulation spectrum
            # end of for loop over blocks for peak detection
        # end  of for loop over bands for modulation spectral weighting

        # Section 7.1.5.2 ECMA-418-2:2025 - Weighting for high modulation rates
        # Equation 85 [G_l,z,i(f_p,i(l,z))]
        roughHiWeight = shmRoughWeight(modRate, modfreqMaxWeight,
                                       roughHiWeightParams)

        # Equation 83 [Atilde_i(l,z)]
        modAmpHiWeight = modAmp*roughScale
        mask = modRate <= resDFT1500
        modAmpHiWeight[mask] = 0
        mask = modRate > modfreqMaxWeight
        modAmpHiWeight[mask] = modAmpHiWeight[mask]*roughHiWeight[mask]

        # Section 7.1.5.3 ECMA-418-2:2025 - Estimation of fundamental modulation rate
        # TODO: replace the loop approach with a parallelised approach!
        # matrix initialisation to ensure zero rates do not cause missing bands in output
        modFundRate = np.zeros([nBlocks, nBands])
        modMaxWeight = np.zeros([10, nBlocks, nBands])

        if waitBar:
            rateIter = tqdm(range(nBands),
                            desc="Modulation weightings")
        else:
            rateIter = range(nBands)

        for zBand in rateIter:
            for lBlock in range(nBlocks):
                # Proceed with rate detection if non-zero modulation rates
                if np.max(modRate[:, lBlock, zBand]) > 0:
                    modRateForLoop = modRate[modRate[:, lBlock, zBand] > 0,
                                             lBlock, zBand]

                    nPeaks = len(modRateForLoop)

                    # initialise empty list for equation 90
                    indSetiPeak = np.empty([nPeaks,], dtype=object)
                    # initialise empty matrix for equation 91
                    harmCompEnergy = np.empty([nPeaks,], dtype=object)

                    for iPeak in range(nPeaks):
                        # Equation 88 [R_i_0(i)]
                        modRateRatio = shmRound(modRateForLoop/modRateForLoop[iPeak])
                        uniqRatios, startGroupInds, countDupes = np.unique(modRateRatio,
                                                                           return_index=True,
                                                                           return_counts=True)

                        # add any non-duplicated ratio indices
                        testIndices = -np.ones([10,]).astype(int)
                        if len(startGroupInds[countDupes == 1]) > 0:
                            testIndices[0:len(startGroupInds[countDupes == 1])] = startGroupInds[countDupes == 1]
                        # end of non-duplicated ratio if branch

                        # loop over duplicated values to select single index
                        if np.max(countDupes) > 1:
                            dupeRatioVals = uniqRatios[countDupes > 1]
                            for jDupe in range(len(dupeRatioVals)):

                                # Equation 89 [i]
                                dupeGroupInds = (modRateRatio == dupeRatioVals[jDupe]).nonzero()[0]
                                denom = modRateRatio[dupeGroupInds]*modRateForLoop[iPeak]
                                testDupe = np.abs(np.divide(modRateForLoop[dupeGroupInds],
                                                            denom,
                                                            out=np.zeros_like(denom),
                                                            where=denom != 0) - 1)

                                # discard if no testDupes
                                if len(testDupe) > 0:
                                    testDupeMin = np.argmin(testDupe)
                                    # append selected index
                                    testIndices[len(startGroupInds[countDupes == 1])
                                                + jDupe] = dupeGroupInds[testDupeMin]
                                # end of if branch for testDupes
                            # end of for loop over duplicated ratios
                        # end of if branch for duplicated ratios

                        # discard negative indices
                        testIndices = testIndices[testIndices >= 0]

                        # Equation 90 [I_i_0]
                        denom = modRateRatio[testIndices]*modRateForLoop[iPeak]
                        harmComplexTest = np.abs(np.divide(modRateForLoop[testIndices],
                                                           denom,
                                                           out=np.zeros_like(denom),
                                                           where=denom != 0) - 1)
                        indSetiPeak[iPeak] = testIndices[harmComplexTest < 0.04]

                        # Equation 91 [E_i_0]
                        harmCompEnergy[iPeak] = np.sum(modAmpHiWeight[indSetiPeak[iPeak],
                                                                      lBlock,
                                                                      zBand])

                    # end of loop over peaks

                    harmCompEnergy = harmCompEnergy.astype(float)
                    iMaxEnergy = np.argmax(harmCompEnergy)
                    indSetMax = indSetiPeak[iMaxEnergy]
                    modFundRate[lBlock, zBand] = modRateForLoop[iMaxEnergy]
                    # Equation 94 [i_peak]
                    iPeakAmp = np.argmax(modAmpHiWeight[indSetMax, lBlock, zBand])
                    iPeak = indSetMax[iPeakAmp]

                    # Equation 93 [w_peak]
                    gravityWeight = 1 + 0.1*np.abs(np.sum(modRateForLoop[indSetMax]
                                                          * modAmpHiWeight[indSetMax,
                                                                           lBlock,
                                                                           zBand],
                                                          axis=0)
                                                   / np.sum(modAmpHiWeight[indSetMax,
                                                                           lBlock,
                                                                           zBand]
                                                            + epsilon, axis=0)
                                                   - modRateForLoop[iPeak])**0.749

                    # Equation 92 [Ahat(i)]
                    modMaxWeight[indSetMax,
                                 lBlock,
                                 zBand] = gravityWeight*modAmpHiWeight[indSetMax,
                                                                       lBlock,
                                                                       zBand]

                # end of if branch for non-zero modulation rates
            # end of for loop over blocks
        # end of for loop over bands

        # Equation 95 [A(l,z)]
        roughLoWeight = shmRoughWeight(modFundRate, modfreqMaxWeight,
                                       roughLoWeightParams)
        modMaxWeightSum = np.sum(modMaxWeight, axis=0)
        modMaxLoWeight = np.sum(roughLoWeight*modMaxWeight, axis=0)
        mask = modFundRate <= resDFT1500
        modMaxLoWeight[mask] = 0
        mask = modFundRate > modfreqMaxWeight
        modMaxLoWeight[mask] = modMaxWeightSum[mask]
        modAmpMax = modMaxLoWeight
        modAmpMax[modAmpMax < 0.074376] = 0

        # Time-dependent specific roughness
        # ---------------------------------
        # Section 7.1.7 ECMA-418-2:2025

        # Section 7.1.7 interpolation to 50 Hz sampling rate    
        t = lBlocksOut/sampleRate48k
        t50 = np.linspace(0, signalT, l_50Last)
        # TODO: check the 2025 version redefinition of t(l) makes sense

        specRoughEst = np.zeros([l_50Last, nBands])
        for zBand in range(nBands):
            interpolator = PchipInterpolator(t, modAmpMax[:, zBand], axis=0)
            specRoughEst[:, zBand] = interpolator(t50)
        # end of for loop for interpolation
        specRoughEst[specRoughEst < 0] = 0  # [R'_est(l_50,z)]

        # Section 7.1.7 Equation 107 [Rtilde'_est(l_50)]
        specRoughEstRMS = np.sqrt(np.mean(specRoughEst**2, axis=1))

        # Section 7.1.7 Equation 108 [Rbar'_est(l_50)]
        specRoughEstAvg = np.mean(specRoughEst, axis=1)

        # Section 7.1.7 Equation 106 [B(l_50)]
        Bl50 = np.zeros(specRoughEstAvg.size)
        mask = specRoughEstAvg != 0
        Bl50[mask] = specRoughEstRMS[mask]/specRoughEstAvg[mask]

        # Section 7.1.7 Equation 105 [E(l_50)]
        El50 = (0.95555 - 0.58449)*(np.tanh(1.6407*(Bl50 - 2.5804)) + 1)*0.5 + 0.58449

        # Section 7.1.7 Equation 104 [Rhat'(l_50,z)]
        specRoughEstTform = cal_R*cal_Rx*(specRoughEst.T**El50).T

        # Section 7.1.7 Equation 109-110 [R'(l_50,z)]
        riseTime = 0.0625
        fallTime = 0.5
        specRoughness[:, :, chan] = shmRoughLowPass(specRoughEstTform, sampleRate50,
                                                    riseTime, fallTime)

    # end of for loop over channels

    # Binaural roughness
    # Section 7.1.11 ECMA-418-2:2025 [R'_B(l_50,z)]
    if chansIn == 2 and binaural:
        # Equation 112
        specRoughness[:, :, 2] = np.sqrt(np.sum(specRoughness[:, :, 0:2]**2,
                                                axis=2)/2)
    # end of if branch for combined binaural

    # Section 7.1.8 ECMA-418-2:2025
    # Time-averaged specific roughness [R'(z)]
    specRoughnessAvg = np.mean(specRoughness[16:, :, :], axis=0)

    # Section 7.1.9 ECMA-418-2:2025
    # Time-dependent roughness Equation 111 [R(l_50)]
    roughnessTDep = np.sum(specRoughness*dz, axis=1)

    # ensure channel dimension is retained (to ease plotting)
    if chansOut == 1:
        specRoughnessAvg = shmDimensional(specRoughnessAvg)
        roughnessTDep = shmDimensional(roughnessTDep)

    # Section 7.1.10 ECMA-418-2:2025
    # Overall roughness [R]
    roughness90Pc = np.percentile(roughnessTDep[16:, :], 90, axis=0)

    # time (s) corresponding with results output [t]
    timeOut = np.arange(0, (specRoughness.shape[0]))/sampleRate50

    # %% Output plotting

    # Plot figures
    # ------------
    if outPlot:
        # Plot results
        for chan in range(chansOut):
            # Plot results
            cmap_inferno = mpl.colormaps['inferno']
            chan_lab = chans[chan]
            fig, axs = plt.subplots(nrows=2, ncols=1, figsize=[10.5, 7.5],
                                    layout='constrained')

            ax1 = axs[0]
            pmesh = ax1.pcolormesh(timeOut, bandCentreFreqs,
                                   np.swapaxes(specRoughness[:, :, chan], 0, 1),
                                   cmap=cmap_inferno,
                                   vmin=0,
                                   vmax=np.ceil(np.max(specRoughness[:, :,
                                                                     chan])*500)/500,
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
                         label=(r"Specific roughness,"
                                "\n"
                                r"$\mathregular{asper_{SHM}}/\mathregular{Bark_{SHM}}$"),
                         aspect=10, cax=cbax)

            ax2 = axs[1]
            ax2.plot(timeOut, roughness90Pc[chan]*np.ones(timeOut.size),
                     color=cmap_inferno(33/255), linewidth=1,
                     label=("90th-" + "\n" + "percentile"))
            ax2.plot(timeOut, roughnessTDep[:, chan],
                     color=cmap_inferno(165/255),
                     linewidth=0.75, label=("Time-" + "\n" + "dependent"))
            ax2.set(xlim=[timeOut[0], timeOut[-1] + timeOut[1] - timeOut[0]],
                    xlabel="Time, s",
                    ylim=[0, 1.1*np.ceil(np.max(roughnessTDep[:, chan])*10)/10],
                    ylabel=(r"Roughness, $\mathregular{asper_{SHM}}$"))
            ax2.grid(alpha=0.075, linestyle='--')
            ax2.legend(bbox_to_anchor=(1, 0.85), title="Overall")

            # Filter signal to determine A-weighted time-averaged level
            if chan == 2:
                pA = A_weight_T(p_re, fs=sampleRate48k)
                LAeq2 = 20*np.log10(shmRMS(pA, axis=0)/2e-5)
                # take the higher channel level as representative (PD ISO/TS
                # 12913-3:2019 Annex D)
                LAeq = np.max(LAeq2)
                lr = np.argmax(LAeq2)
                # identify which channel is higher
                if lr == 0:
                    whichEar = " left ear"
                else:
                    whichEar = " right ear"
                # end of if branch to identify which channel is higher

                chan_lab = chan_lab + whichEar
            else:
                pA = A_weight_T(p_re[:, chan], fs=sampleRate48k)
                LAeq = 20*np.log10(shmRMS(pA)/2e-5)

            fig.suptitle(t=(chan_lab + " signal sound pressure level = " +
                            str(roundTrad(LAeq, 1)) +
                            r"dB $\mathregular{\mathit{L}_{Aeq}}$"))
            fig.show()
        # end of for loop over channels
    # end of if branch for plotting

    # %% Output assignment

    # Discard singleton dimensions
    if chansOut == 1:
        specRoughness = np.squeeze(specRoughness)
        specRoughnessAvg = np.squeeze(specRoughnessAvg)
        roughnessTDep = np.squeeze(roughnessTDep)
    # end of if branch for singleton dimensions

    # Assign outputs to structure
    if chansOut == 3:
        roughnessSHM = dict()
        roughnessSHM.update({'specRoughness': specRoughness[:, :, 0:2]})
        roughnessSHM.update({'specRoughnessAvg': specRoughnessAvg[:, 0:2]})
        roughnessSHM.update({'roughnessTDep': roughnessTDep[:, 0:2]})
        roughnessSHM.update({'roughness90Pc': roughness90Pc[0:2]})
        roughnessSHM.update({'specRoughnessBin': specRoughness[:, :, 2]})
        roughnessSHM.update({'specRoughnessAvgBin': specRoughnessAvg[:, 2]})
        roughnessSHM.update({'roughnessTDepBin': roughnessTDep[:, 2]})
        roughnessSHM.update({'roughness90PcBin': np.array(roughness90Pc[2])})
        roughnessSHM.update({'bandCentreFreqs': bandCentreFreqs})
        roughnessSHM.update({'timeOut': timeOut})
        roughnessSHM.update({'soundField': soundField})
    else:
        roughnessSHM.update({'specRoughness': specRoughness})
        roughnessSHM.update({'specRoughnessAvg': specRoughnessAvg})
        roughnessSHM.update({'roughnessTDep': roughnessTDep})
        roughnessSHM.update({'roughness90Pc': roughness90Pc})
        roughnessSHM.update({'bandCentreFreqs': bandCentreFreqs})
        roughnessSHM.update({'timeOut': timeOut})
        roughnessSHM.update({'soundField': soundField})

    return roughnessSHM

# end of acousticSHRoughness function
