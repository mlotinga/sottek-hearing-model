# -*- coding: utf-8 -*-
# %% Preamble
"""
test_shmLoudness.py
----------------------

Tests the validity of ECMA-418-2 loudness implementation (Sottek Hearing Model) using the reference signal.

Requirements
------------
pytest
numpy

Ownership and Quality Assurance
-------------------------------
Author: Mike JB Lotinga (m.j.lotinga@edu.salford.ac.uk)
Institution: University of Salford

Date created: 02/10/2025
Date last modified: 09/10/2025
Python version: 3.11

Copyright statement: This code has been devloped during work undertaken within
the RefMap project (www.refmap.eu), based on the RefMap code repository
(https://github.com/acoustics-code-salford/refmap-psychoacoustics),
and as such is subject to copyleft licensing as detailed in the code repository
(https://github.com/acoustics-code-salford/sottek-hearing-model).

As per the licensing information, please be aware that this code is WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.

"""

# %% Import block
import pytest
import numpy as np
from sottek_hearing_model.shmLoudnessECMA import (shmLoudnessECMA, shmLoudnessECMAFromComp)
from sottek_hearing_model.shmReferenceSignals import shmGenerateRefSignals


# %% test_shmLoudness
def test_shmLoudness():
    loudnessRefSignal, _, _ = shmGenerateRefSignals(10)

    loudnessSHM = shmLoudnessECMA(p=loudnessRefSignal, sampleRateIn=48e3,
                                  axisN=0, soundField='freeFrontal',
                                  waitBar=False, outPlot=False, binaural=False)

    assert loudnessSHM['loudnessPowAvg'] == pytest.approx(1.0, abs=1e-4)
    assert loudnessSHM['specLoudnessPowAvg'][17] == pytest.approx(0.3477, abs=1e-4)
    assert np.all(loudnessSHM['loudnessTDep'][57:87] == pytest.approx(1.0, abs=1e-2))
    assert np.all(loudnessSHM['loudnessTDep'][87:] == pytest.approx(1.0, abs=1e-4))
    assert np.all(loudnessSHM['specLoudness'][57:87, 17] == pytest.approx(0.3477, abs=1e-2))
    assert np.all(loudnessSHM['specLoudness'][87:, 17] == pytest.approx(0.3477, abs=1e-4))


# %% test_shmLoudnessFromComp
def test_shmLoudnessFromComp():
    loudnessRefSignal, _, _ = shmGenerateRefSignals(10)

    loudnessSHMFull = shmLoudnessECMA(p=loudnessRefSignal, sampleRateIn=48e3,
                                      axisN=0, soundField='freeFrontal',
                                      waitBar=False, outPlot=False, binaural=False)
    
    loudnessSHMFromComp = shmLoudnessECMAFromComp(loudnessSHMFull['specTonalLoudness'],
                                                  loudnessSHMFull['specNoiseLoudness'],
                                                  outPlot=False, binaural=False)

    assert loudnessSHMFromComp['loudnessPowAvg'] == pytest.approx(1.0, abs=1e-4)
    assert loudnessSHMFromComp['specLoudnessPowAvg'][17] == pytest.approx(0.3477, abs=1e-4)
    assert np.all(loudnessSHMFromComp['loudnessTDep'][57:87] == pytest.approx(1.0, abs=1e-2))
    assert np.all(loudnessSHMFromComp['loudnessTDep'][87:] == pytest.approx(1.0, abs=1e-4))
    assert np.all(loudnessSHMFromComp['specLoudness'][57:87, 17] == pytest.approx(0.3477, abs=1e-2))
    assert np.all(loudnessSHMFromComp['specLoudness'][87:, 17] == pytest.approx(0.3477, abs=1e-4))
