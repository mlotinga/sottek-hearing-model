# -*- coding: utf-8 -*-
# %% Preamble
"""
test_shmTonality.py
----------------------

Tests the validity of ECMA-418-2 tonality implementation (Sottek Hearing Model) using the reference signal.

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
from sottek_hearing_model.shmTonalityECMA import shmTonalityECMA
from sottek_hearing_model.shmReferenceSignals import shmGenerateRefSignals


# %% test_shmTonality
def test_shmTonality():
    tonalityRefSignal, _, _ = shmGenerateRefSignals(10)
    
    tonalitySHM = shmTonalityECMA(p=tonalityRefSignal, sampleRateIn=48e3,
                                  axisN=0, soundField='freeFrontal',
                                  waitBar=False, outPlot=False)

    assert tonalitySHM['tonalityAvg'] == pytest.approx(1.0, abs=1e-4)
    assert tonalitySHM['specTonalityAvg'][17] == pytest.approx(1.0, abs=1e-4)
    assert tonalitySHM['specTonalityAvgFreqs'][17] == pytest.approx(1000, abs=1)
    assert np.all(tonalitySHM['tonalityTDep'][57:87] == pytest.approx(1.0, abs=1e-2))
    assert np.all(tonalitySHM['tonalityTDep'][87:] == pytest.approx(1.0, abs=1e-4))
    assert np.all(tonalitySHM['specTonality'][57:87, 17] == pytest.approx(1.0, abs=1e-2))
    assert np.all(tonalitySHM['specTonality'][87:, 17] == pytest.approx(1.0, abs=1e-4))
    assert np.all(tonalitySHM['tonalityTDepFreqs'][57:] == pytest.approx(1000, abs=1))
    assert np.all(tonalitySHM['specTonalityFreqs'][57:, 17] == pytest.approx(1000, abs=1))

