# -*- coding: utf-8 -*-
# %% Preamble
"""
test_shmRoughness.py
--------------------

Tests the validity of ECMA-418-2 roughness implementation (Sottek Hearing Model) using the reference signal.

Requirements
------------
pytest
numpy

Ownership and Quality Assurance
-------------------------------
Author: Mike JB Lotinga (m.j.lotinga@edu.salford.ac.uk)
Institution: University of Salford

Date created: 09/10/2025
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
from sottek_hearing_model.shmRoughnessECMA import shmRoughnessECMA
from sottek_hearing_model.shmReferenceSignals import shmGenerateRefSignals

# %% test_shmRoughness
def test_shmRoughness():
    _, roughnessRefSignal, _ = shmGenerateRefSignals(10)

    roughnessSHM = shmRoughnessECMA(p=roughnessRefSignal, sampleRateIn=48e3,
                                     axisN=0, soundField='freeFrontal',
                                     waitBar=False, outPlot=False, binaural=False)

    assert roughnessSHM['roughness90Pc'] == pytest.approx(1.0, abs=1e-4)
    assert roughnessSHM['specRoughnessAvg'][17] == pytest.approx(0.374, abs=1e-3)
    assert np.all(roughnessSHM['roughnessTDep'][16:25] == pytest.approx(1.0, abs=1e-2))
    assert np.all(roughnessSHM['roughnessTDep'][25:] == pytest.approx(1.0, abs=1e-3))
    assert np.all(roughnessSHM['specRoughness'][16:25, 17] == pytest.approx(0.374, abs=1e-2))
    assert np.all(roughnessSHM['specRoughness'][25:, 17] == pytest.approx(0.374, abs=1e-3))
