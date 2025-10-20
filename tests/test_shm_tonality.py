# -*- coding: utf-8 -*-
# %% Preamble
"""
test_shm_tonality.py
--------------------

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

"""

# %% Import block
import pytest
import numpy as np
from sottek_hearing_model.shm_tonality_ecma import shm_tonality_ecma
from sottek_hearing_model.shmReferenceSignals import shm_generate_ref_signals


# %% test_shm_tonality
def test_shm_tonality():
    tonality_ref_signal, _, _ = shm_generate_ref_signals(10)

    tonality = shm_tonality_ecma(p=tonality_ref_signal, samp_rate_in=48e3,
                                 axis=0, soundfield='free_frontal',
                                 wait_bar=False, out_plot=False)

    assert tonality['tonality_avg'] == pytest.approx(1.0, abs=1e-4)
    assert tonality['spec_tonality_avg'][17] == pytest.approx(1.0, abs=1e-4)
    assert tonality['spec_tonality_avg_freqs'][17] == pytest.approx(1000, abs=1)
    assert np.all(tonality['tonality_t'][57:87] == pytest.approx(1.0, abs=1e-2))
    assert np.all(tonality['tonality_t'][87:] == pytest.approx(1.0, abs=1e-4))
    assert np.all(tonality['spec_tonality'][57:87, 17] == pytest.approx(1.0, abs=1e-2))
    assert np.all(tonality['spec_tonality'][87:, 17] == pytest.approx(1.0, abs=1e-4))
    assert np.all(tonality['tonality_t_freqs'][57:] == pytest.approx(1000, abs=1))
    assert np.all(tonality['spec_tonality_freqs'][57:, 17] == pytest.approx(1000, abs=1))

