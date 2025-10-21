# -*- coding: utf-8 -*-
# %% Preamble
"""
test_shm_loudness.py
--------------------

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
import pytest  # pyright: ignore[reportMissingImports]
import numpy as np
from sottek_hearing_model.shm_loudness_ecma import (shm_loudness_ecma, shm_loudness_ecma_from_comp)
from sottek_hearing_model.shm_reference_signals import shm_generate_ref_signals


# %% test_shm_loudness
def test_shm_loudness():
    loudness_ref_signal, _, _ = shm_generate_ref_signals(5)

    loudness_ref_signal = np.vstack((loudness_ref_signal, loudness_ref_signal))

    loudness = shm_loudness_ecma(p=loudness_ref_signal, samp_rate_in=48e3,
                                 axis=1, soundfield='free_frontal',
                                 wait_bar=False, out_plot=False, binaural=True)

    assert loudness['loudness_powavg'][0] == pytest.approx(1.0, abs=1e-4)
    assert loudness['spec_loudness_powavg'][17, 0] == pytest.approx(0.3477, abs=1e-4)
    assert np.all(loudness['loudness_t'][57:87, 0] == pytest.approx(1.0, abs=1e-2))
    assert np.all(loudness['loudness_t'][87:, 0] == pytest.approx(1.0, abs=1e-3))
    assert np.all(loudness['spec_loudness'][57:87, 17, 0] == pytest.approx(0.3477, abs=1e-2))
    assert np.all(loudness['spec_loudness'][87:, 17, 0] == pytest.approx(0.3477, abs=1e-4))
    assert loudness['loudness_powavg'][0] == pytest.approx(1.0, abs=1e-4)
    assert loudness['spec_loudness_powavg_bin'][17] == pytest.approx(0.3477, abs=1e-4)
    assert np.all(loudness['loudness_t_bin'][57:87] == pytest.approx(1.0, abs=1e-2))
    assert np.all(loudness['loudness_t_bin'][87:] == pytest.approx(1.0, abs=1e-3))
    assert np.all(loudness['spec_loudness_bin'][57:87, 17] == pytest.approx(0.3477, abs=1e-2))
    assert np.all(loudness['spec_loudness_bin'][87:, 17] == pytest.approx(0.3477, abs=1e-4))


# %% test_shm_loudness_from_comp
def test_shm_loudness_from_comp():
    loudness_ref_signal, _, _ = shm_generate_ref_signals(5)

    loudness = shm_loudness_ecma(p=loudness_ref_signal, samp_rate_in=48e3,
                                 axis=0, soundfield='free_frontal',
                                 wait_bar=False, out_plot=False, binaural=False)
    
    loudness_ref_signal = np.vstack((loudness_ref_signal, loudness_ref_signal))

    loudness_from_comp = shm_loudness_ecma_from_comp(loudness['spec_tonal_loudness'],
                                                     loudness['spec_noise_loudness'],
                                                     out_plot=False, binaural=True)

    assert loudness_from_comp['loudness_powavg'] == pytest.approx(1.0, abs=1e-4)
    assert loudness_from_comp['spec_loudness_powavg'][17] == pytest.approx(0.3477, abs=1e-4)
    assert np.all(loudness_from_comp['loudness_t'][57:87] == pytest.approx(1.0, abs=1e-2))
    assert np.all(loudness_from_comp['loudness_t'][87:] == pytest.approx(1.0, abs=1e-3))
    assert np.all(loudness_from_comp['spec_loudness'][57:87, 17] == pytest.approx(0.3477, abs=1e-2))
    assert np.all(loudness_from_comp['spec_loudness'][87:, 17] == pytest.approx(0.3477, abs=1e-4))
    assert loudness_from_comp['loudness_powavg'][0] == pytest.approx(1.0, abs=1e-4)
    assert loudness_from_comp['spec_loudness_powavg_bin'][17] == pytest.approx(0.3477, abs=1e-4)
    assert np.all(loudness_from_comp['loudness_t_bin'][57:87] == pytest.approx(1.0, abs=1e-2))
    assert np.all(loudness_from_comp['loudness_t_bin'][87:] == pytest.approx(1.0, abs=1e-3))
    assert np.all(loudness_from_comp['spec_loudness_bin'][57:87, 17] == pytest.approx(0.3477, abs=1e-2))
    assert np.all(loudness_from_comp['spec_loudness_bin'][87:, 17] == pytest.approx(0.3477, abs=1e-4))
