# -*- coding: utf-8 -*-
# %% Preamble
"""
test_shm_roughness.py
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
from sottek_hearing_model.shm_roughness_ecma import shm_roughness_ecma
from sottek_hearing_model.shm_reference_signals import shm_generate_ref_signals

# %% test_shm_roughness
def test_shm_roughness():
    _, roughness_ref_signal, _ = shm_generate_ref_signals(10)

    roughness = shm_roughness_ecma(p=roughness_ref_signal, samp_rate_in=48e3,
                                   axis=0, soundfield='free_frontal',
                                   wait_bar=False, out_plot=False, binaural=False)

    assert roughness['roughness90Pc'] == pytest.approx(1.0, abs=1e-4)
    assert roughness['spec_roughness_avg'][17] == pytest.approx(0.374, abs=1e-3)
    assert np.all(roughness['roughness_t'][16:35] == pytest.approx(1.0, abs=1e-1))
    assert np.all(roughness['roughness_t'][35:] == pytest.approx(1.0, abs=1e-2))
    assert np.all(roughness['spec_roughness'][16:35, 17] == pytest.approx(0.374, abs=1e-1))
    assert np.all(roughness['spec_roughness'][35:, 17] == pytest.approx(0.374, abs=1e-3))
