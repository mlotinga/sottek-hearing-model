# -*- coding: utf-8 -*-
# %% Preamble
"""
test_shm_fluctuation.py
-----------------------

Tests the validity of ECMA-418-2 fluctuation strength implementation (Sottek
Hearing Model) using the reference signal, and the correctness of the
High-resolution Spectral Analysis (HSA) subfunctions using synthetic signals
with known ground truth.

The expected values for the reference signal (1 kHz sinusoid, 100 % amplitude
modulated at 4 Hz, 60 dB sound pressure level) are taken from the reference
implementation (HEAD acoustics ArtemiS v17): overall fluctuation strength
1.00 vacil_HMS, and time-aggregated specific fluctuation strength 0.395
vacil_HMS/Bark_HMS in the 1 kHz critical band (band index 17). The
tolerances reflect the agreement achieved by this implementation (see the
validation report).

Requirements
------------
pytest
numpy

Ownership and Quality Assurance
-------------------------------
Author: Mike JB Lotinga (m.j.lotinga@edu.salford.ac.uk)
Institution: University of Salford

Date created: 18/09/2026
Date last modified: 18/09/2026
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
from contextlib import nullcontext as does_not_raise  # pyright: ignore[reportMissingImports]
import pytest  # pyright: ignore[reportMissingImports]
import numpy as np
from sottek_hearing_model.shm_fluctuation_ecma import shm_fluctuation_ecma
from sottek_hearing_model.shm_reference_signals import shm_generate_ref_signals
from sottek_hearing_model.shm_subs import (shm_hsa, shm_hsa_window_response,
                                           shm_fluct_weight, shm_loud_nonlin,
                                           shm_mov_median)


# %% test_shm_fluctuation_ref_48k
def test_shm_fluctuation_ref_48k():
    _, _, fluctuation_ref_signal = shm_generate_ref_signals(10)

    fluctuation_ref_signal = np.vstack((fluctuation_ref_signal, fluctuation_ref_signal))

    fluctuation = shm_fluctuation_ecma(p=fluctuation_ref_signal, samp_rate_in=48e3,
                                       axis=1, soundfield='free_frontal',
                                       wait_bar=False, out_plot=False,
                                       binaural=True, parallel_cores=None)

    assert fluctuation['fluctuation90pc'][0] == pytest.approx(1.0, abs=1e-2)
    assert fluctuation['spec_fluctuation_avg'][17, 0] == pytest.approx(0.395, abs=1e-2)
    assert np.all(fluctuation['fluctuation_t'][100:200, 0] == pytest.approx(1.0, abs=1e-1))
    assert np.all(fluctuation['fluctuation_t'][200:, 0] == pytest.approx(1.0, abs=2e-2))
    assert np.all(fluctuation['spec_fluctuation'][100:200, 17, 0] == pytest.approx(0.395, abs=5e-2))
    assert np.all(fluctuation['spec_fluctuation'][200:, 17, 0] == pytest.approx(0.395, abs=1e-2))
    assert fluctuation['fluctuation90pc_bin'] == pytest.approx(1.0, abs=1e-2)
    assert fluctuation['spec_fluctuation_avg_bin'][17] == pytest.approx(0.395, abs=1e-2)
    assert np.all(fluctuation['fluctuation_t_bin'][100:200] == pytest.approx(1.0, abs=1e-1))
    assert np.all(fluctuation['fluctuation_t_bin'][200:] == pytest.approx(1.0, abs=2e-2))
    assert np.all(fluctuation['spec_fluctuation_bin'][100:200, 17] == pytest.approx(0.395, abs=5e-2))
    assert np.all(fluctuation['spec_fluctuation_bin'][200:, 17] == pytest.approx(0.395, abs=1e-2))
    # the two identical channels must give identical results
    assert np.all(fluctuation['spec_fluctuation'][:, :, 0] == fluctuation['spec_fluctuation'][:, :, 1])
    # only the bands around 1 kHz carry fluctuation strength
    assert np.all(fluctuation['spec_fluctuation_avg'][:12, 0] == 0)
    assert np.all(fluctuation['spec_fluctuation_avg'][25:, 0] == 0)


# %% test_shm_fluctuation_ref_44k
def test_shm_fluctuation_ref_44k():
    _, _, fluctuation_ref_signal = shm_generate_ref_signals(10, samp_rate=44.1e3)

    fluctuation_ref_signal = np.vstack((fluctuation_ref_signal, fluctuation_ref_signal))

    fluctuation = shm_fluctuation_ecma(p=fluctuation_ref_signal, samp_rate_in=44.1e3,
                                       axis=1, soundfield='free_frontal',
                                       wait_bar=False, out_plot=False,
                                       binaural=True, parallel_cores=None)

    assert fluctuation['fluctuation90pc'][0] == pytest.approx(1.0, abs=1e-2)
    assert fluctuation['spec_fluctuation_avg'][17, 0] == pytest.approx(0.395, abs=1e-2)
    assert np.all(fluctuation['fluctuation_t'][100:200, 0] == pytest.approx(1.0, abs=1e-1))
    assert np.all(fluctuation['fluctuation_t'][200:, 0] == pytest.approx(1.0, abs=2e-2))
    assert np.all(fluctuation['spec_fluctuation'][100:200, 17, 0] == pytest.approx(0.395, abs=5e-2))
    assert np.all(fluctuation['spec_fluctuation'][200:, 17, 0] == pytest.approx(0.395, abs=1e-2))
    assert fluctuation['fluctuation90pc_bin'] == pytest.approx(1.0, abs=1e-2)
    assert fluctuation['spec_fluctuation_avg_bin'][17] == pytest.approx(0.395, abs=1e-2)
    assert np.all(fluctuation['fluctuation_t_bin'][100:200] == pytest.approx(1.0, abs=1e-1))
    assert np.all(fluctuation['fluctuation_t_bin'][200:] == pytest.approx(1.0, abs=2e-2))
    assert np.all(fluctuation['spec_fluctuation_bin'][100:200, 17] == pytest.approx(0.395, abs=5e-2))
    assert np.all(fluctuation['spec_fluctuation_bin'][200:, 17] == pytest.approx(0.395, abs=1e-2))


# %% test parallel_cores argument runs without an error (the output value is not important)
@pytest.mark.parametrize("parallel_cores, expectation", [
    (1, does_not_raise()),
    (2, does_not_raise()),
    (None, does_not_raise()),
])
def test_shm_fluctuation_ref_parallel_cores(parallel_cores, expectation):
    _, _, fluctuation_ref_signal = shm_generate_ref_signals(2)

    with expectation:
        assert shm_fluctuation_ecma(p=fluctuation_ref_signal, samp_rate_in=48e3,
                                    axis=1, soundfield='free_frontal',
                                    wait_bar=False, out_plot=False,
                                    binaural=False,
                                    parallel_cores=parallel_cores) is not None


# %% test that the mono and serial/parallel outputs are identical
def test_shm_fluctuation_mono_parallel_consistency():
    _, _, fluctuation_ref_signal = shm_generate_ref_signals(2)

    fluctuation_serial = shm_fluctuation_ecma(p=fluctuation_ref_signal,
                                              samp_rate_in=48e3, axis=0,
                                              wait_bar=False, out_plot=False,
                                              parallel_cores=1)
    fluctuation_parallel = shm_fluctuation_ecma(p=fluctuation_ref_signal,
                                                samp_rate_in=48e3, axis=0,
                                                wait_bar=False, out_plot=False,
                                                parallel_cores=2)

    assert fluctuation_serial['spec_fluctuation'].shape == (101, 53)
    assert np.all(fluctuation_serial['spec_fluctuation']
                  == fluctuation_parallel['spec_fluctuation'])
    assert np.all(fluctuation_serial['fluctuation_t']
                  == fluctuation_parallel['fluctuation_t'])


# %% test that a too-short signal raises an error
def test_shm_fluctuation_short_signal():
    _, _, fluctuation_ref_signal = shm_generate_ref_signals(1.3)

    with pytest.raises(ValueError):
        shm_fluctuation_ecma(p=fluctuation_ref_signal, samp_rate_in=48e3,
                             axis=0, wait_bar=False, out_plot=False)


# %% HSA subfunction tests (synthetic signals with known ground truth)

# common analysis parameters (matching shm_fluctuation_ecma)
BLOCK_SIZE1500 = 2048  # s~b
SAMP_RATE1500 = 1500  # r~s
N_ZB = 64  # default n_zb
N_ZE = 64  # default n_ze


def _windowed_spectrum(envelope, n_zb=N_ZB, n_ze=N_ZE):
    env_window = np.zeros(BLOCK_SIZE1500)
    env_window[n_zb:BLOCK_SIZE1500 - n_ze] = 1
    return np.fft.fft(envelope*env_window, BLOCK_SIZE1500)


# %% test_shm_hsa_window_response
@pytest.mark.parametrize("mod_rate, n_zb, n_ze", [
    (0.0, 64, 64),
    (4.1, 64, 64),
    (-4.1, 64, 64),
    (11.3, 200, 500),
    (0.25, 64, 1000),
])
def test_shm_hsa_window_response(mod_rate, n_zb, n_ze):
    # the window response (Equation 127) must equal the DFT of the
    # rectangular analysis window modulated by exp(j*2*pi*f_c*n/r_s)
    k_indices = np.arange(49)
    n = np.arange(BLOCK_SIZE1500)
    env_window = np.zeros(BLOCK_SIZE1500)
    env_window[n_zb:BLOCK_SIZE1500 - n_ze] = 1
    dft_ref = np.fft.fft(env_window*np.exp(1j*2*np.pi*mod_rate*n/SAMP_RATE1500),
                         BLOCK_SIZE1500)[k_indices]

    window_response = shm_hsa_window_response(k_indices, mod_rate,
                                              BLOCK_SIZE1500, SAMP_RATE1500,
                                              n_zb, n_ze)

    # (relative tolerance: the epsilon term in Equation 127 shifts the phase
    # by ~1e-8 relative to the exact DFT)
    assert (np.max(np.abs(window_response - dft_ref))
            / np.max(np.abs(dft_ref))) == pytest.approx(0.0, abs=1e-6)


# %% test_shm_hsa_single_line
def test_shm_hsa_single_line():
    # single-line (Mc = 1) recovery accuracy for an off-bin modulation rate:
    # a clean two-term synthetic envelope p_E(t) = A0 + A1*cos(2*pi*f1*t + phi)
    # is windowed exactly as the pipeline windows it, and the HSA must
    # recover A0 and the two-sided complex line amplitude (A1/2)*exp(1j*phi)
    t = np.arange(BLOCK_SIZE1500)/SAMP_RATE1500
    a0 = 0.05  # true DC (constant) amplitude [Pa]
    a1 = 0.02  # true AC (cosine) amplitude [Pa]
    f1 = 4.1  # true modulation rate [Hz] (off-bin on purpose)
    phi1 = 0.7  # true phase [rad]
    p_true = (a1/2)*np.exp(1j*phi1)

    spectrum_e = _windowed_spectrum(a0 + a1*np.cos(2*np.pi*f1*t + phi1))

    p_hat, err_lz = shm_hsa(f1, spectrum_e, BLOCK_SIZE1500, SAMP_RATE1500,
                            N_ZB, N_ZE)

    assert p_hat.shape == (2,)
    assert np.abs(p_hat[0] - a0)/a0 == pytest.approx(0.0, abs=1e-6)
    assert np.abs(p_hat[1] - p_true)/np.abs(p_true) == pytest.approx(0.0, abs=1e-6)
    assert err_lz == pytest.approx(0.0, abs=1e-6*np.sum(np.abs(spectrum_e[:17])**2))
    # the error function is an even function of the modulation rate
    _, err_neg = shm_hsa(-f1, spectrum_e, BLOCK_SIZE1500, SAMP_RATE1500,
                         N_ZB, N_ZE)
    assert err_neg == pytest.approx(err_lz, abs=1e-9)
    # the error function increases away from the true modulation rate
    _, err_off = shm_hsa(f1 + 0.3, spectrum_e, BLOCK_SIZE1500, SAMP_RATE1500,
                         N_ZB, N_ZE)
    assert err_off > 1e3*max(err_lz, 1e-12)


# %% test_shm_hsa_multi_line
def test_shm_hsa_multi_line():
    # multi-line (Mc = 3) recovery of three genuine lines with an asymmetric
    # analysis window
    t = np.arange(BLOCK_SIZE1500)/SAMP_RATE1500
    true_freqs = np.array([2.0, 4.9, 11.3])  # Hz
    true_amps = np.array([0.030, 0.015, 0.008])  # Pa
    true_phi = np.array([0.2, -1.1, 2.4])  # rad
    n_zb = 150
    n_ze = 400

    envelope = 0.04*np.ones(t.size)
    for i_line in range(true_freqs.size):
        envelope += true_amps[i_line]*np.cos(2*np.pi*true_freqs[i_line]*t
                                             + true_phi[i_line])
    spectrum_e = _windowed_spectrum(envelope, n_zb, n_ze)

    p_hat, _ = shm_hsa(true_freqs, spectrum_e, BLOCK_SIZE1500, SAMP_RATE1500,
                       n_zb, n_ze)

    assert p_hat.shape == (4,)
    assert np.real(p_hat[0]) == pytest.approx(0.04, rel=1e-6)
    assert np.abs(p_hat[1:]) == pytest.approx(true_amps/2, rel=1e-6)
    assert np.angle(p_hat[1:]) == pytest.approx(true_phi, abs=1e-6)


# %% test_shm_fluct_weight
def test_shm_fluct_weight():
    # unity weighting at f_max = 4.8659 Hz, zero weighting at zero
    # modulation rate, and band-pass shape either side of f_max
    assert shm_fluct_weight(4.8659, 1000.0)[0] == pytest.approx(1.0)
    assert shm_fluct_weight(0.0, 1000.0)[0] == 0.0
    weights = shm_fluct_weight(np.array([0.5, 2.0, 4.8659, 10.0, 30.0]), 1000.0)
    assert np.all(np.diff(weights[:3]) > 0)
    assert np.all(np.diff(weights[2:]) < 0)
    # the carrier-frequency correction only affects the high-rate branch,
    # and is unity at 1 kHz
    assert shm_fluct_weight(2.0, 250.0)[0] == pytest.approx(shm_fluct_weight(2.0, 1000.0)[0])
    assert shm_fluct_weight(10.0, 250.0)[0] < shm_fluct_weight(10.0, 1000.0)[0]


# %% test_shm_loud_nonlin
def test_shm_loud_nonlin():
    # zero pressure gives zero loudness, and the nonlinearity is monotonic
    assert shm_loud_nonlin(0.0) == 0.0
    p_rms = 2e-5*10**(np.arange(0, 100, 10)/20)
    loudness = shm_loud_nonlin(p_rms)
    assert loudness.shape == p_rms.shape
    assert np.all(np.diff(loudness) > 0)
    # a 1 kHz sinusoid at 40 dB SPL (p_rms = 2 mPa) gives a band loudness of
    # approximately 0.396 sone_HMS/Bark_HMS before the threshold in quiet
    assert shm_loud_nonlin(2e-5*10**(40/20)) == pytest.approx(0.396, abs=1e-3)


# %% test_shm_mov_median
@pytest.mark.parametrize("win_len, n", [
    (33, 2048),
    (71, 501),
    (3, 20),
    (33, 10),
    (1, 5),
])
def test_shm_mov_median(win_len, n):
    # centred moving median with shrinking endpoint windows (MATLAB
    # movmedian equivalent) against a direct evaluation
    rng = np.random.default_rng(0)
    vals = rng.random((n, 3))
    half = (win_len - 1)//2
    med_ref = np.array([np.median(vals[max(0, ii - half):ii + half + 1], axis=0)
                        for ii in range(n)])

    assert np.all(shm_mov_median(vals, win_len, axis=0) == med_ref)
    assert np.all(shm_mov_median(vals.T, win_len, axis=1) == med_ref.T)

    with pytest.raises(ValueError):
        shm_mov_median(vals, 4, axis=0)
