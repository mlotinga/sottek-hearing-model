# -*- coding: utf-8 -*-
# %% Preamble
"""
fftw.py
----------------------

Provides FFTW wrapper functions for fast Fourier transforms

Requirements
------------
pyfftw
numpy

Ownership and Quality Assurance
-------------------------------
Author: Mike JB Lotinga (m.j.lotinga@edu.salford.ac.uk)
Institution: University of Salford

Date created: 06/10/2025
Date last modified: 06/10/2025
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
import os
import multiprocessing
import numpy as np
import pyfftw

# Detect logical CPU cores
n_cores = multiprocessing.cpu_count()

fftw_threads = min(max(n_cores - 2, 1), int(os.getenv("FFTW_THREADS", 8)))
print(f"[FFTW] Using {fftw_threads} threads on {n_cores} detected cores")

fft_plan_cache = {}

def get_fftw_plan(shape_in, dtype_in, transform_type, n_fft=None,
                  axis=0, threads=fftw_threads, order='F'):
    """
    Get or create FFTW plans for input shape and transform type.

    Inputs
    ------
    shape_in : tuple
               Shape of the input array (rows, cols)
    dtype_in : data-type
               Data type of the input array (np.float64 or np.complex128)
    transform_type : one of ['r2c', 'c2r', 'c2c_fwd', 'c2c_inv']
                     Type of FFT transform (real-to-complex, complex-to-real,
                     complex-to-complex forward, complex-to-complex inverse)
    n_fft : int, optional
            Length of the FFT (default: None, uses full length of axis)
    axis : int, optional
           Axis along which to perform the FFT (default: 0)
    threads : int
              Number of threads to use for FFTW (default: fftw_threads).
    order : {'C', 'F'}, optional
            Memory layout of the arrays (default: 'F' for Fortran order)
    """

    # Adjust shape for the requested FFT length
    n_fft = int(n_fft or shape_in[axis])
    shape_in_mod = list(shape_in)
    shape_in_mod[axis] = n_fft
    shape_in_mod = tuple(shape_in_mod)

    key = (shape_in_mod, dtype_in, transform_type, axis, threads, order)
    if key in fft_plan_cache:
        return fft_plan_cache[key]

    if transform_type == 'r2c':
        shape_out = list(shape_in_mod)
        shape_out[axis] = n_fft // 2 + 1
        a = pyfftw.empty_aligned(shape_in_mod, dtype='float64', order=order)
        b = pyfftw.empty_aligned(tuple(shape_out), dtype='complex128', order=order)
        plan = pyfftw.FFTW(a, b, axes=(axis,), direction='FFTW_FORWARD',
                           threads=threads, flags=('FFTW_ESTIMATE',))

    elif transform_type == 'c2r':
        shape_in_cplx = list(shape_in_mod)
        shape_in_cplx[axis] = n_fft // 2 + 1
        b = pyfftw.empty_aligned(tuple(shape_in_cplx), dtype='complex128', order=order)
        a = pyfftw.empty_aligned(shape_in_mod, dtype='float64', order=order)
        plan = pyfftw.FFTW(b, a, axes=(axis,), direction='FFTW_BACKWARD',
                           threads=threads, flags=('FFTW_ESTIMATE',))

    elif transform_type == 'c2c_fwd':
        a = pyfftw.empty_aligned(shape_in_mod, dtype='complex128', order=order)
        b = pyfftw.empty_aligned(shape_in_mod, dtype='complex128', order=order)
        plan = pyfftw.FFTW(a, b, axes=(axis,), direction='FFTW_FORWARD',
                           threads=threads, flags=('FFTW_ESTIMATE',))

    elif transform_type == 'c2c_inv':
        b = pyfftw.empty_aligned(shape_in_mod, dtype='complex128', order=order)
        a = pyfftw.empty_aligned(shape_in_mod, dtype='complex128', order=order)
        plan = pyfftw.FFTW(b, a, axes=(axis,), direction='FFTW_BACKWARD',
                           threads=threads, flags=('FFTW_ESTIMATE',))

    else:
        raise ValueError(f"Unknown transform_type: {transform_type}")

    fft_plan_cache[key] = (a, b, plan)
    return fft_plan_cache[key]


def fft_forward_r2c(array_in, n_fft=None, axis=0,
                    threads=fftw_threads, order='F', copy=True):
    """
    Perform forward real-to-complex FFT using FFTW.
    Inputs
    ------
    array_in : ndarray
               Input real-valued array (1D or 2D)
    n_fft : int, optional
            Length of the FFT (default: None, uses full length of axis)
    axis : int, optional
           Axis along which to perform the FFT (default: 0)
    threads : int
              Number of threads to use for FFTW (default: fftw_threads)
    order : {'C', 'F'}, optional
            Memory layout of the input array (default: 'F' for Fortran order)
    copy : bool, optional
           Whether to return a copy of the output array (default: True)
    
    Returns
    -------
    array_out : ndarray
                Output complex-valued FFT array
    
    """
    shape_in = array_in.shape
    n_in = shape_in[axis]
    n_fft = int(n_fft or n_in)

    if n_fft != n_in:
        pad_width = [(0, 0)] * array_in.ndim
        pad_width[axis] = (0, max(0, n_fft - n_in))
        array_in = np.pad(array_in, pad_width, mode='constant')

    a, b, plan = get_fftw_plan(shape_in, dtype_in=np.float64, transform_type='r2c',
                               n_fft=n_fft, axis=axis, threads=threads, order=order)
    src = ensure_dtype_and_order(array_in, np.float64, order)
    np.copyto(a, src, casting='no')
    plan()
    return b.copy() if copy else b


def fft_inverse_c2r(array_in, n_fft=None, axis=0, original_shape=None,
                    threads=fftw_threads, order='F', copy=True):
    """
    Perform inverse complex-to-real FFT using FFTW.
    Inputs
    ------
    array_in : ndarray
               Input complex-valued array (1D or 2D)
    n_fft : int, optional
            Length of the FFT (default: None, uses full length of axis)
    axis : int, optional
           Axis along which to perform the inverse FFT (default: 0)
    original_shape : tuple, optional
                     Original shape of the real-valued array before FFT.
                     If None, it is inferred from array_in shape.
    threads : int
              Number of threads to use for FFTW (default: fftw_threads)
    order : {'C', 'F'}, optional
            Memory layout of the input array (default: 'F' for Fortran order)
    copy : bool, optional
           Whether to return a copy of the output array (default: True)
    
    Returns
    -------
    array_out : ndarray
                Output real-valued inverse FFT array
    
    """
    shape_in = array_in.shape
    if original_shape is None:
        # infer from complex spectrum length along axis
        complex_len = shape_in[axis]
        inferred_nfft = 2 * (complex_len - 1)
        n_fft = int(n_fft or inferred_nfft)
        shape_real = list(shape_in)
        shape_real[axis] = n_fft
        shape_real = tuple(shape_real)
    else:
        shape_real = tuple(original_shape)
        n_fft = int(n_fft or shape_real[axis])

    n_fft = int(n_fft or (array_shape[axis]))
    a, b, plan = get_fftw_plan(shape_real, dtype_in=np.float64, transform_type='c2r',
                               n_fft=n_fft, axis=axis, threads=threads, order=order)
    if b.shape != shape_in:
        arr_for_copy = np.empty(b.shape, dtype=np.complex128, order=order)
        arr_for_copy[:] = array_in  # allow upcast if needed
        src = arr_for_copy
    else:
        src = ensure_dtype_and_order(array_in, np.complex128, order)
    
    np.copyto(b, src, casting='no')
    plan()
    norm = n_fft
    a /= norm

    if original_shape is not None:
        # Trim back to the original shape if zero-padded
        if a.shape[axis] > original_shape[axis]:
            slicer = [slice(None)] * a.ndim
            slicer[axis] = slice(0, original_shape[axis])
            a = a[tuple(slicer)]

    return a.copy() if copy else a


def fft_forward_c2c(array_in, n_fft=None, axis=0,
                    threads=fftw_threads, order='F', copy=True):
    """
    Perform forward complex-to-complex FFT using FFTW.
    Inputs
    ------
    array_in : ndarray
               Input complex-valued array (1D or 2D)
    n_fft : int, optional
            Length of the FFT (default: None, uses full length of axis)
    axis : int, optional
           Axis along which to perform the FFT (default: 0)
    threads : int
              Number of threads to use for FFTW (default: fftw_threads).
    order : {'C', 'F'}, optional
            Memory layout of the input array (default: 'F' for Fortran order)
    copy : bool, optional
           Whether to return a copy of the output array (default: True)
    
    Returns
    -------
    array_out : ndarray
                Output complex-valued FFT array
    
    """
    shape_in = array_in.shape
    n_in = shape_in[axis]

    # Pad or truncate if necessary
    if n_fft != n_in:
        if n_fft > n_in:
            pad_width = [(0, 0)] * array_in.ndim
            pad_width[axis] = (0, max(0, n_fft - n_in))
            array_in = np.pad(array_in, pad_width, mode='constant')
        else:
            slicer = [slice(None)] * array_in.ndim
            slicer[axis] = slice(0, n_fft)
            array_in = array_in[tuple(slicer)]

    shape_mod = tuple(array_in.shape)

    a, b, plan = get_fftw_plan(shape_mod, dtype_in=np.complex128,
                               transform_type='c2c_fwd', n_fft=n_fft,
                               axis=axis, threads=threads, order=order)
    
    src = ensure_dtype_and_order(array_in, np.complex128, order)
    np.copyto(a, src, casting='no')
    plan()
    return b.copy() if copy else b


def fft_inverse_c2c(array_in, n_fft=None, axis=0,
                    threads=fftw_threads,
                    order='F', copy=True):
    """
    Perform inverse complex-to-complex FFT using FFTW.
    Inputs
    ------
    array_in : ndarray
               Input complex-valued array (1D or 2D)
    n_fft : int, optional
            Length of the FFT (default: None, uses full length of axis)
    axis : int, optional
           Axis along which to perform the inverse FFT (default: 0)
    threads : int
              Number of threads to use for FFTW (default: fftw_threads)
    order : {'C', 'F'}, optional
            Memory layout of the input array (default: 'F' for Fortran order)
    copy : bool, optional
           Whether to return a copy of the output array (default: True)
    
    Returns
    -------
    array_out : ndarray
                Output complex-valued inverse FFT array
    
    """
    shape_in = array_in.shape
    n_in = shape_in[axis]

    # Pad or truncate if necessary
    if n_fft != n_in:
        if n_fft > n_in:
            pad_width = [(0, 0)] * array_in.ndim
            pad_width[axis] = (0, max(0, n_fft - n_in))
            array_in = np.pad(array_in, pad_width, mode='constant')
        else:
            slicer = [slice(None)] * array_in.ndim
            slicer[axis] = slice(0, n_fft)
            array_in = array_in[tuple(slicer)]

    shape_mod = tuple(array_in.shape)

    a, b, plan = get_fftw_plan(shape_mod, dtype_in=np.complex128,
                               axis=axis, n_fft=n_fft, transform_type='c2c_inv',
                               threads=threads, order=order)
    src = ensure_dtype_and_order(array_in, np.complex128, order)
    np.copyto(b, src, casting='no')
    plan()
    norm = n_fft
    a /= norm
    return a.copy() if copy else a

def ensure_dtype_and_order(arr, dtype, order):
    """Return an array view or new array with requested dtype and order.
    Always returns an array with exact dtype and memory order.
    """
    if arr.dtype != np.dtype(dtype):
        arr = arr.astype(dtype, copy=False)
    # ensure memory order
    if order == 'F':
        if not arr.flags.f_contiguous:
            arr = np.asfortranarray(arr)
    else:
        if not arr.flags.c_contiguous:
            arr = np.ascontiguousarray(arr)
    return arr

# end of file
