import numpy as np

CORRELATION_CODEC = {
    None: (1, np.nan),
    np.dtype(np.int8): (100.0, 127.0),
    np.dtype(np.int16): (10000.0, 32767.0),
}


def correlation_codec(dtype: type | np.dtype | None = None, arr = None) -> tuple[float, float]:
    """
    The (scale, nan) pair pyg uses to pack a correlation into dtype, so that a correlation c is stored as
    round(c * scale) and a missing correlation is stored as nan. Floating point dtypes are unscaled and use np.nan.

    :Example:
    ---------
    >>> assert correlation_codec(np.int8) == (100.0, 127.0)
    >>> assert correlation_codec(np.float32)[0] == 1.0
    """
    if dtype is None and arr is not None:
        dtype = arr.dtype
    return CORRELATION_CODEC.get(np.dtype(dtype), (1.0, np.nan))


def decode_correlations(correlations: np.ndarray, dtype: type | np.dtype | None = None) -> np.ndarray:
    """
    Unpacks correlations from pyg's integer representation into float64, mapping the integer nan sentinel to np.nan.
    dtype defaults to the dtype of the array itself.

    :Example:
    ---------
    >>> packed = np.array([[100, 50], [50, 100]], dtype=np.int8)
    >>> assert np.allclose(decode_correlations(packed), np.array([[1.0, 0.5], [0.5, 1.0]]))
    """
    scale, codec_nan = correlation_codec(correlations.dtype if dtype is None else dtype)
    res = np.asarray(correlations, dtype=float)
    if scale == 1.0:
        return res
    return np.where(res == codec_nan, np.nan, res) / scale


def encode_correlations(correlations: np.ndarray, dtype: type | np.dtype | None = None) -> np.ndarray:
    """
    Packs float correlations into dtype, rounding rather than truncating. Rounding is unbiased where truncation
    systematically shrinks correlations towards zero, which quietly overstates diversification.

    :Example:
    ---------
    >>> packed = encode_correlations(np.array([[1.0, 0.505], [0.505, 1.0]]), np.int8)
    >>> assert packed.dtype == np.int8 and packed[0, 1] == 51
    """
    scale, nan = correlation_codec(dtype)
    if scale == 1.0:
        return np.asarray(correlations, dtype=float if dtype is None else dtype)
    res = np.round(np.clip(correlations, -1.0, 1.0) * scale)
    return np.where(np.isnan(res), nan, res).astype(dtype)


def correlation_resolution(dtype: type | np.dtype | None) -> float:
    """
    The smallest correlation difference dtype can represent: 1/scale for pyg's packed integers, and the machine
    epsilon for floats, correlations being of order one.
    """
    scale, _ = correlation_codec(dtype)
    if scale != 1.0:
        return 1 / scale
    return float(np.finfo(float if dtype is None else dtype).eps)# -*- coding: utf-8 -*-

