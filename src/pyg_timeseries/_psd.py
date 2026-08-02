import numpy as np
from pyg_timeseries._cor import correlation_resolution, correlation_codec, decode_correlations, encode_correlations

"""
Positive semi-definite correlation matrices.

A correlation matrix estimated pair-by-pair - as pyg's ``ewmcorr`` does, giving every pair its own return interval,
its own decay clock and its own forward-fill - has no representation as a Gram matrix and is therefore routinely
indefinite. This module holds the matrix-level primitives used to detect that and to repair it:

- ``clip_eigenvalues``    : project onto the positive semi-definite cone (the "B1" repair)
- ``normalise_diagonal``  : congruence back to a unit diagonal, which preserves the sign of every eigenvalue
- ``shrink_to_identity``  : blend towards the identity, which raises the smallest eigenvalue by a known amount
- ``psd_correlation``     : the three composed, NaN aware, for a single matrix or a whole timeseries
- ``psd_correlation_3d``  : the same over a (t, m, m) timeseries, with control over the stored dtype
- ``min_eigenvalues``     : diagnostic, the smallest eigenvalue of each matrix in a timeseries

It also holds the codec for packed integer correlations, so that a repaired timeseries can be handed back in
the same dtype it arrived in. See ``ewm_psd_correlation`` for an estimator that is positive semi-definite
by construction and so needs no repair.
"""


def quantisation_noise(dtype: type | np.dtype | None, m: int) -> float:
    """
    An estimate of how far storing into dtype can move the smallest eigenvalue of an m x m correlation matrix.

    Rounding to a resolution r adds a symmetric error matrix whose entries are roughly uniform on +/- r/2, so with a
    standard deviation of r/sqrt(12). By the semi-circle law the spectral norm of such a matrix is about
    2 * sigma * sqrt(m). A matrix whose smallest eigenvalue is below this can come back indefinite once stored, so
    it is the natural floor to demand of ``min_eigenvalue``/``shrinkage``, and the natural tolerance to test to.

    For floats this is unavoidable arithmetic noise. For packed integers it is a deliberate trade of accuracy
    for memory, and it is large: int8 at 200 markets can move the smallest eigenvalue by about 0.08.

    :Example:
    ---------
    >>> assert 0.05 < quantisation_noise(np.int8, 200) < 0.12    # int8 needs a real eigenvalue floor
    >>> assert quantisation_noise(np.int16, 200) < 0.002         # int16 barely does
    >>> assert quantisation_noise(np.float32, 200) < 1e-5        # float32 does not
    """
    return 2 * np.sqrt(m / 12) * correlation_resolution(dtype)


def clip_eigenvalues(matrix: np.ndarray, min_eigenvalue: float = 0.0) -> np.ndarray:
    """
    Projects a symmetric matrix onto the cone of matrices whose eigenvalues are at least min_eigenvalue. This is the
    nearest such matrix in the Frobenius norm, and needs a single eigendecomposition, where Higham's nearest
    correlation matrix additionally imposes a unit diagonal and needs tens of them.

    :Example:
    ---------
    >>> indefinite = np.array([[1.0, 0.9, -0.9], [0.9, 1.0, 0.9], [-0.9, 0.9, 1.0]])
    >>> assert np.linalg.eigvalsh(indefinite)[0] < 0
    >>> assert np.linalg.eigvalsh(clip_eigenvalues(indefinite))[0] > -1e-12
    """
    values, vectors = np.linalg.eigh(matrix)
    if values[0] >= min_eigenvalue:
        return matrix
    return (vectors * np.maximum(values, min_eigenvalue)) @ vectors.T


def normalise_diagonal(matrix: np.ndarray) -> np.ndarray:
    """
    Rescales a positive definite matrix to a unit diagonal. This is a congruence, so every eigenvalue keeps its sign
    and the result is still positive definite, though the smallest eigenvalue itself is not preserved.
    """
    scale = np.sqrt(np.diag(matrix))
    return matrix / np.outer(scale, scale)


def shrink_to_identity(matrix: np.ndarray, shrinkage: float = 0) -> np.ndarray:
    """
    Blends a correlation matrix towards the identity. The diagonal is untouched and the smallest eigenvalue moves
    from lambda to (1 - shrinkage) * lambda + shrinkage, so any shrinkage guarantees a floor of that size.

    :Example:
    ---------
    >>> matrix = np.array([[1.0, 0.99], [0.99, 1.0]])
    >>> assert np.linalg.eigvalsh(shrink_to_identity(matrix, 0.1))[0] > 0.1
    """
    if not shrinkage:
        return matrix
    return (1 - shrinkage) * matrix + shrinkage * np.eye(matrix.shape[0])


def _valid_markets(matrix: np.ndarray) -> np.ndarray:
    """
    The markets carrying a correlation on a given date, identified by a non-nan diagonal. Dropping the others leaves
    a principal submatrix, which is positive semi-definite whenever the full matrix is, so gating market by market
    is safe where gating pair by pair is not.
    """
    return ~np.isnan(np.diag(matrix))


def _psd_correlation_3d(
        correlations: np.ndarray,
        min_eigenvalue: float = 1e-8,
        symmetric = True,
        shrinkage: float = 0.0,
        ffill: bool = True,
        dtype: type | np.dtype | None = None,
) -> np.ndarray:
    """
    Repairs a (t, m, m) timeseries of correlation matrices one date at a time.
    pairwise estimate, with all the information it extracts from partially overlapping histories, and pay an
    eigendecomposition per date to make each matrix usable.

    Forward-filling and repairing compose in exactly the right order here. Carrying each entry forward on its own is
    one of the reasons a pairwise matrix is indefinite in the first place - it freezes different pairs at different
    dates - but doing it *before* the eigenvalue clip costs nothing, because the clip repairs whatever it is handed.
    So a pair with no estimate today gets its most recent estimate rather than a zero, and the matrix is still
    positive semi-definite. Only pairs that have never had an estimate fall back to zero.

    Leading dates, where a market has no correlation to carry forward, stay nan; markets absent on a date are
    dropped from it and returned as nan, since a principal submatrix of a positive semi-definite matrix is itself
    positive semi-definite.

    :Parameters:
    ------------
    correlations : np.ndarray
        (t, m, m) correlations, float or packed into int8/int16 by pyg.
    min_eigenvalue : float
        eigenvalues below this are clipped up to it. The default is a token positive value: exact zeros make the
        matrix singular and the diagonal renormalisation that follows undefined for a null row.
    shrinkage : float
        blend towards the identity applied after the repair, leaving a guaranteed eigenvalue floor. Should be at
        least ``quantisation_noise(dtype, m)`` when packing back into int8 or int16.
    ffill : bool
        carry each correlation forward over dates where it has no estimate, before repairing.
    dtype : type
        dtype of the result. The default keeps the dtype of the input.

    :Returns:
    ---------
    np.ndarray of the same shape as correlations.

    :Example:
    ---------
    >>> from pyg import ewmcorr                                             # doctest: +SKIP
    >>> correlations = psd_correlation_3d(ewmcorr(rtn, 32))                 # doctest: +SKIP
    >>> assert np.nanmin(min_eigenvalues(correlations)) > -1e-6             # doctest: +SKIP
    """
    dtype = correlations.dtype if dtype is None else np.dtype(dtype)
    _, nan = correlation_codec(dtype)
    res = np.full(correlations.shape, nan, dtype=dtype)
    previous = np.full(correlations.shape[1:], np.nan)
    for i in range(correlations.shape[0]):
        matrix = decode_correlations(correlations[i])
        if ffill:
            matrix = np.where(np.isnan(matrix), previous, matrix)
            previous = matrix
        repaired = psd_correlation(matrix, min_eigenvalue=min_eigenvalue, shrinkage=shrinkage, symmetric = symmetric)
        res[i] = encode_correlations(repaired, dtype=dtype)
    return res


def psd_correlation(matrix: np.ndarray, 
                    min_eigenvalue: float = 1e-8, 
                    shrinkage: float = 0.0,
                    ffill: bool = True,
                    dtype: type | np.dtype | None = None,
                    symmetric = True,
                   ) -> np.ndarray:
    """
    Repairs an indefinite correlation matrix: clip the spectrum, rescale back to a unit diagonal, then optionally
    shrink towards the identity to leave a guaranteed eigenvalue floor.

    Accepts either a single (m, m) matrix or a (t, m, m) timeseries of them, the latter by delegating to
    ``psd_correlation_3d``. LAPACK cannot decompose a matrix containing nan, so markets with no correlation (a nan
    diagonal) are dropped and returned as nan, and any remaining nan among the survivors is read as a zero
    correlation - the only choice that does not propagate one gap across the whole matrix.

    min_eigenvalue is applied before the diagonal is renormalised and so does not survive as an exact floor; use
    shrinkage where an exact floor matters, for instance to survive packing into int8. It defaults to a token
    positive value rather than zero because a fully clipped eigenvalue leaves a null row, whose diagonal cannot
    then be renormalised.

    :Example:
    ---------
    >>> indefinite = np.array([[1.0, 0.9, -0.9], [0.9, 1.0, 0.9], [-0.9, 0.9, 1.0]])
    >>> repaired = psd_correlation(indefinite)
    >>> assert np.allclose(np.diag(repaired), 1.0)
    >>> assert np.linalg.eigvalsh(repaired)[0] >= 0

    :Example: a market that carries no correlation on the date
    ---------
    >>> partial = np.array([[1.0, 0.9, np.nan], [0.9, 1.0, np.nan], [np.nan, np.nan, np.nan]])
    >>> repaired = psd_correlation(partial)
    >>> assert np.isnan(repaired[2]).all() and not np.isnan(repaired[:2, :2]).any()
    """
    if matrix.ndim == 3:
        return _psd_correlation_3d(matrix, min_eigenvalue=min_eigenvalue, shrinkage=shrinkage, ffill = ffill, dtype = dtype)
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("psd_correlation needs a square (m, m) matrix or a (t, m, m) timeseries, got %s"
                         % (matrix.shape,))
    if symmetric:
        matrix = 0.5*(matrix + matrix.T)
    valid = _valid_markets(matrix)
    if not valid.any():
        return np.full(matrix.shape, np.nan)
    block = matrix if valid.all() else matrix[np.ix_(valid, valid)]
    if np.isnan(block).any():
        block = np.nan_to_num(block, nan=0.0)
    clipped = clip_eigenvalues(block, min_eigenvalue)
    if clipped is not block:
        clipped = normalise_diagonal(clipped)
    repaired = shrink_to_identity(clipped, shrinkage)
    if valid.all():
        return repaired
    res = np.full(matrix.shape, np.nan)
    res[np.ix_(valid, valid)] = repaired
    return res


def min_eigenvalues(correlations: np.ndarray, dtype: type | np.dtype | None = None) -> np.ndarray:
    """
    The smallest eigenvalue of each matrix in a (t, m, m) timeseries, over the markets that carry a correlation on
    that date. Negative values are the symptom being chased: a matrix with a negative eigenvalue admits a portfolio
    of apparently negative variance, which a mean-variance optimiser will happily walk towards.

    Dates with no valid market give nan.

    :Example:
    ---------
    >>> matrices = np.array([np.eye(3), [[1.0, 0.9, -0.9], [0.9, 1.0, 0.9], [-0.9, 0.9, 1.0]]])
    >>> assert np.allclose(min_eigenvalues(matrices)[0], 1.0)
    >>> assert min_eigenvalues(matrices)[1] < 0
    """
    res = np.full(correlations.shape[0], np.nan)
    for i in range(correlations.shape[0]):
        matrix = decode_correlations(correlations[i], dtype)
        valid = _valid_markets(matrix)
        if not valid.any():
            continue
        res[i] = np.linalg.eigvalsh(np.nan_to_num(matrix[np.ix_(valid, valid)], nan=0.0))[0]
    return res
# -*- coding: utf-8 -*-

