import numpy as np
from pyg import compiled, df_concat, dictattr, is_df
from analytics.psd_correlation import correlation_codec, quantisation_noise

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

CORRELATION_CODEC = {
    np.dtype(np.int8): (100.0, 127.0),
    np.dtype(np.int16): (10000.0, 32767.0),
}


def correlation_codec(dtype: type | np.dtype | None) -> tuple[float, float]:
    """
    The (scale, nan) pair pyg uses to pack a correlation into dtype, so that a correlation c is stored as
    round(c * scale) and a missing correlation is stored as nan. Floating point dtypes are unscaled and use np.nan.

    :Example:
    ---------
    >>> assert correlation_codec(np.int8) == (100.0, 127.0)
    >>> assert correlation_codec(np.float32)[0] == 1.0
    """
    if dtype is None:
        return 1.0, np.nan
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
    return float(np.finfo(float if dtype is None else dtype).eps)


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


def shrink_to_identity(matrix: np.ndarray, shrinkage: float) -> np.ndarray:
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
        repaired = psd_correlation(matrix, min_eigenvalue=min_eigenvalue, shrinkage=shrinkage)
        res[i] = encode_correlations(repaired, dtype=dtype)
    return res


def psd_correlation(matrix: np.ndarray, 
                    min_eigenvalue: float = 1e-8, 
                    shrinkage: float = 0.0,
                    ffill: bool = True,
                    dtype: type | np.dtype | None = None,                   
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


@compiled
def _ewmcorr_psd(a, w, min_sample, min_periods, demean, shrinkage, scale, nan, is_int, dtype, s,
                 mean,
                 weight,
                 count):
    """
    The synchronised update. a is a (t, m) array of returns, w the decay, and s/mean/weight/count the state carried
    across calls. See the module docstring for the algorithm; this is a straight transcription of it.
    """
    t = a.shape[0]
    m = a.shape[1]
    res = np.full((t, m, m), nan, dtype=dtype)
    decay = np.empty(m)
    x = np.empty(m)
    ok = np.empty(m, dtype=np.bool_)
    root = np.sqrt(w)
    off_diagonal_scale = scale * (1.0 - shrinkage)
    for i in range(t):
        for j in range(m):
            value = a[i, j]
            if np.isnan(value):
                decay[j] = 1.0
                x[j] = 0.0
            else:
                decay[j] = root
                if demean and weight[j] > 0:
                    x[j] = value - mean[j] / weight[j]
                else:
                    x[j] = value
                mean[j] = mean[j] * w + (1.0 - w) * value
                weight[j] = weight[j] * w + (1.0 - w)
                count[j] += 1
        for j in range(m):
            dj = decay[j]
            xj = (1.0 - w) * x[j]
            for k in range(j + 1):
                value = s[j, k] * dj * decay[k] + xj * x[k]
                s[j, k] = value
                s[k, j] = value
            ok[j] = s[j, j] > 0 and weight[j] >= min_sample and count[j] >= min_periods
        for j in range(m):
            if not ok[j]:
                continue
            res[i, j, j] = scale
            sj = np.sqrt(s[j, j])
            for k in range(j):
                if not ok[k]:
                    continue
                c = off_diagonal_scale * s[j, k] / (sj * np.sqrt(s[k, k]))
                if is_int:
                    c = np.floor(c + 0.5) if c >= 0 else -np.floor(0.5 - c)
                res[i, j, k] = c
                res[i, k, j] = c
    return res, s, mean, weight, count


def _w(n: float) -> float:
    """
    pyg's decay convention: n days corresponds to a decay of n / (n + 1) and an effective sample of 2n + 1
    observations. Fractions below 1 are taken as the decay itself. Kept identical to ``pyg_timeseries._math._w`` so
    that a given n means the same thing here as it does in ``ewmcorr``.
    """
    return n / (n + 1) if n >= 1 else float(n)


def effective_sample(n: float) -> float:
    """
    The number of observations an exponentially weighted average of parameter n is worth, (1 + w) / (1 - w) = 2n + 1.

    Worth checking against the number of markets before trusting the small eigenvalues of the result: at n = 128 and
    200 markets the ratio is 200 / 257, at which the eigenvalues of pure noise already span roughly [0.014, 3.55].
    Being positive semi-definite and being well conditioned are separate problems.

    :Example:
    ---------
    >>> assert effective_sample(128) == 257
    """
    w = _w(n)
    return (1 + w) / (1 - w)


def correlation_attenuation(n: float, both: float, neither: float = 0.0) -> float:
    """
    How much the geometric mean decay shrinks the correlation of a pair, given the fraction of dates on which the
    pair both traded and the fraction on which neither did. The remainder are the mixed dates that decay at sqrt(w)
    rather than not at all.

    The cross term reaches a steady state of (1 - w) * both / (1 - E[decay]) times the true covariance, while the
    two variances are unbiased, so this ratio is the attenuation of the correlation itself.

    :Example:
    ---------
    >>> assert correlation_attenuation(128, both=1.0) == 1.0        # no holidays, no attenuation
    >>> assert 0.95 < correlation_attenuation(128, both=0.94) < 1.0 # 6% mixed dates, a few percent
    """
    w = _w(n)
    mixed = 1.0 - both - neither
    expected_decay = both * w + mixed * np.sqrt(w) + neither
    return (1 - w) * both / (1 - expected_decay)


def ewmcorr_psd_(
        a,
        n: float = 128,
        min_sample: float = 0.25,
        min_periods: int = 1,
        demean: bool = True,
        shrinkage: float = 0.0,
        instate: dict | None = None,
        join: str = "outer",
        method: str | None = None,
        dtype: type | np.dtype | None = None,
) -> dictattr:
    """
    A full correlation matrix as a timeseries, positive semi-definite on every date, together with the state needed
    to carry the calculation forward on the next batch of data. See ``ewmcorr_psd`` for the parameters.
    """
    arr = df_concat(a, join=join, method=method) if isinstance(a, (list, dict)) else a
    values = arr.values if is_df(arr) else arr
    if len(values.shape) != 2:
        raise ValueError("a correlation matrix needs a 2-dimensional (t, m) input, got %s" % (values.shape,))
    m = values.shape[1]
    dtype = np.float32 if dtype is None else np.dtype(dtype)
    scale, nan = correlation_codec(dtype)
    if scale != 1.0 and shrinkage < quantisation_noise(dtype, m):
        raise ValueError(
            "shrinkage of %s is below the %s quantisation noise of %.4f at %s markets, so packing can undo the "
            "positive semi-definiteness. Raise shrinkage or use a wider dtype."
            % (shrinkage, np.dtype(dtype).name, quantisation_noise(dtype, m), m)
        )
    state = {} if instate is None else instate
    s = state.get("s", np.zeros((m, m)))
    mean = state.get("mean", np.zeros(m))
    weight = state.get("weight", np.zeros(m))
    count = state.get("count", np.zeros(m, dtype=np.int64))
    res, s, mean, weight, count = _ewmcorr_psd(
        np.asarray(values, dtype=float),
        w=_w(n),
        min_sample=min_sample,
        min_periods=min_periods,
        demean=demean,
        shrinkage=shrinkage,
        scale=scale,
        nan=nan,
        is_int=scale != 1.0,
        dtype=dtype,
        s=s,
        mean=mean,
        weight=weight,
        count=count,
    )
    state = dictattr(s=s, mean=mean, weight=weight, count=count)
    index = arr.index if is_df(arr) else None
    columns = list(arr.columns) if is_df(arr) else None
    return dictattr(data=res, state=state, index=index, columns=columns)


ewmcorr_psd_.output = ["data", "state", "index", "columns"]


def ewmcorr_psd(
        a,
        n: float = 128,
        min_sample: float = 0.25,
        min_periods: int = 1,
        demean: bool = True,
        shrinkage: float = 0.0,
        instate: dict | None = None,
        join: str = "outer",
        method: str | None = None,
        dtype: type | np.dtype | None = None,
) -> np.ndarray:
    """
    A full correlation matrix as a timeseries, positive semi-definite on every date.

    A drop-in alternative to ``pyg.ewmcorr`` with the same n convention and the same integer packing, differing in
    that all state is kept per market rather than per pair. That is what buys the positive semi-definiteness; the
    cost is that a pair is only updated on the dates both its markets traded, rather than across the gaps.


    Synchronised exponentially weighted correlation, positive semi-definite by construction.
    
    ``ewmcorr`` is a *pairwise* estimator: ``_ewmcorrelation`` gives every pair (j, k) its own return interval
    (``prev[j, k]`` is j's last value on a date when both j and k traded), its own decay clock (``n0[j, k]`` only
    advances when both traded) and its own forward-fill. Each of those is a defensible pairwise choice, and together
    they mean no single vector x_t exists whose outer products build the matrix. Without that Gram representation
    positive semi-definiteness is not merely unenforced, it is unattainable, and the resulting matrix routinely admits
    portfolios of negative variance. Use psd = True to ensure it comes back with a PSD correlation.
    
    The estimator here keeps all state **per column** rather than per pair, so the matrix is a weighted sum of rank one
    outer products and is positive semi-definite by construction:
    
        S_t = diag(d_t) . S_{t-1} . diag(d_t)  +  (1 - w) . x_t x_t'
    
    where x_t is the (demeaned) return vector with missing markets set to zero, and d_j = sqrt(w) if market j traded on
    date t and 1 if it did not. Both terms are positive semi-definite - the first is a congruence of S_{t-1}, the
    second a rank one outer product - so S_t is too, for any starting S_0 = 0.
    
    Why sqrt(w). What you would like on a missing date is to decay S_jk only when *both* j and k traded, that is to
    multiply elementwise by M_jk = w ** (o_j AND o_k). That M is not positive semi-definite, so it breaks the matrix.
    M_jk = w ** ((o_j + o_k) / 2) = d_j * d_k is rank one, and an elementwise product with a rank one matrix is exactly
    the diagonal congruence above. The geometric mean is therefore the unique positive-semi-definite-safe reading of
    "decay only when both traded". The price is that mixed pairs (j traded, k did not) decay at sqrt(w) rather than not
    at all, so cross correlations to markets on a different holiday calendar are attenuated slightly; see
    ``correlation_attenuation`` for the size of that, which is a few percent at typical holiday frequencies.
    
    Everything else follows from keeping state per market:
    
    - the correlation is D^-1/2 S D^-1/2, a congruence, so it is positive semi-definite with an exactly unit diagonal;
    - the mean is subtracted using the mean up to t-1, keeping each update a genuine rank one outer product;
    - min_sample/min_periods gate whole rows and columns rather than individual pairs, because a principal submatrix of
      a positive semi-definite matrix is positive semi-definite but a matrix with individual entries nan'd out is not;
    - the exponentially weighted initialisation bias is a per-market factor and so cancels in the correlation.

    :Parameters:
    ------------
    a : pd.DataFrame or np.ndarray or list/dict of them
        (t, m) *returns*, nan on dates a market did not trade. Note that ``pyg.ewmcorr`` also takes returns while
        ``pyg.ewmcorrelation`` takes total returns; this follows ``ewmcorr``. For overlapping multi-day returns -
        the usual remedy for markets in different time zones closing at different times - pass ``rolling_sum(a, k)``
        in, as ``analytics.min_variance_weights.ewmcorrelation`` already does.
    n : float
        the exponential decay in days; the decay is n / (n + 1) and the effective sample 2n + 1 observations.
    min_sample : float
        accumulated weight a market needs before it carries a correlation. Gates the whole row and column.
    min_periods : int
        observations a market needs before it carries a correlation. Gates the whole row and column.
    demean : bool
        subtract an exponentially weighted mean, using the mean up to t-1 so that the update stays rank one. True
        matches ``ewmcorr``; False assumes a zero mean, which is usually the better risk estimate for daily returns
        and is marginally less noisy.
    shrinkage : float
        blend towards the identity, leaving a guaranteed smallest eigenvalue of at least this. For the packed
        integer dtypes it is required to be at least ``quantisation_noise(dtype, m)`` and rejected otherwise, since
        rounding a correlation to int8 perturbs the spectrum by roughly 0.08 at 200 markets and can undo the
        positive semi-definiteness on its own.
    instate : dict
        state from a previous call, to continue the calculation on new data rather than recompute from scratch.
    join, method : str
        passed to ``df_concat`` when a is a list or dict of timeseries.
    dtype : type
        dtype of the result, defaulting to np.float32 as pyg does. np.int8 packs a correlation as a percentage and
        np.int16 to four decimal places, at 1/4 and 1/2 of the memory; a (7000, 200, 200) result is 1.1GB as
        float32, 560MB as int16 and 280MB as int8.

    :Returns:
    ---------
    np.ndarray of shape (t, m, m).



    :Example: positive semi-definite where the pairwise estimator is not
    ---------
    >>> import numpy as np
    >>> from pyg import ewmcorr
    >>> from analytics.portfolio import turnover
    >>> from analytics.psd_correlation import min_eigenvalues, psd_correlation
    >>> dates = drange(-4999)
    >>> rtn = np.random.normal(0, 1, (5000, 20))
    >>> rtn[np.random.uniform(0, 1, rtn.shape) < 0.2] = np.nan
    >>> c = ewmcorr(rtn, 32)
    >>> d = ewmcorr_psd(rtn, 32)
    >>> c1 = psd_correlation(c)
    >>> i = 0; j = 1
    >>> turnover(pd.DataFrame(c[:,:,0], dates)).mean()
    >>> turnover(pd.DataFrame(d[:,:,0], dates)).mean()
    >>> ts = pd.DataFrame(dict(pyg = c[:,i,j], psd = d[:,i,j]), dates)
    >>> ts.plot()

    >>> assert pd.Series(min_eigenvalues(c)).min() < 0
    >>> assert pd.Series(min_eigenvalues(d)).min() >= -1e-6
    """
    return ewmcorr_psd_(
        a,
        n=n,
        min_sample=min_sample,
        min_periods=min_periods,
        demean=demean,
        shrinkage=shrinkage,
        instate=instate,
        join=join,
        method=method,
        dtype=dtype,
    ).data


def ewmcorrelation_psd(
        a,
        n: float = 128,
        columns: list | None = None,
        min_sample: float = 0.25,
        min_periods: int = 1,
        demean: bool = True,
        shrinkage: float = 0.0,
        dtype: type | np.dtype | None = None,
) -> dictattr:
    """
    ``ewmcorr_psd`` presented the way ``analytics.min_variance_weights.ewmcorrelation`` presents ``pyg.ewmcorr``, as
    a dict of correlations, index and columns, so that the two can be swapped in the optimiser and compared.

    Note that there is no ``method``/forward-fill argument. ``ewmcorrelation`` forward-fills the correlations after
    the fact, which freezes different pairs at different dates and is one of the three reasons its output is
    indefinite; here a market either carries a correlation to every other qualifying market or to none.
    """
    a = df_concat(a, columns=columns)
    res = ewmcorr_psd_(
        a,
        n=n,
        min_sample=min_sample,
        min_periods=min_periods,
        demean=demean,
        shrinkage=shrinkage,
        dtype=dtype,
    )
    return dictattr(correlations=res.data, index=res.index, columns=res.columns)


ewmcorrelation_psd.output = ["correlations", "index", "columns"]


