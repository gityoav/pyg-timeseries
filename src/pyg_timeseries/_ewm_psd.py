import numpy as np
from pyg_base import df_concat, dictattr, is_df
from pyg_timeseries._cor import correlation_codec
from pyg_timeseries._psd import  quantisation_noise
from pyg_timeseries._math import  _w
from pyg_timeseries._decorators import compiled
from pyg_timeseries._rolling import rolling_sum

def _overlapping_returns(values, overlapping, tail = None):
    """
    Calendar-anchored k-day returns
    
    :Parameters:
    ------------
    values : np.ndarray
        (t, m) returns, nan where a market did not trade.
    overlapping : int
        window length k in rows. 1 returns the input unchanged.
    tail : np.ndarray
        the last k-1 rows of the previous batch, so that an incremental call sees full windows from its first date.

    :Returns:
    ---------
    (x, tail) - the (t, m) k-day returns aligned to the input dates, and the tail to carry into the next call.

    :Example:
    ---------
    >>> a = np.array([np.nan, np.nan, 1., 2., np.nan, 3., np.nan, np.nan, np.nan, np.nan]).reshape(-1, 1)
    >>> x, _ = _overlapping_returns(a, 3)
    >>> # observations in window: -, -, 1, 2, 2, 2, 1, 1, 0, 0
    >>> # k-day return          : -, -, 1, 3, 3, 5, 3, 3, -, -   <- nan once the window holds nothing
    """
    if overlapping <= 1:
        return values, np.zeros((0, values.shape[1]))
    prefix = 0 if tail is None else len(tail)
    joined = values if prefix == 0 else np.concatenate([tail, values])
    observed = (~np.isnan(joined)).astype(float)
    total = rolling_sum(np.where(observed > 0, joined, 0.0), overlapping)
    count = rolling_sum(observed, overlapping)
    x = np.where(count > 0, total, np.nan)
    return x[prefix:], joined[-(overlapping - 1):]
  
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




def effective_sample(n: float, overlapping: int = 1) -> float:
    """
    The number of observations an exponentially weighted average of parameter n is worth, (1 + w) / (1 - w) = 2n + 1,
    divided by the overlap.

    Sampling a k-day return every day means consecutive observations share k-1 days, so the exponentially weighted
    average counts roughly k times more samples than it has independent information. This is a precision correction
    rather than a bias one - for a stationary process the overlapping estimator is still consistent, because the
    overlap scales the numerator and the denominator alike - but it is what should be compared against the number of
    markets before trusting small eigenvalues or an inverse.

    The 1/k factor is the large-sample limit for a flat rectangular window over serially uncorrelated returns. The
    true factor depends on the window shape and on the autocorrelation of the underlying returns, so treat it as an
    order-of-magnitude correction and not a precise one.

    :Example:
    ---------
    >>> assert effective_sample(128) == 257
    >>> assert effective_sample(1024, overlapping = 5) == 2049 / 5      # ~410, against 146-200 markets
    """
    w = _w(n)
    return (1 + w) / (1 - w) / overlapping


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
        overlapping: int = 1,
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
    values, tail = _overlapping_returns(np.asarray(values, dtype=float),                   # CHANGED: build the
                                        overlapping, state.get("tail"))                    #          k-day return
    res, s, mean, weight, count = _ewmcorr_psd(
        values,
        w=_w(n),
        min_sample=min_sample,                                 # a fraction of the decay window, so not scaled
        min_periods=min_periods * overlapping,                 # CHANGED: min_periods counts independent windows
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
    state = dictattr(s=s, mean=mean, weight=weight, count=count, tail=tail)                # CHANGED: carry the tail
    index = arr.index if is_df(arr) else None
    columns = list(arr.columns) if is_df(arr) else None
    return dictattr(data=res, state=state, index=index, columns=columns)


ewmcorr_psd_.output = ["data", "state", "index", "columns"]


def ewmcorr_psd(
        a,
        n: float = 128,
        overlapping: int = 1,
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
        overlapping = overlapping,
        min_sample=min_sample,
        min_periods=min_periods,
        demean=demean,
        shrinkage=shrinkage,
        instate=instate,
        join=join,
        method=method,
        dtype=dtype,
    ).data




