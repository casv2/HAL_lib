# based on
#     https://github.com/numpy/numpy/blob/8d61ebc25a117337d148f1e3d96066653bd6419a/numpy/random/mtrand.pyx#L4041
# but adding hermitian=True to svd call
import numpy as np
import warnings

def multivariate_normal(mean, cov, size=None, check_valid='warn', tol=1e-8):
    """
    multivariate_normal(mean, cov, size=None, check_valid='warn', tol=1e-8)
    Draw random samples from a multivariate normal distribution.
    The multivariate normal, multinormal or Gaussian distribution is a
    generalization of the one-dimensional normal distribution to higher
    dimensions.  Such a distribution is specified by its mean and
    covariance matrix.  These parameters are analogous to the mean
    (average or "center") and variance (standard deviation, or "width,"
    squared) of the one-dimensional normal distribution.
    .. note::
        New code should use the
        `~numpy.random.Generator.multivariate_normal`
        method of a `~numpy.random.Generator` instance instead;
        please see the :ref:`random-quick-start`.
    Parameters
    ----------
    mean : 1-D array_like, of length N
        Mean of the N-dimensional distribution.
    cov : 2-D array_like, of shape (N, N)
        Covariance matrix of the distribution. It must be symmetric and
        positive-semidefinite for proper sampling.
    size : int or tuple of ints, optional
        Given a shape of, for example, ``(m,n,k)``, ``m*n*k`` samples are
        generated, and packed in an `m`-by-`n`-by-`k` arrangement.  Because
        each sample is `N`-dimensional, the output shape is ``(m,n,k,N)``.
        If no shape is specified, a single (`N`-D) sample is returned.
    check_valid : { 'warn', 'raise', 'ignore' }, optional
        Behavior when the covariance matrix is not positive semidefinite.
    tol : float, optional
        Tolerance when checking the singular values in covariance matrix.
        cov is cast to double before the check.
    Returns
    -------
    out : ndarray
        The drawn samples, of shape *size*, if that was provided.  If not,
        the shape is ``(N,)``.
        In other words, each entry ``out[i,j,...,:]`` is an N-dimensional
        value drawn from the distribution.
    See Also
    --------
    random.Generator.multivariate_normal: which should be used for new code.
    Notes
    -----
    The mean is a coordinate in N-dimensional space, which represents the
    location where samples are most likely to be generated.  This is
    analogous to the peak of the bell curve for the one-dimensional or
    univariate normal distribution.
    Covariance indicates the level to which two variables vary together.
    From the multivariate normal distribution, we draw N-dimensional
    samples, :math:`X = [x_1, x_2, ... x_N]`.  The covariance matrix
    element :math:`C_{ij}` is the covariance of :math:`x_i` and :math:`x_j`.
    The element :math:`C_{ii}` is the variance of :math:`x_i` (i.e. its
    "spread").
    Instead of specifying the full covariance matrix, popular
    approximations include:
      - Spherical covariance (`cov` is a multiple of the identity matrix)
      - Diagonal covariance (`cov` has non-negative elements, and only on
        the diagonal)
    This geometrical property can be seen in two dimensions by plotting
    generated data-points:
    >>> mean = [0, 0]
    >>> cov = [[1, 0], [0, 100]]  # diagonal covariance
    Diagonal covariance means that points are oriented along x or y-axis:
    >>> import matplotlib.pyplot as plt
    >>> x, y = np.random.multivariate_normal(mean, cov, 5000).T
    >>> plt.plot(x, y, 'x')
    >>> plt.axis('equal')
    >>> plt.show()
    Note that the covariance matrix must be positive semidefinite (a.k.a.
    nonnegative-definite). Otherwise, the behavior of this method is
    undefined and backwards compatibility is not guaranteed.
    References
    ----------
    .. [1] Papoulis, A., "Probability, Random Variables, and Stochastic
           Processes," 3rd ed., New York: McGraw-Hill, 1991.
    .. [2] Duda, R. O., Hart, P. E., and Stork, D. G., "Pattern
           Classification," 2nd ed., New York: Wiley, 2001.
    Examples
    --------
    >>> mean = (1, 2)
    >>> cov = [[1, 0], [0, 1]]
    >>> x = np.random.multivariate_normal(mean, cov, (3, 3))
    >>> x.shape
    (3, 3, 2)
    Here we generate 800 samples from the bivariate normal distribution
    with mean [0, 0] and covariance matrix [[6, -3], [-3, 3.5]].  The
    expected variances of the first and second components of the sample
    are 6 and 3.5, respectively, and the expected correlation
    coefficient is -3/sqrt(6*3.5) ≈ -0.65465.
    >>> cov = np.array([[6, -3], [-3, 3.5]])
    >>> pts = np.random.multivariate_normal([0, 0], cov, size=800)
    Check that the mean, covariance, and correlation coefficient of the
    sample are close to the expected values:
    >>> pts.mean(axis=0)
    array([ 0.0326911 , -0.01280782])  # may vary
    >>> np.cov(pts.T)
    array([[ 5.96202397, -2.85602287],
           [-2.85602287,  3.47613949]])  # may vary
    >>> np.corrcoef(pts.T)[0, 1]
    -0.6273591314603949  # may vary
    We can visualize this data with a scatter plot.  The orientation
    of the point cloud illustrates the negative correlation of the
    components of this sample.
    >>> import matplotlib.pyplot as plt
    >>> plt.plot(pts[:, 0], pts[:, 1], '.', alpha=0.5)
    >>> plt.axis('equal')
    >>> plt.grid()
    >>> plt.show()
    """
    from numpy.linalg import svd

    # Check preconditions on arguments
    mean = np.array(mean)
    cov = np.array(cov)
    if size is None:
        shape = []
    elif isinstance(size, (int, np.integer)):
        shape = [size]
    else:
        shape = size

    if len(mean.shape) != 1:
        raise ValueError("mean must be 1 dimensional")
    if (len(cov.shape) != 2) or (cov.shape[0] != cov.shape[1]):
        raise ValueError("cov must be 2 dimensional and square")
    if mean.shape[0] != cov.shape[0]:
        raise ValueError("mean and cov must have same length")

    # Compute shape of output and create a matrix of independent
    # standard normally distributed random numbers. The matrix has rows
    # with the same length as mean and as many rows are necessary to
    # form a matrix of shape final_shape.
    final_shape = list(shape[:])
    final_shape.append(mean.shape[0])
    x = np.random.standard_normal(final_shape).reshape(-1, mean.shape[0])

    # Transform matrix of standard normals into matrix where each row
    # contains multivariate normals with the desired covariance.
    # Compute A such that dot(transpose(A),A) == cov.
    # Then the matrix products of the rows of x and A has the desired
    # covariance. Note that sqrt(s)*v where (u,s,v) is the singular value
    # decomposition of cov is such an A.
    #
    # Also check that cov is positive-semidefinite. If so, the u.T and v
    # matrices should be equal up to roundoff error if cov is
    # symmetric and the singular value of the corresponding row is
    # not zero. We continue to use the SVD rather than Cholesky in
    # order to preserve current outputs. Note that symmetry has not
    # been checked.

    # GH10839, ensure double to make tol meaningful
    cov = cov.astype(np.double)
    (u, s, v) = svd(cov, hermitian=True)

    if check_valid != 'ignore':
        if check_valid != 'warn' and check_valid != 'raise':
            raise ValueError(
                "check_valid must equal 'warn', 'raise', or 'ignore'")

        psd = np.allclose(np.dot(v.T * s, v), cov, rtol=tol, atol=tol)
        if not psd:
            if check_valid == 'warn':
                warnings.warn("covariance is not symmetric positive-semidefinite.",
                    RuntimeWarning)
            else:
                raise ValueError(
                    "covariance is not symmetric positive-semidefinite.")

    x = np.dot(x, np.sqrt(s)[:, None] * v)
    x += mean
    x.shape = tuple(final_shape)
    return x
