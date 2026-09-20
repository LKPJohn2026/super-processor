"""Small compiled primitives used by the toolchain and CV estimators."""


cpdef unsigned long long sum_squares(Py_ssize_t count):
    """Return the sum of ``value * value`` for values below ``count``."""
    if count < 0:
        raise ValueError("count must be non-negative")

    cdef Py_ssize_t index
    cdef unsigned long long total = 0

    for index in range(count):
        total += <unsigned long long>index * <unsigned long long>index

    return total


cpdef double mean_luma(const unsigned char[::1] samples):
    """Return the arithmetic mean of a contiguous 8-bit luma buffer."""
    cdef Py_ssize_t size = samples.shape[0]
    if size == 0:
        raise ValueError("samples must not be empty")

    cdef Py_ssize_t index
    cdef unsigned long long total = 0

    with nogil:
        for index in range(size):
            total += samples[index]

    return total / <double>size


cpdef void histogram_u8(const unsigned char[::1] samples, unsigned long long[::1] out_256):
    """Fill a 256-bin histogram for an 8-bit buffer."""
    cdef Py_ssize_t size = samples.shape[0]
    cdef Py_ssize_t bins = out_256.shape[0]
    if bins != 256:
        raise ValueError("histogram output must have 256 bins")

    cdef Py_ssize_t index
    for index in range(256):
        out_256[index] = 0

    with nogil:
        for index in range(size):
            out_256[samples[index]] += 1


cpdef double percentile_u8(const unsigned char[::1] samples, double percentile):
    """Return an approximate percentile from an 8-bit buffer via histogram."""
    if percentile < 0.0 or percentile > 100.0:
        raise ValueError("percentile must be in [0, 100]")
    cdef Py_ssize_t size = samples.shape[0]
    if size == 0:
        raise ValueError("samples must not be empty")

    cdef unsigned long long hist[256]
    cdef Py_ssize_t index
    for index in range(256):
        hist[index] = 0
    with nogil:
        for index in range(size):
            hist[samples[index]] += 1

    cdef unsigned long long target
    if percentile >= 100.0:
        target = <unsigned long long>size
    else:
        target = <unsigned long long>((percentile / 100.0) * size)
        if target == 0:
            target = 1

    cdef unsigned long long cumulative = 0
    for index in range(256):
        cumulative += hist[index]
        if cumulative >= target:
            return <double>index
    return 255.0


cpdef double variance_u8(const unsigned char[::1] samples):
    """Return the population variance of an 8-bit buffer."""
    cdef Py_ssize_t size = samples.shape[0]
    if size == 0:
        raise ValueError("samples must not be empty")

    cdef Py_ssize_t index
    cdef double mean = 0.0
    cdef double total = 0.0
    cdef double delta

    with nogil:
        for index in range(size):
            total += samples[index]
        mean = total / <double>size
        total = 0.0
        for index in range(size):
            delta = <double>samples[index] - mean
            total += delta * delta

    return total / <double>size


cpdef unsigned long long sad_u8(
    const unsigned char[::1] left,
    const unsigned char[::1] right,
):
    """Return the sum of absolute differences between two equal-length buffers."""
    cdef Py_ssize_t size = left.shape[0]
    if size == 0:
        raise ValueError("samples must not be empty")
    if right.shape[0] != size:
        raise ValueError("buffers must have the same length")

    cdef Py_ssize_t index
    cdef unsigned long long total = 0
    cdef int diff

    with nogil:
        for index in range(size):
            diff = <int>left[index] - <int>right[index]
            if diff < 0:
                diff = -diff
            total += <unsigned long long>diff

    return total
