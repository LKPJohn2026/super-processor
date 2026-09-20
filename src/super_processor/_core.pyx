"""Small compiled primitives used to verify the Cython toolchain."""


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
