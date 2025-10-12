import numpy as np
import math

def binarize(signal):
    median = np.median(signal)
    binary_seq = (signal > median).astype(int)
    return binary_seq

def lempel_ziv_complexity(signal):

    # signal = convert_to_binary(signal)

    binary_seq = binarize(signal)

    s = ''.join(map(str, binary_seq))
    i, k, l = 0, 1, 1
    n = len(s)
    C = 1

    while True:
        if s[i + k - 1] != s[l + k - 1]:
            if k > l:
                l = k
            i += 1
            if i == l:
                C += 1
                l += 1
                if l + 1 > n:
                    break
                i = 0
            k = 1
        else:
            k += 1
            if l + k > n:
                C += 1
                break
    
    C = C / (n / math.log2(n))

    return C

def multiscale_entropy(signal, m=2, r=None, maxscale =20):

    N = len(signal)
    signal = np.array(signal, dtype=np.float64)

    if r is None:
        r = 0.15 * np.std(signal)

    mse_values = []

    for scale in range(1, maxscale + 1):
        if N//scale < m + 1:
            mse_values.append(0.0)
            continue
        coarse_grain = np.mean(signal[:N - N % scale].reshape(-1, scale), axis=1)
        sampen = sample_entropy(coarse_grain, m, r)
        mse_values.append(sampen)

    return float(np.nanmean(mse_values))
    

def sample_entropy(signal, m=2, r=None, maxscale =20):
    N = len(signal)
    signal = np.array(signal, dtype=np.float64)

    if N < m + 1:
        return 0.0

    if r is None:
        r = 0.2 * np.std(signal)

    def _phi(m):
        if N <= m:
            return 0.0
        x = np.array([signal[i:i + m] for i in range(N - m + 1)])
        C = 0
        for i in range(len(x)):
            C += np.sum(np.max(np.abs(x - x[i]), axis=1) <= r) - 1
        return C / (N - m + 1)
    phi_m = _phi(m)
    phi_m1 = _phi(m + 1)

    if phi_m == 0 or phi_m1 == 0 or np.isnan(phi_m) or np.isnan(phi_m1):
        return 0.0
    
    return float(-np.log(phi_m1 / phi_m))