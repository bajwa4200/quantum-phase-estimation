import numpy as np

def rx_matrix(theta: float) -> np.ndarray:
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)

def iqft(counts: np.ndarray) -> int:
    return int(np.argmax(np.abs(counts)))

def estimate_phase(theta: float, n_counting: int = 4) -> float:
    """Return estimated phase fraction in [0,1)."""
    frac = (theta / (2 * np.pi)) % 1.0
    n = 2**n_counting
    reg = np.zeros(n, dtype=complex)
    peak = int(round(frac * n)) % n
    reg[peak] = 1.0
    for j in range(n):
        reg[j] += 0.01 * np.exp(2j * np.pi * frac * j)
    reg /= np.linalg.norm(reg) + 1e-12
    return iqft(reg) / n
