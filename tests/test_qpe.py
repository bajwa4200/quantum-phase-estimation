import numpy as np
from qpe.core import estimate_phase, rx_matrix, iqft

def test_rx_unitary():
    u = rx_matrix(0.5)
    assert np.allclose(u @ u.conj().T, np.eye(2), atol=1e-6)

def test_estimate_near():
    est = estimate_phase(0.25 * 2 * np.pi, n_counting=5)
    assert abs(est - 0.25) < 0.15

def test_iqft_peak():
    c = np.zeros(8, dtype=complex)
    c[3] = 1
    assert iqft(c) == 3

def test_phase_range():
    e = estimate_phase(1.0, n_counting=4)
    assert 0 <= e < 1

def test_nontrivial_angle():
    assert estimate_phase(0.8, n_counting=4) >= 0
