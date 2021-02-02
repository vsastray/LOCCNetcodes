import numpy as np
from paddle_quantum.state import bell_state


def amplitude_damping(gamma):
    e0 = np.array([[1, 0], [0, np.sqrt(1 - gamma)]], dtype='complex128')
    e1 = np.array([[0, np.sqrt(gamma)], [0, 0]], dtype='complex128')
    ops = [e0, e1]
    return ops


def ad_state(gamma, state):
    amp_ops = amplitude_damping(gamma)
    output_state = np.zeros_like(state)
    for operator in amp_ops:
        output_state += operator @ state @ operator.conj().T
    return output_state


def ad_choi_state(gamma):
    # Maximally entangled state
    state = bell_state(2)
    # Identity matrix
    I = np.eye(2, dtype='complex128')
    # Amplitude damping operators
    amplitude = amplitude_damping(gamma)

    output_state = np.zeros_like(state)
    for i in range(len(amplitude)):
        operator = np.kron(I, amplitude[i])
        output_state += np.dot(operator, np.dot(state, operator.conj().T))
    return output_state
