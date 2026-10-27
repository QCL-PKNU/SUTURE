from qiskit_aer import AerSimulator

def detect_device():
    """Return 'GPU' if GPU device is available, otherwise return 'CPU'."""
    if 'GPU' in AerSimulator().available_devices():
        return 'GPU'
    return 'CPU'

# simulation_device = detect_simulation_device()
