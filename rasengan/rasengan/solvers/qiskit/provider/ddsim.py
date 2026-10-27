from .provider import Provider, EXTENDED_BASIS_GATES
from qiskit import QuantumCircuit, transpile
from mqt import ddsim
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from mqt.ddsim.stochasticnoisesimulator import StochasticNoiseSimulatorBackend
from typing import Dict
from qiskit_ibm_runtime.fake_provider import FakeKyiv, FakeTorino, FakeBrisbane,FakePeekskill, FakeQuebec


class DdsimProvider(Provider):
    def __init__(self, transpile_mode=1):
        super().__init__(transpile_mode=transpile_mode)
        self.backend = ddsim.DDSIMProvider().get_backend("qasm_simulator")
        self.pass_manager = generate_preset_pass_manager(
            optimization_level=2,
            basis_gates=EXTENDED_BASIS_GATES,
        )

    def get_counts(self, qc: QuantumCircuit, shots: int):
        job = self.backend.run(qc, shots=shots)
        counts = job.result().get_counts()
        return counts
        # counts = job.result().get_counts(qc)

class NoisyDdsimProvider(Provider):
    def __init__(self, **kwargs):
        super().__init__()
        self.transpile_backend = FakeQuebec()
        self.backend = StochasticNoiseSimulatorBackend()
        self.pass_manager = generate_preset_pass_manager(
            optimization_level=2,
            basis_gates=EXTENDED_BASIS_GATES,
        )
        # 0.0, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05
        self.amp_damping_probability = kwargs.get('amp_damping_probability', 0.0025)
        self.noise_probability = kwargs.get('noise_probability',0.00035)


    def get_counts(self, qc: QuantumCircuit, shots: int):
        # Create noisy simulator backend
        # A	Amplitude Damping	振幅衰减噪声	非对称退相干（模拟能量弛豫，即 T₁）
        # P	Phase Damping	相位衰减噪声	纯退相干（模拟相干性丧失，即 T₂）
        # D	Depolarizing	退极化噪声	均匀引入 Pauli 错误（X, Y, Z）
        job = self.backend.run(
            qc,
            shots=shots,
            noise_effects="APD",
            noise_probability=self.noise_probability,
            amp_damping_probability=self.amp_damping_probability,
            multi_qubit_gate_factor=25,
        )
        counts = job.result().get_counts()
        return counts

    def get_probabilities(self, qc: QuantumCircuit, shots: int) -> Dict:
        counts = self.get_counts(qc, shots)
        probabilities = {}
        for key, value in counts.items():
            probabilities[key] = value / shots
        return probabilities
    
    # def transpile(self, qc: QuantumCircuit) -> QuantumCircuit:
    #     return transpile(qc, self.transpile_backend)
