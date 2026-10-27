import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter

from rasengan.solvers.abstract_solver import Solver
from rasengan.solvers.optimizers import Optimizer
from rasengan.solvers.options import CircuitOption, OptimizerOption, ModelOption
from rasengan.solvers.options.circuit_option import ChCircuitOption
from rasengan.model import LinearConstrainedBinaryOptimization as LcboModel
from rasengan.utils import iprint
from rasengan.utils.linear_system import to_row_echelon_form, greedy_simplification_of_transition_Hamiltonian
from .circuit import QiskitCircuit
from .provider import Provider
from .circuit.circuit_components import obj_compnt, new_compnt
from .explorer.qto_explorer import QtoExplorer

class RasenganCircuit(QiskitCircuit[ChCircuitOption]):
    def __init__(self, circuit_option: ChCircuitOption, model_option: ModelOption):
        super().__init__(circuit_option, model_option)
        self.inference_circuit = self.create_circuit()

    def get_num_params(self):
        return len(self.model_option.Hd_bitstr_list)
    
    def inference(self, params):
        final_qc = self.inference_circuit.assign_parameters(params)
        counts = self.circuit_option.provider.get_counts_with_time(final_qc, shots=self.circuit_option.shots)
        collapse_state, probs = self.process_counts(counts)
        return collapse_state, probs

    def create_circuit(self) -> QuantumCircuit:
        
        mcx_mode = self.circuit_option.mcx_mode
        num_layers = self.circuit_option.num_layers
        num_qubits = self.model_option.num_qubits
        
        if mcx_mode == "constant":
            qc = QuantumCircuit(num_qubits + 2, num_qubits)
            anc_idx = [num_qubits, num_qubits + 1]
        elif mcx_mode == "linear":
            qc = QuantumCircuit(2 * num_qubits, num_qubits)
            anc_idx = list(range(num_qubits, 2 * num_qubits))

        qc = self.circuit_option.provider.transpile(qc)
        
        num_bitstrs = len(self.model_option.Hd_bitstr_list)
        Hd_params_lst = [Parameter(f"Hd_params[{i}, {j}]") for j in range(num_bitstrs) for i in range(num_layers)]

        for i in np.nonzero(self.model_option.feasible_state)[0]:
            qc.x(i)

        new_compnt(
            qc,
            Hd_params_lst,
            self.model_option.Hd_bitstr_list,
            anc_idx,
            mcx_mode,
        )
        qc.measure(range(num_qubits), range(num_qubits)[::-1])
        transpiled_qc = self.circuit_option.provider.transpile(qc)
        return transpiled_qc

class RasenganSolver(Solver):
    def __init__(
        self,
        *,
        prb_model: LcboModel,
        optimizer: Optimizer,
        provider: Provider,
        num_layers: int = 1,
        shots: int = 1024,
        opt_mode: list = [1, 1], # 是否贪心、是否裁剪
        mcx_mode: str = "constant",
    ):
        super().__init__(prb_model, optimizer)
        # 根据排列理论，直接赋值
        num_layers = len(self.model_option.Hd_bitstr_list)
        # 贪心减少非零元 优化跃迁哈密顿量
        if opt_mode[0] == 1:
            self.model_option.Hd_bitstr_list = greedy_simplification_of_transition_Hamiltonian(self.model_option.Hd_bitstr_list)
        
        self.circuit_option = ChCircuitOption(
            provider=provider,
            num_layers=num_layers,
            shots=shots,
            mcx_mode=mcx_mode,
        )

        if opt_mode[1] == 1:
            explorer = QtoExplorer(
            # explorer = QtoSearchFastSolver(
                prb_model=prb_model,
                optimizer=optimizer,
                provider=provider,
                num_layers=num_layers,
                shots=shots,
                mcx_mode=mcx_mode
            )
            min_id = 0
            max_id = -1

            # ***逐个测试方案***
            _, set_basis_lists, _ = explorer.explore_with_time()
            useful_idx = []
            already_set = set()
            if len(set_basis_lists[0]) != 1:
                useful_idx.append(0)

            already_set.update(set_basis_lists[0])
            for i in range(1, len(set_basis_lists)):
                if len(set_basis_lists[i - 1]) == 1 and min_id == i - 1:
                    min_id = i
                if set_basis_lists[i] - already_set:
                    already_set.update(set_basis_lists[i])
                    max_id = i

            # ***逐层搜索+回溯方案***
            # max_id = explorer.search()

            max_id += 1 # 左闭右开+1
            iprint(f'range({min_id}, {max_id})')
            Hd_bitstr_list = np.tile(self.model_option.Hd_bitstr_list, (num_layers, 1))
            self.model_option.Hd_bitstr_list = [item for i, item in enumerate(Hd_bitstr_list) if i >= min_id and i < max_id]
        else:
            self.model_option.Hd_bitstr_list = np.tile(self.model_option.Hd_bitstr_list, (num_layers, 1))


    @property
    def circuit(self):
        if self._circuit is None:
            self._circuit = RasenganCircuit(self.circuit_option, self.model_option)
        return self._circuit


