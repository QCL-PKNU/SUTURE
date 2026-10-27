import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter

from rasengan.solvers.abstract_explorer import Explorer
from rasengan.solvers.optimizers import Optimizer
from rasengan.solvers.options import CircuitOption, OptimizerOption, ModelOption
from rasengan.solvers.options.circuit_option import ChCircuitOption
from rasengan.model import LinearConstrainedBinaryOptimization as LcboModel
from rasengan.utils.linear_system import to_row_echelon_form, greedy_simplification_of_transition_Hamiltonian
from rasengan.utils.gadget import iprint
from ..circuit import QiskitCircuit
from ..provider import Provider
from ..circuit.circuit_components import obj_compnt, search_evolution_space_by_hdi_bitstr
from ..circuit.hdi_decompose import driver_component


class QtoExplorerCircuit(QiskitCircuit[ChCircuitOption]):
    def __init__(self, circuit_option: ChCircuitOption, model_option: ModelOption):
        super().__init__(circuit_option, model_option)
        iprint(self.model_option.Hd_bitstr_list)
        self._transpiled_hlist = self.transpile_hlist()
        self._hlist = self.hlist()
        self.result = self.inference()

    def get_num_params(self):
        return self.circuit_option.num_layers * 2

    def transpile_hlist(self):
        mcx_mode = self.circuit_option.mcx_mode
        num_qubits = self.model_option.num_qubits
        if mcx_mode == "constant":
            qc = QuantumCircuit(num_qubits + 2, num_qubits)
            anc_idx = [num_qubits, num_qubits + 1]
        elif mcx_mode == "linear":
            qc = QuantumCircuit(2 * num_qubits, num_qubits)
            anc_idx = list(range(num_qubits, 2 * num_qubits))
        self.qc = qc
        

        transpiled_hlist = []
        for hdi_vct in self.model_option.Hd_bitstr_list:
            qc_temp: QuantumCircuit = qc.copy()
            nonzero_indices = np.nonzero(hdi_vct)[0].tolist()
            hdi_bitstr = [0 if x == -1 else 1 for x in hdi_vct if x != 0]
            H_param = Parameter("1")
            driver_component(qc_temp, nonzero_indices, anc_idx, hdi_bitstr, H_param, mcx_mode)
            transpiled_qc = self.circuit_option.provider.transpile(qc_temp)
            transpiled_hlist.append(transpiled_qc)
            
        return transpiled_hlist

    def hlist(self):
        """没有经过编译的逻辑电路

        Returns:
            list: 逻辑电路列表
        """
        mcx_mode = self.circuit_option.mcx_mode
        num_qubits = self.model_option.num_qubits
        if mcx_mode == "constant":
            qc = QuantumCircuit(num_qubits + 2, num_qubits)
            anc_idx = [num_qubits, num_qubits + 1]
        elif mcx_mode == "linear":
            qc = QuantumCircuit(2 * num_qubits, num_qubits)
            anc_idx = list(range(num_qubits, 2 * num_qubits))
        self.qc = qc
        

        hlist = []
        for hdi_vct in self.model_option.Hd_bitstr_list:
            qc_temp: QuantumCircuit = qc.copy()
            nonzero_indices = np.nonzero(hdi_vct)[0].tolist()
            hdi_bitstr = [0 if x == -1 else 1 for x in hdi_vct if x != 0]
            H_param = Parameter("1")
            driver_component(qc_temp, nonzero_indices, anc_idx, hdi_bitstr, H_param, mcx_mode)
            hlist.append(qc_temp)
            
        return hlist
    
    def inference(self):
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
        # Ho_params = np.random.rand(num_layers)

        for i in np.nonzero(self.model_option.feasible_state)[0]:
            qc.x(i)
        num_basis_lists = []
        set_basis_lists = []
        depth_lists = []
        already_set = set()
        for layer in range(num_layers):
            # Hd_params = np.full(num_layers, np.random.uniform(0.1, np.pi / 4 - 0.1))
            Hd_params = np.full(num_layers, np.pi / 4)
            iprint(f"===== times of repetition: {layer + 1} ======")
            num_basis_list, set_basis_list, depth_list = search_evolution_space_by_hdi_bitstr(
                qc,
                Hd_params,
                # self.transpiled_hlist, # low_cost 
                self.model_option.Hd_bitstr_list, # high_cost
                anc_idx,
                mcx_mode,
                num_qubits,
                self.circuit_option.shots,
                self.circuit_option.provider,
                already_set,
            )
            num_basis_lists.extend(num_basis_list)
            set_basis_lists.extend(set_basis_list)
            depth_lists.extend(depth_list)
            this_time = set.union(*set_basis_list)
            iprint(num_basis_list)
            # 早停
            if this_time - already_set:
                already_set.update(this_time)
            else:
                break
            
        return num_basis_lists, set_basis_lists, depth_lists


class QtoExplorer(Explorer):
    def __init__(
        self,
        *,
        prb_model: LcboModel,
        optimizer: Optimizer,
        provider: Provider,
        shots: int = 1024,
        mcx_mode: str = "constant",
        num_layers: int = 1,
    ):
        super().__init__(prb_model, optimizer)
        # m 层
        num_layers = len(self.model_option.Hd_bitstr_list)

        # need to refix
        from rasengan.solvers.qiskit import DdsimProvider

        self.original_provider = provider
        self.explore_provider = DdsimProvider()
        self.circuit_option = ChCircuitOption(
            provider=self.explore_provider,
            num_layers=num_layers,
            shots=shots * 100,
            mcx_mode=mcx_mode,
        )

    @property
    def circuit(self):
        if self._circuit is None:
            self._circuit = QtoExplorerCircuit(self.circuit_option, self.model_option)
        return self._circuit

    def explore(self):
        return self.circuit.result
    
    @property
    def transpiled_hlist(self):
        return self.circuit._transpiled_hlist
    

    @property
    def hlist(self):
        return self.circuit._hlist
