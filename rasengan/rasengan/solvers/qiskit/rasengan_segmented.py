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

class RasenganSegmentedCircuit(QiskitCircuit[ChCircuitOption]):
    def __init__(self, circuit_option: ChCircuitOption, model_option: ModelOption, hlist: list[QuantumCircuit], segmentation, do_purify):
        super().__init__(circuit_option, model_option)
        self.inference_circuit = self.create_circuit()
        self.hlist = hlist
        self.segments_list = segmentation[0]   #分割方案 [[1, 2], [3, 4]]
        self.segments_index = segmentation[1]  # 分割的下表索引 [(0, 2), (2, 4)]
        self.do_purify = do_purify

    def get_num_params(self):
        return len(self.hlist)
    
    def inference(self, params):
        counts = self.segmented_excute_circuit(params)
        collapse_state, probs = self.process_counts(counts)
        return collapse_state, probs
    
    def segmented_excute_circuit(self, params) -> QuantumCircuit:
        mcx_mode = self.circuit_option.mcx_mode
        num_qubits = self.model_option.num_qubits

        def run_and_pick(dict:dict, hdi_qc_list: list[QuantumCircuit], params):
            dicts = []
            total_count = sum(dict.values())
            for key, value in dict.items():
                if mcx_mode == "constant":
                    qc_temp = QuantumCircuit(num_qubits + 2, num_qubits)
                elif mcx_mode == "linear":
                    qc_temp = QuantumCircuit(2 * num_qubits, num_qubits)

                for idx, key_i in enumerate(key):
                    if key_i == '1':
                        qc_temp.x(idx)

                for idx, hdi_qc in enumerate(hdi_qc_list):
                    qc_add = hdi_qc.assign_parameters([params[idx]])
                    qc_temp.compose(qc_add, inplace=True)

                qc_temp.measure(range(num_qubits), range(num_qubits)[::-1])
                qc_temp = self.circuit_option.provider.transpile(qc_temp)
                count = self.circuit_option.provider.get_counts_with_time(qc_temp, shots=self.circuit_option.shots * value // total_count)
                dicts.append(count)

            merged_dict = {}
            for d in dicts:
                for key, value in d.items():
                    if not self.do_purify or all([np.dot([int(char) for char in key], constr[:-1]) == constr[-1] for constr in self.model_option.lin_constr_mtx]):
                        merged_dict[key] = merged_dict.get(key, 0) + value
            return merged_dict


        register_counts = {''.join(map(str, self.model_option.feasible_state)): 1}
        for h_tau_list, (start_idx, end_idx) in zip(self.segments_list, self.segments_index):
            register_counts = run_and_pick(register_counts, h_tau_list, params[start_idx: end_idx])

        return register_counts

class RasenganSegmentedSolver(Solver):
    def __init__(
        self,
        *,
        prb_model: LcboModel,
        optimizer: Optimizer,
        provider: Provider,
        num_layers: int = 1,
        shots: int = 1024,
        mcx_mode: str = "constant",
        opt_mode: list = [1, 1, 1, 1], # 是否贪心、是否裁剪、是否分段执行、是否过滤错误
        num_segments = 1,
    ):
        super().__init__(prb_model, optimizer)
        self.do_purify = opt_mode[3]
        # 根据排列理论，直接赋值
        num_layers = len(self.model_option.Hd_bitstr_list)

        if opt_mode[0] == 1:
            # 贪心减少非零元 优化跃迁哈密顿量
            self.model_option.Hd_bitstr_list = greedy_simplification_of_transition_Hamiltonian(self.model_option.Hd_bitstr_list)
        
        self.num_segments = num_segments

        self.circuit_option = ChCircuitOption(
            provider=provider,
            num_layers=num_layers,
            shots=shots,
            mcx_mode=mcx_mode,
        )

        explorer = QtoExplorer(
            prb_model=prb_model,
            optimizer=optimizer,
            provider=provider,
            num_layers=num_layers,
            shots=shots,
            mcx_mode=mcx_mode
        )

        # 编译过的transpiled_hlist \O/
        hlist = explorer.hlist
        
        if opt_mode[1] == 1:
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

            max_id += 1 # 左闭右开+1
        else:
            min_id = 0
            max_id = num_layers ** 2
        iprint(f'range({min_id}, {max_id})')

        if opt_mode[2] == 1:
            num_segments = min(max_id - min_id, num_segments)
        else:
            num_segments = 1
        
        # 保证自定义段数 可以被满足
        assert num_segments >= 1 and num_segments <= max_id - min_id

        def split_list_into_segments(hlist, num_segments):
            n = len(hlist)
            # 每段的基本长度
            segment_length = n // num_segments
            # 剩余的元素数量
            remainder = n % num_segments
            
            segments = []
            index_list = []
            start = 0

            for i in range(num_segments):
                # 如果当前段应该多分配一个元素（在余数范围内）
                end = start + segment_length + (1 if i < remainder else 0)
                segments.append(hlist[start:end])
                index_list.append((start, end))
                start = end
            
            return segments, index_list

        self.hlist = []
        hlist_len = len(hlist)
        for i in range(min_id, max_id):
            self.hlist.append(hlist[i % hlist_len])

        self.segmentation = split_list_into_segments(self.hlist, num_segments)


    @property
    def circuit(self):
        if self._circuit is None:
            self._circuit = RasenganSegmentedCircuit(self.circuit_option, self.model_option, self.hlist, self.segmentation, self.do_purify)
        return self._circuit


