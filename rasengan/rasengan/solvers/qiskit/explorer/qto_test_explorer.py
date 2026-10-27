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

import matplotlib.pyplot as plt
import networkx as nx

class QtoTestExplorer(Explorer):
    def __init__(
        self,
        *,
        prb_model: LcboModel,
        feasible_state = None
    ):
        super().__init__(prb_model, None)

        if feasible_state is None:
            self.feasible_state = self.model_option.feasible_state
            iprint("n: ", len(self.feasible_state))
        else:
            self.feasible_state = feasible_state
            iprint("n: ", len(self.feasible_state[0]))

    def explore(self):
        # 计数器，每次取值自动加一
        counter = 0
        # 重复次数
        self.model_option.Hd_bitstr_list = self.model_option.Hd_bitstr_list[::-1]
        self.num_hdi = len(self.model_option.Hd_bitstr_list)
        self.num_repetition = self.num_hdi
        iprint("m:", self.num_hdi)
        

        feasible_state = self.feasible_state
        # 用于记录状态和计数器的对应关系
        dict_state = {}
        # 概率
        dict_prob = {}
        # 由谁分裂而来的
        dict_split = {}

        if isinstance(feasible_state[0], list):
            # prob = 1 / len(feasible_state)
            for state in feasible_state:
                dict_state[tuple(map(int, state))] = (counter, None, -1)
                counter += 1
                dict_prob[tuple(map(int, state))] = 1 / len(feasible_state)
                dict_split[tuple(map(int, state))] = []
        else:
            dict_state[tuple(map(int, feasible_state))] = (counter, None, -1)
            counter += 1
            dict_prob[tuple(map(int, feasible_state))] = 1
            dict_split[tuple(map(int, feasible_state))] = []

        max_order = -1
        # max_shares = 0
        max_split = 0
        longest_train = []
        space_explore = [1, ]
        meaningful_idx = set()
        for order in range(self.num_repetition):
            # 创建字典键的副本
            for idx, hdi in enumerate(self.model_option.Hd_bitstr_list):
                state_list = list(dict_state.keys())
                for state in state_list:
                    preorder_counter = dict_state[state][0]
                    state = list(map(int, state))
                    # 判断是否可以组成新解
                    new_state_add = [int(x) + int(y) for x, y in zip(state, hdi)]
                    new_state_sub = [int(x) - int(y) for x, y in zip(state, hdi)]
                    
                    if all(elem in [0, 1] for elem in new_state_add):
                        if tuple(new_state_add) not in dict_state:
                            meaningful_idx.add(order * self.num_hdi + idx)
                            dict_state[tuple(new_state_add)] = (counter, preorder_counter, order * self.num_hdi + idx)
                            counter += 1
                            max_order = max(max_order, order * self.num_hdi + idx)
                            
                            share_prob = dict_prob[tuple(state)] / 2
                            dict_prob[tuple(new_state_add)] = share_prob
                            dict_prob[tuple(state)] = share_prob

                            split_chain = dict_split[tuple(state)] + [idx]
                            dict_split[tuple(new_state_add)] = split_chain
                            if len(split_chain) > max_split:
                                max_split = len(split_chain)
                                longest_train = split_chain
                        else:
                            share_prob = (dict_prob[tuple(state)] + dict_prob[tuple(new_state_add)]) / 2
                            dict_prob[tuple(new_state_add)] = share_prob
                            dict_prob[tuple(state)] = share_prob

                    if all(elem in [0, 1] for elem in new_state_sub):
                        if tuple(new_state_sub) not in dict_state:
                            meaningful_idx.add(order * self.num_hdi + idx)
                            dict_state[tuple(new_state_sub)] = (counter, preorder_counter, order * self.num_hdi + idx)
                            counter += 1     
                            max_order = max(max_order, order * self.num_hdi + idx)

                            share_prob = dict_prob[tuple(state)] / 2
                            dict_prob[tuple(new_state_sub)] = share_prob
                            dict_prob[tuple(state)] = share_prob

                            split_chain = dict_split[tuple(state)] + [idx]
                            dict_split[tuple(new_state_sub)] = split_chain
                            if len(split_chain) > max_split:
                                max_split = len(split_chain)
                                longest_train = split_chain

                        else:
                            share_prob = (dict_prob[tuple(state)] + dict_prob[tuple(new_state_sub)]) / 2
                            dict_prob[tuple(new_state_sub)] = share_prob
                            dict_prob[tuple(state)] = share_prob
                space_explore.append(len(dict_state))


        self.dict_state = dict_state
        self.max_order = max_order
        self.counter = counter
        import math
        iprint("min_prob: ", min(dict_prob.values()), math.log2(min(dict_prob.values())))
        iprint("max_prob: ", max(dict_prob.values()), math.log2(max(dict_prob.values())))
        iprint("max_split: ", max_split)
        iprint("longest_train: ", longest_train)
        # return space_explore, dict_state, max_order, counter, max_split, longest_train
        return list(meaningful_idx)
    
    def print_generation_relationship(self):
        dict_state = self.dict_state
        num_repetition = len(self.model_option.Hd_bitstr_list)

        # 文字版
        iprint("State and their indices:")
        for state, (index, preorder_counter, order) in dict_state.items():
            iprint(f"State: {state}, Index: {index}, Preorder Counter: {preorder_counter}, Order: {order}")

        iprint("\nGeneration relationships:")
        iprint("Start:")
        for state, (index, preorder_counter, order) in dict_state.items():
            if preorder_counter is None:
                iprint(f"Index {index} generated by {preorder_counter}")

        for order in range(num_repetition):
            iprint(f"\nOrder {order}:")
            for state, (index, preorder_counter, state_order) in dict_state.items():
                if state_order == order and preorder_counter is not None:
                    iprint(f"Index {index} generated by {preorder_counter}")


    def plot_generation_relationship(self, node_size: int = 500, figsize: tuple[int, int] = (12, 8), 
                                    use_absolute_position=False, highlight_nodes=None):
        if highlight_nodes is None:
            highlight_nodes = []
            
        dict_state = self.dict_state

        # 图版本
        G = nx.DiGraph()
        labels = {}
        colors = []
        node_list = []

        for state, (index, preorder_counter, order) in dict_state.items():
            G.add_node(index, layer=order)
            labels[index] = index
            node_list.append(index)
            colors.append('red' if index in highlight_nodes else plt.cm.Blues(order / self.max_order))  # 设定颜色

            if preorder_counter is not None:
                G.add_edge(preorder_counter, index)

        # 使用 multipartite_layout 获取 y 轴值
        pos = nx.multipartite_layout(G, subset_key="layer")
        
        plt.figure(figsize=figsize)
        # 使用 order 作为 x 轴的值，保留 multipartite_layout 的 y 轴值
        if use_absolute_position:
            pos = {index: (order, pos[index][1]) for index, (_, (index, _, order)) in enumerate(dict_state.items())}
            # 添加竖线
            for x_value in range(0, self.num_hdi * self.num_repetition + 1, self.num_hdi):
                plt.axvline(x=x_value - 0.5, color='r', linestyle='--', alpha=0.4)

        # 绘制普通节点
        normal_nodes = [index for index in node_list if index not in highlight_nodes]
        normal_colors = [plt.cm.Blues(self.dict_state[state][2] / self.max_order) for state in dict_state if self.dict_state[state][0] in normal_nodes]
        nx.draw(G, pos, nodelist=normal_nodes, labels=labels, node_color=normal_colors, cmap=plt.cm.Blues, 
                with_labels=True, node_size=node_size, font_size=10, font_color='black', edge_color='gray')

        # 绘制高亮节点
        nx.draw(G, pos, nodelist=highlight_nodes, labels=labels, node_color='red', 
                with_labels=True, node_size=node_size, font_size=10, font_color='black', edge_color='gray')

        plt.title(f"max_order: {self.max_order}, num_node: {self.counter}")
        plt.show()



