from src.qiwrap import quantumLayer

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp, Pauli

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .model import Model

class VqcBase(Model):
    def __init__(self, input_dim, input_params, circuit, circuit_params, observables, output_dim, config) -> None:
        super().__init__()

        self.circuit = circuit
        self.observables = observables
        
        self.model = quantumLayer.torch_layer(
            qc = self.circuit, 
            weight_params = circuit_params, 
            input_params = input_params, 
            batch_size = config["batch_size"],
            data_reuploading_layers = config["data_reuploading_layers"],
            data_reuploading_type = config["data_reuploading_type"],
            observables = observables, 
            grad_type = config["grad_type"],
            epsilon = config["epsilon"],
            quantum_compute_method = config["quantum_compute_method"],
            quantum_weight_initialization = config["quantum_weight_initialization"],
            output_dim = output_dim,
            input_dim = input_dim,
            add_input_weights = config["add_input_weights"],
            add_output_weights = config["add_output_weights"],
            input_scaling = config["input_scaling"],
            output_scaling = config["output_scaling"],
            input_weight_initialization = config["input_weight_initialization"],
            output_weight_initialization = config["output_weight_initialization"],
            post_process_decode = config["post_process_decode"],
            num_parallel_process = config["num_parallel_process"],
            shots = config["shots"],
            ibmq_session_max_time = config["ibmq_session_max_time"],
            fix_parameter_indices = config.get("fix_parameter_indices", None)
            )
        
        self.model.set_session()
        self.model = self.model.float()

    def load_pretrained(self, filename:str, load_index_dict:dict):
        """ Load pre-trained weights from a torch checkpoint.
        Args
        filename            Path to the checkpoint.pth
        load_parameter_dict Dictionary with parameter name as key and index of parameter as value.
        """
        raise NotImplementedError(f"load_pretrained() not implemented for {self}.")
    